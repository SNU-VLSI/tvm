"""
Joint Node Mapping + Routing ILP for IMCFlow

This module implements a joint ILP solver that simultaneously optimizes:
1. Node placement (mapping call nodes to IMCE hardware)
2. Routing (finding NoC paths for data flows)

This approach guarantees global optimum (if feasible) unlike sequential
approaches (mapping then routing) which may fail if initial mapping
makes routing impossible.

Usage:
    from joint_pnr_ilp import JointPnRILP

    solver = JointPnRILP()
    result = solver.run(mod)  # Per imcflow function
    # result.mapping: Dict[graph_node_id -> Coord]
    # result.routes: Dict[commodity_id -> List[Edge]]
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
import logging
import os
import re

import pulp
import tvm
from tvm import relay
from tvm.relay import op
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, TensorID, NodeID
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.backend.contrib.imcflow.transform_utils import *

logger = logging.getLogger(__name__)

# ============================================================
# Data Structures
# ============================================================

class Direction(Enum):
    """Direction in 2D mesh NoC"""
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    LOCAL = "Local"


@dataclass
class Coord:
    """2D coordinate in mesh"""
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if isinstance(other, Coord):
            return self.row == other.row and self.col == other.col
        return False

    def __repr__(self):
        return f"({self.row}, {self.col})"


@dataclass
class Edge:
    """Directed edge in NoC mesh"""
    src: Coord
    dst: Coord

    def __hash__(self):
        return hash((self.src, self.dst))

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.src == other.src and self.dst == other.dst
        return False

    def __repr__(self):
        return f"{self.src} -> {self.dst}"

    def get_direction(self) -> Direction:
        """Get direction of this edge"""
        if self.src.col < self.dst.col:
            return Direction.EAST
        elif self.src.col > self.dst.col:
            return Direction.WEST
        elif self.src.row < self.dst.row:
            return Direction.SOUTH
        elif self.src.row > self.dst.row:
            return Direction.NORTH
        return Direction.LOCAL


@dataclass
class HWCommodity:
    """Hardware commodity representing a data flow from source to destination.

    This replaces MCFCommodity for policy table generation.
    Used after Joint PnR to create commodities from noc_paths (which includes
    both TensorEdge paths and instruction paths).
    """
    id: int
    source: Coord
    destination: Coord
    metadata: Any = None  # (edge, mapping_info) where edge is TensorEdge or NodeID

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"HWCommodity({self.id}, {self.source} -> {self.destination})"


@dataclass
class RoutingResult:
    """Result of routing a single commodity"""
    commodity: HWCommodity
    path: List[Coord]  # List of coordinates from source to destination
    edges: List[Edge]  # List of edges used

    def get_path_length(self) -> int:
        return len(self.edges)


@dataclass
class BaseRoutingResult:
    """Base class for routing results from any router implementation.

    All router implementations should return a result that inherits from this class.
    Provides common interface and helper methods for Phase 2 (Tree Builder).
    """
    routes: Dict[int, RoutingResult]  # commodity_id -> routing result

    def get_path(self, commodity_id: int) -> List[Coord]:
        """Get the path for a specific commodity."""
        if commodity_id not in self.routes:
            raise KeyError(f"Commodity {commodity_id} not found in routes")
        return self.routes[commodity_id].path

    def get_edges(self, commodity_id: int) -> List[Edge]:
        """Get the edges used by a specific commodity."""
        if commodity_id not in self.routes:
            raise KeyError(f"Commodity {commodity_id} not found in routes")
        return self.routes[commodity_id].edges

    def get_commodity(self, commodity_id: int) -> HWCommodity:
        """Get the commodity object by id."""
        if commodity_id not in self.routes:
            raise KeyError(f"Commodity {commodity_id} not found in routes")
        return self.routes[commodity_id].commodity

    def get_all_commodity_ids(self) -> List[int]:
        """Get all commodity ids."""
        return list(self.routes.keys())


@dataclass
class MeshTopology:
    """2D Mesh NoC topology"""
    rows: int
    cols: int

    def get_all_nodes(self) -> List[Coord]:
        """Get all nodes in the mesh"""
        return [Coord(r, c) for r in range(self.rows) for c in range(self.cols)]

    def get_inode_nodes(self) -> List[Coord]:
        """Get INODE nodes (column 0)"""
        return [Coord(r, 0) for r in range(self.rows)]

    def get_imce_nodes(self) -> List[Coord]:
        """Get IMCE nodes (columns 1-4)"""
        return [Coord(r, c) for r in range(self.rows) for c in range(1, self.cols)]

    def get_var_inodes(self) -> List[Coord]:
        """Get INODEs for input variables (rows 0, 1)"""
        return [Coord(0, 0), Coord(1, 0)]

    def get_funcout_inodes(self) -> List[Coord]:
        """Get INODEs for function outputs (rows 2, 3)"""
        return [Coord(2, 0), Coord(3, 0)]

    def get_neighbors(self, coord: Coord) -> List[Tuple[Coord, Direction]]:
        """Get valid neighbors of a node with their directions"""
        neighbors = []
        # North
        if coord.row > 0:
            neighbors.append((Coord(coord.row - 1, coord.col), Direction.NORTH))
        # South
        if coord.row < self.rows - 1:
            neighbors.append((Coord(coord.row + 1, coord.col), Direction.SOUTH))
        # West
        if coord.col > 0:
            neighbors.append((Coord(coord.row, coord.col - 1), Direction.WEST))
        # East
        if coord.col < self.cols - 1:
            neighbors.append((Coord(coord.row, coord.col + 1), Direction.EAST))
        return neighbors

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if coordinate is within mesh bounds"""
        return 0 <= coord.row < self.rows and 0 <= coord.col < self.cols

    def get_all_edges(self) -> List[Edge]:
        """Get all directed edges in the mesh"""
        edges = []
        for node in self.get_all_nodes():
            for neighbor, _direction in self.get_neighbors(node):
                edges.append(Edge(node, neighbor))
        return edges


class NodeType(Enum):
    """Type of graph node for placement constraints"""
    VAR = "var"           # Input variable -> INODE 0,1
    FUNC_OUT = "func_out" # Function output -> INODE 2,3
    CONST = "const"       # Constant -> same row INODE as consumer
    SPLIT = "split"       # Split -> same as producer
    CONCAT = "concat"     # Concat -> same as last producer (topological order)
    CALL = "call"         # Regular call -> IMCE


@dataclass
class GraphNode:
    """Represents a node in the relay graph for placement"""
    id: Any                    # graph_node_id (relay expr or custom ID)
    node_type: NodeType
    relay_expr: Any = None     # Original relay expression
    producer: Any = None       # For split: producer node id
    last_producer: Any = None  # For concat: last producer in topo order
    consumers: List[Any] = field(default_factory=list)
    topo_order: int = 0        # Topological order for determining last producer


@dataclass
class Commodity:
    """A data flow from source to destination"""
    id: int
    source_node_id: Any     # graph_node_id of source
    dest_node_id: Any       # graph_node_id of destination
    source_type: NodeType   # Type of source node
    dest_type: NodeType     # Type of destination node
    tensor_type: str        # e.g., 'data', 'weight', 'config'
    split_idx: Optional[int] = None
    metadata: Any = None    # Original TensorEdge for reference
    is_multicast: bool = True  # False for split outputs that need individual routing

    def get_congestion_group(self) -> str:
        """Get congestion group for this commodity"""
        CONST_TYPES = frozenset([
            'weight', 'config', 'min', 'max', 'fused_scale', 'fused_bias',
            'bias', 'scale', 'threshold'
        ])
        DATA_TYPES = frozenset([
            'odata', 'data', 'lhs', 'rhs', 'var'
        ])

        # Check for func_out (destination is function)
        if self.tensor_type and self.tensor_type.startswith('func_out'):
            return 'data'

        if self.tensor_type in CONST_TYPES:
            return 'const'
        elif self.tensor_type in DATA_TYPES:
            return 'data'
        else:
            return 'data'  # Default to data (more conservative)


@dataclass
class GraphInfo:
    """Extracted graph information for ILP"""
    nodes: Dict[Any, GraphNode]       # graph_node_id -> GraphNode
    commodities: List[Commodity]      # All data flows
    call_nodes: List[Any]             # IDs of call nodes needing IMCE
    split_nodes: List[Any]            # IDs of split nodes
    concat_nodes: List[Any]           # IDs of concat nodes
    var_nodes: List[Any]              # IDs of var nodes
    const_nodes: List[Any]            # IDs of const nodes
    funcout_nodes: List[Any]          # IDs of func_out nodes
    var_to_inode: Dict[Any, Coord] = None      # var node_id -> INODE Coord
    funcout_to_inode: Dict[Any, Coord] = None  # funcout node_id -> INODE Coord
    const_to_inode: Dict[Any, Coord] = None    # const node_id -> INODE Coord
    tensor_edge_to_commodity_id: Dict[Any, int] = None  # TensorEdge -> commodity_id


@dataclass
class JointPnRResult:
    """Result of joint place and route"""
    mapping: Dict[Any, Coord]           # graph_node_id -> Coord (IMCE mappings)
    routes: Dict[int, List[Edge]]       # commodity_id -> list of edges
    commodities: List[Commodity]
    max_congestion: int
    total_hops: int
    solver_status: str
    success: bool = True
    var_to_inode: Dict[Any, Coord] = None      # var node_id -> INODE Coord
    funcout_to_inode: Dict[Any, Coord] = None  # funcout node_id -> INODE Coord
    const_to_inode: Dict[Any, Coord] = None    # const node_id -> INODE Coord
    tensor_edge_to_commodity_id: Dict[Any, int] = None  # TensorEdge -> commodity_id


# ============================================================
# TensorEdgeList → GraphInfo Conversion
# ============================================================

# Constants for node type inference from tensor_type tags
CONST_TENSOR_TYPES = frozenset([
    'weight', 'config', 'bias', 'scale', 'fused_scale', 'fused_bias',
    'min', 'max', 'threshold'
])


def getInnerNodeID(node):
    """Handle composite nodes where graph_node_id is tuple (outer_id, inner_id).

    Composite nodes have a tuple graph_node_id where:
    - First element: outer (wrapper) node ID
    - Second element: inner (actual op) node ID

    For non-composite nodes, return the node as-is.
    """
    if isinstance(node, tuple):
        return node[1]
    else:
        return node


def getOuterNodeID(node):
    """Get outer node ID for composite nodes.

    For composite nodes (tuple), returns the outer (wrapper) node ID.
    This is the ID that GraphExtractor uses for the composite as a whole.

    Examples:
        (41, 33) → 41   (composite call)
        (41, -17) → 41  (composite internal constant)
        40 → 40         (regular node)

    For non-composite nodes, returns the node as-is.
    """
    if isinstance(node, tuple):
        return node[0]
    else:
        return node


def parse_funcout_tuple_idx(tensor_type: str) -> int:
    """Parse tuple index from func_out tensor type.

    Examples:
        'func_out0' -> 0
        'func_out1' -> 1
        'func_out' -> 0 (default)
    """
    if not tensor_type.startswith('func_out'):
        raise ValueError(f"Not a func_out tensor type: {tensor_type}")
    idx_str = tensor_type[8:]  # Remove 'func_out' prefix
    return int(idx_str) if idx_str else 0


def _check_split_concat_or_call(graph_node_id: Any) -> NodeType:
    """Check if node is split, concat, or regular call using CustomIDToNode.

    Uses the custom_id stored in graph_node_id to look up the actual relay node
    and determine if it's a split or concatenate operation.

    Args:
        graph_node_id: The graph_node_id from TensorID (may be tuple for composite)

    Returns:
        NodeType.SPLIT, NodeType.CONCAT, or NodeType.CALL
    """
    try:
        inner_id = getInnerNodeID(graph_node_id)
        node = CustomIDToNode().get(inner_id)
        if node is None:
            return NodeType.CALL

        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == "split":
                return NodeType.SPLIT
            elif node.op.name == "concatenate":
                return NodeType.CONCAT
        return NodeType.CALL
    except (KeyError, AttributeError, TypeError):
        return NodeType.CALL  # Default to CALL if lookup fails


def infer_node_type(graph_node_id: Any, tensor_type: str, is_source: bool) -> NodeType:
    """
    Infer node type from TensorID.tensor_type tag and CustomIDToNode lookup.

    This function first checks tensor_type for VAR, CONST, and FUNC_OUT.
    For other types (like 'odata'), it uses CustomIDToNode to detect split/concat.

    Args:
        graph_node_id: The graph_node_id from TensorID
        tensor_type: The tensor_type string from TensorID
        is_source: True if this is the source node, False if destination

    Returns:
        NodeType enum value
    """
    if is_source:
        if tensor_type == 'var':
            return NodeType.VAR
        elif tensor_type in CONST_TENSOR_TYPES:
            return NodeType.CONST
        else:  # 'odata', 'data', etc. - need to check for split/concat
            return _check_split_concat_or_call(graph_node_id)
    else:  # destination
        if tensor_type.startswith('func_out'):
            return NodeType.FUNC_OUT
        else:  # 'data', 'lhs', 'rhs', etc. - need to check for split/concat
            return _check_split_concat_or_call(graph_node_id)
class GraphExtractor:
    """Extract graph structure from relay function for ILP"""

    def __init__(self):
        self.nodes: Dict[Any, GraphNode] = {}
        self.commodities: List[Commodity] = []
        self.topo_order = 0
        self.commodity_id = 0

        # Track node relationships
        self.node_to_producers: Dict[Any, List[Any]] = {}  # node -> [producer nodes]
        self.use_def_chain: Dict[Any, List[Any]] = {}      # expr -> [users]

        # Var assignment tracking
        self.var_inode_iter = None
        self.funcout_inode_iter = None
        self.var_to_inode: Dict[Any, Coord] = {}
        self.funcout_to_inode: Dict[Any, Coord] = {}
        self.const_to_inode: Dict[Any, Coord] = {}  # const inner_id -> INODE Coord
        self.tensor_edge_to_commodity_id: Dict[Any, int] = {}  # TensorEdge -> commodity_id

    def extract(self, func: relay.Function, func_name: str,
                tensor_edge_list: List = None) -> GraphInfo:
        """Extract graph info from relay function.

        Args:
            func: Relay function to extract from
            func_name: Name of the function
            tensor_edge_list: Optional list of TensorEdge for commodity extraction.
                              If provided, uses TensorEdgeList-based extraction (robust).
                              If None, falls back to relay parsing (legacy).

        Returns:
            GraphInfo with nodes, commodities, and categorized node lists
        """
        # Reset state
        self.nodes = {}
        self.graph_commodities = []
        self.topo_order = 0
        self.commodity_id = 0
        self.node_to_producers = {}
        self.use_def_chain = {}

        # Initialize INODE iterators
        self.var_inode_iter = cycle([Coord(0, 0), Coord(1, 0)])
        self.funcout_inode_iter = cycle([Coord(3, 0), Coord(2, 0)])
        self.var_to_inode = {}
        self.funcout_to_inode = {}
        self.const_to_inode = {}

        # Build use-def chain first
        self._build_use_def_chain(func)

        # Visit function to extract nodes
        self._visit(func)

        for expr, node in self.var_to_inode.items():
            debug_print(f"[GraphExtractor] Var node ID: {getNodeID(expr)}, assigned INODE: {node}")
        
        for expr, node in self.funcout_to_inode.items():
            debug_print(f"[GraphExtractor] FuncOut node ID: {getNodeID(expr)}, assigned INODE: {node}")

        for node_id, node in self.nodes.items():
            debug_print(f"[GraphExtractor] Node ID: {node_id}, Type: {node.node_type}, Topo Order: {node.topo_order}")
            if node.node_type == NodeType.FUNC_OUT:
              debug_print(f"  Name : {CustomIDToName()[getOuterNodeID(node_id)]}")
            else:
              debug_print(f"  Name : {CustomIDToName()[getInnerNodeID(node_id)]}")

        # Extract graph_commodities (data flows)
        self._extract_graph_commodities_from_tensor_edge_list(tensor_edge_list, func_name)

        # Collect nodes by type
        call_nodes = []
        split_nodes = []
        concat_nodes = []
        var_nodes = []
        const_nodes = []
        funcout_nodes = []

        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.CALL:
                call_nodes.append(node_id)
            elif node.node_type == NodeType.SPLIT:
                split_nodes.append(node_id)
            elif node.node_type == NodeType.CONCAT:
                concat_nodes.append(node_id)
            elif node.node_type == NodeType.VAR:
                var_nodes.append(node_id)
            elif node.node_type == NodeType.CONST:
                const_nodes.append(node_id)
            elif node.node_type == NodeType.FUNC_OUT:
                funcout_nodes.append(node_id)

        debug_print(f"[GraphExtractor] Extracted: {len(call_nodes)} calls, "
                   f"{len(split_nodes)} splits, {len(concat_nodes)} concats, "
                   f"{len(var_nodes)} vars, {len(const_nodes)} consts, "
                   f"{len(funcout_nodes)} func_outs, "
                   f"{len(self.commodities)} commodities")

        return GraphInfo(
            nodes=self.nodes,
            commodities=self.commodities,
            call_nodes=call_nodes,
            split_nodes=split_nodes,
            concat_nodes=concat_nodes,
            var_nodes=var_nodes,
            const_nodes=const_nodes,
            funcout_nodes=funcout_nodes,
            var_to_inode=self.var_to_inode.copy(),
            funcout_to_inode=self.funcout_to_inode.copy(),
            const_to_inode=self.const_to_inode.copy(),
            tensor_edge_to_commodity_id=self.tensor_edge_to_commodity_id.copy(),
        )

    def _build_use_def_chain(self, func):
        """Build use-def chain: expr -> [users]"""
        class UseDefVisitor(relay.ExprVisitor):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor

            def visit_call(self, call):
                for arg in call.args:
                    if arg not in self.extractor.use_def_chain:
                        self.extractor.use_def_chain[arg] = []
                    self.extractor.use_def_chain[arg].append(call)
                    self.visit(arg)
                if isinstance(call.op, relay.Function):
                    self.visit(call.op)

            def visit_tuple(self, tup):
                for field in tup.fields:
                    if field not in self.extractor.use_def_chain:
                        self.extractor.use_def_chain[field] = []
                    self.extractor.use_def_chain[field].append(tup)
                    self.visit(field)

            def visit_tuple_getitem(self, tgi):
                if tgi.tuple_value not in self.extractor.use_def_chain:
                    self.extractor.use_def_chain[tgi.tuple_value] = []
                self.extractor.use_def_chain[tgi.tuple_value].append(tgi)
                self.visit(tgi.tuple_value)

        UseDefVisitor(self).visit(func)

    def _get_node_id(self, expr) -> Any:
        """Get unique node ID for an expression.

        For Call/Function nodes with custom_id attr, use custom_id (matches TensorEdgeList IDs).
        For other nodes, fallback to hash (for internal consistency).
        """
        return getNodeID(expr)

    def _visit(self, expr, in_composite=False, composite_node_id=None):
        """Visit expression and extract nodes/commodities"""
        if isinstance(expr, relay.Function):
            self._visit_function(expr, in_composite, composite_node_id)
        elif isinstance(expr, relay.Var):
            self._visit_var(expr, in_composite)
        elif isinstance(expr, relay.Constant):
            self._visit_constant(expr, in_composite)
        elif isinstance(expr, relay.Call):
            self._visit_call(expr, in_composite, composite_node_id)
        elif isinstance(expr, relay.Tuple):
            for field in expr.fields:
                self._visit(field, in_composite, composite_node_id)
        elif isinstance(expr, relay.TupleGetItem):
            self._visit(expr.tuple_value, in_composite, composite_node_id)

    def _visit_function(self, fn, in_composite, composite_node_id):
        """Visit function node"""
        node_id = self._get_node_id(fn)

        if not in_composite:
            # Top-level function -> func_out
            # Check if function returns a tuple (multiple outputs)
            if isinstance(fn.body, relay.Tuple):
                # Multiple outputs - create separate FUNC_OUT node for each tuple element
                for tuple_idx in range(len(fn.body.fields)):
                    funcout_key = (node_id, tuple_idx)
                    self.topo_order += 1
                    self.nodes[funcout_key] = GraphNode(
                        id=funcout_key,
                        node_type=NodeType.FUNC_OUT,
                        relay_expr=fn,
                        topo_order=self.topo_order,
                    )
                    # Assign each tuple output to a different INODE
                    if funcout_key not in self.funcout_to_inode:
                        self.funcout_to_inode[funcout_key] = next(self.funcout_inode_iter)
            else:
                # Single output - use (node_id, 0) for consistency with TensorEdgeList format
                funcout_key = (node_id, 0)
                self.topo_order += 1
                self.nodes[funcout_key] = GraphNode(
                    id=funcout_key,
                    node_type=NodeType.FUNC_OUT,
                    relay_expr=fn,
                    topo_order=self.topo_order,
                )
                if funcout_key not in self.funcout_to_inode:
                    self.funcout_to_inode[funcout_key] = next(self.funcout_inode_iter)

        # Visit function body
        self._visit(fn.body, in_composite, composite_node_id or node_id)

    def _visit_var(self, var, in_composite):
        """Visit variable node"""
        if in_composite:
            return  # Skip vars inside composite

        node_id = self._get_node_id(var)
        self.topo_order += 1
        self.nodes[node_id] = GraphNode(
            id=node_id,
            node_type=NodeType.VAR,
            relay_expr=var,
            topo_order=self.topo_order,
        )
        # Assign to funcin INODE
        if node_id not in self.var_to_inode:
            self.var_to_inode[node_id] = next(self.var_inode_iter)

    def _visit_constant(self, const, in_composite):
        """Visit constant node"""
        node_id = self._get_node_id(const)
        self.topo_order += 1
        self.nodes[node_id] = GraphNode(
            id=node_id,
            node_type=NodeType.CONST,
            relay_expr=const,
            topo_order=self.topo_order,
        )

    def _visit_call(self, call, in_composite, composite_node_id):
        """Visit call node"""
        # Visit args first (post-order)
        for arg in call.args:
            self._visit(arg, in_composite, composite_node_id)

        node_id = self._get_node_id(call)

        if in_composite:
            # Inside composite - node is mapped to composite's location
            return

        # Check if this is split/concat
        is_split = isinstance(call.op, tvm.ir.Op) and call.op.name == "split"
        is_concat = isinstance(call.op, tvm.ir.Op) and call.op.name == "concatenate"
        is_composite = (isinstance(call.op, relay.Function) and
                       hasattr(call.op.attrs, "Composite") and
                       call.op.attrs["Composite"] and
                       re.match(r"imcflow.*", str(call.op.attrs["Composite"])))

        self.topo_order += 1

        if is_split:
            # Split: same location as producer
            producer_id = self._get_node_id(call.args[0])
            self.nodes[node_id] = GraphNode(
                id=node_id,
                node_type=NodeType.SPLIT,
                relay_expr=call,
                producer=producer_id,
                topo_order=self.topo_order,
            )
        elif is_concat:
            # Concat: same location as last producer (topological order)
            # Find last producer among concat inputs
            inputs = call.args[0].fields if isinstance(call.args[0], relay.Tuple) else [call.args[0]]
            last_producer_id = None
            max_topo = -1
            for inp in inputs:
                inp_id = self._get_node_id(inp)
                if inp_id in self.nodes and self.nodes[inp_id].topo_order > max_topo:
                    max_topo = self.nodes[inp_id].topo_order
                    last_producer_id = inp_id

            self.nodes[node_id] = GraphNode(
                id=node_id,
                node_type=NodeType.CONCAT,
                relay_expr=call,
                last_producer=last_producer_id,
                topo_order=self.topo_order,
            )
        else:
            # Regular call -> needs IMCE
            self.nodes[node_id] = GraphNode(
                id=node_id,
                node_type=NodeType.CALL,
                relay_expr=call,
                topo_order=self.topo_order,
            )

        # Handle composite function
        if is_composite:
            self._visit(call.op, in_composite=True, composite_node_id=node_id)

    def _extract_graph_commodities_from_tensor_edge_list(self, tensor_edge_list: List, func_name: str):
        """Extract commodities from TensorEdgeList.

        Uses validated TensorEdgeList instead of parsing relay function.
        This is more robust as the edge information is already validated
        and tensor_type is accurate.

        Handles constants inside composites:
        - Adds constant nodes to self.nodes
        - Creates commodities from constant -> composite (INODE -> IMCE)
        - const_to_inode is populated later based on composite placement

        Args:
            tensor_edge_list: List of TensorEdge from constructTensorEdgeList()
            func_name: Function name for SplitInfo lookup
        """
        # CustomIDToNode is imported at module level
        id_to_node = CustomIDToNode()

        # Get SplitInfo for this function to determine multicast vs unicast for split outputs
        from tvm.contrib.imcflow import ImcflowDeviceConfig
        split_info_dict = ImcflowDeviceConfig().SplitInfo.get(func_name, {})

        for edge in tensor_edge_list:
            src_graph_id = edge.src_id.graph_node_id
            dst_graph_id = edge.dst_id.graph_node_id
            tensor_type = edge.src_id.tensor_type

            # Check if source is a constant inside composite
            src_is_const_in_composite = False
            src_inner_id = None
            src_outer_id = None
            if isinstance(src_graph_id, tuple):
                src_outer_id, src_inner_id = src_graph_id[0], src_graph_id[1]
                src_relay_node = id_to_node.get(src_inner_id)
                if src_relay_node is not None and isinstance(src_relay_node, relay.Constant):
                    src_is_const_in_composite = True

            # Get destination node info
            dst_outer_id = getOuterNodeID(dst_graph_id)
            dst_inner_id = getInnerNodeID(dst_graph_id) if isinstance(dst_graph_id, tuple) else dst_graph_id

            if src_is_const_in_composite:
                # Constant inside composite: add constant node and commodity
                # The constant will be loaded from INODE at the same row as the composite's IMCE

                # Add constant node if not already present
                if src_inner_id not in self.nodes:
                    self.topo_order += 1
                    self.nodes[src_inner_id] = GraphNode(
                        id=src_inner_id,
                        node_type=NodeType.CONST,
                        relay_expr=id_to_node.get(src_inner_id),
                        topo_order=self.topo_order,
                    )

                # Get destination node (the composite)
                dst_node = self.nodes.get(src_outer_id)  # composite outer_id
                if dst_node is None:
                    debug_print(f"[_extract_commodities] Composite {src_outer_id} not found for const {src_inner_id}")
                    continue

                # Add commodity: constant -> composite
                # Source is INODE (determined later), destination is composite IMCE
                self.add_commodity(
                    src_inner_id, src_outer_id,
                    NodeType.CONST, dst_node.node_type,
                    tensor_type=tensor_type,
                    split_idx=edge.split_idx,
                    metadata=edge
                )
            else:
                # Regular edge: use outer_id for both src and dst
                src_node_id = getOuterNodeID(src_graph_id)

                # Check if destination is FUNC_OUT (tensor_type starts with 'func_out')
                # FUNC_OUT nodes use tuple key (node_id, tuple_idx) for multiple outputs
                dst_tensor_type = edge.dst_id.tensor_type
                if dst_tensor_type.startswith('func_out'):
                    # Parse tuple index from 'func_outN' (e.g., 'func_out0' -> 0)
                    tuple_idx = parse_funcout_tuple_idx(dst_tensor_type)
                    dst_node_id = (dst_outer_id, tuple_idx)
                else:
                    dst_node_id = dst_outer_id

                # Look up nodes from GraphExtractor's extracted nodes
                src_node = self.nodes.get(src_node_id)
                dst_node = self.nodes.get(dst_node_id)

                # Skip if nodes not found
                if src_node is None or dst_node is None:
                    if isinstance(src_node_id, int) and src_node_id > 0:
                        debug_print(f"[_extract_commodities_from_tensor_edge_list] "
                                   f"Node not found: src_found={src_node is not None}, "
                                   f"dst_found={dst_node is not None}, "
                                   f"node_ids=({src_node_id}, {dst_node_id})")
                    continue

                # Skip self-loops (same composite)
                if src_node_id == dst_node_id:
                    continue

                # Determine multicast flag for split outputs
                is_multicast = True
                if src_node.node_type == NodeType.SPLIT:
                    debug_print(f"[GraphExtractor.split_multicast] Checking edge {edge} for multicast enable at split node {src_node_id}")
                    split_inner_id = getInnerNodeID(src_node_id)
                    if split_inner_id in split_info_dict:
                        is_multicast = split_info_dict[split_inner_id].get('is_multi_cast', True)
                        debug_print(f"  edge {edge}: is_multicast={is_multicast} from SplitInfo")
                    else:
                        debug_print(f"  edge {edge}: No SplitInfo found for split node {split_inner_id}, defaulting to multicast")

                # Add commodity with validated tensor_type from TensorEdgeList
                self.add_commodity(
                    src_node_id, dst_node_id,
                    src_node.node_type, dst_node.node_type,
                    tensor_type=tensor_type,
                    split_idx=edge.split_idx,
                    metadata=edge,
                    is_multicast=is_multicast,
                )

        debug_print(f"[_extract_commodities_from_tensor_edge_list] "
                   f"Extracted {len(self.commodities)} commodities from "
                   f"{len(tensor_edge_list)} tensor edges")
        for comm in self.commodities:
            debug_print(f"  Commodity {comm.id}: "
                       f"{comm.source_node_id} ({comm.source_type}) -> "
                       f"{comm.dest_node_id} ({comm.dest_type}), "
                       f"type={comm.tensor_type}")

        # Build TensorEdge -> commodity_id map for routing result matching
        self.tensor_edge_to_commodity_id = {}
        for comm in self.commodities:
            if comm.metadata is not None:
                self.tensor_edge_to_commodity_id[comm.metadata] = comm.id

    def add_commodity(self, src_node_id: Any, dst_node_id: Any,
                      src_type: NodeType, dst_type: NodeType,
                      tensor_type: str, split_idx: Optional[int] = None,
                      metadata: Any = None, is_multicast: bool = True):
        """Add a commodity (data flow) between nodes"""
        commodity = Commodity(
            id=self.commodity_id,
            source_node_id=src_node_id,
            dest_node_id=dst_node_id,
            source_type=src_type,
            dest_type=dst_type,
            tensor_type=tensor_type,
            split_idx=split_idx,
            metadata=metadata,
            is_multicast=is_multicast,
        )
        self.commodities.append(commodity)
        self.commodity_id += 1
        return commodity


# ============================================================
# Joint PnR ILP Solver
# ============================================================

class JointPnRILP:
    """Joint node mapping and routing ILP solver"""

    def __init__(self, topology: MeshTopology = None, preassigned_placements: Optional[Dict[str, Dict[Any, 'Coord']]] = None):
        self.topology = topology or MeshTopology(4, 5)
        # Pre-assigned placements: func_name -> {graph_node_id -> Coord}
        self.preassigned_placements = preassigned_placements or {}

        # ILP model components (set during build)
        self.prob = None
        self.graph_info = None

        # Decision variables
        self.p = {}   # p[n][v] = 1 if call node n placed at IMCE v
        self.x = {}   # x[k][e] = 1 if commodity k uses edge e
        self.y = {}   # y[g][e] = 1 if multicast group g uses edge e
        self.src = {} # src[k][v] = 1 if commodity k's source is at node v
        self.dst = {} # dst[k][v] = 1 if commodity k's destination is at node v

        # Index mappings
        self.all_edges = []
        self.edge_to_idx = {}
        self.all_nodes = []
        self.node_to_idx = {}
        self.imce_nodes = []
        self.inode_nodes = []

    def run(
        self,
        mod: tvm.IRModule,
        tensor_edge_list_dict: Dict[str, List],
    ) -> Dict[str, JointPnRResult]:
        """Run joint PnR for all imcflow functions.

        Args:
            mod: TVM IRModule containing imcflow functions
            tensor_edge_list_dict: Dict mapping func_name -> List[TensorEdge]

        Returns:
            Dict mapping function name to JointPnRResult
        """
        results = {}

        # Get ImcflowFuncMap - TensorEdgeList uses func_info.func_node, we should too
        imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
        debug_print(f"[JointPnRILP] ImcflowFuncMap keys: {list(imcflow_func_map.keys())}")

        for gv, func in mod.functions.items():
            if (isinstance(func, relay.Function) and
                hasattr(func.attrs, "Compiler") and
                func.attrs["Compiler"] == "imcflow"):

                func_name = gv.name_hint
                debug_print(f"[JointPnRILP] Processing function: {func_name}")

                if func_name not in tensor_edge_list_dict:
                    raise ValueError(
                        f"[JointPnRILP] tensor_edge_list_dict missing entry for '{func_name}'. "
                        f"Ensure constructTensorEdgeList() is called before Joint PnR."
                    )

                # Use func_info.func_node from ImcflowFuncMap (same as TensorEdgeList uses)
                # This ensures GraphExtractor sees the same node instances with matching custom_ids
                if func_name in imcflow_func_map:
                    actual_func = imcflow_func_map[func_name].func_node
                    debug_print(f"[JointPnRILP] Using func_info.func_node for {func_name}")
                else:
                    actual_func = func
                    debug_print(f"[JointPnRILP] Warning: {func_name} not in ImcflowFuncMap, using mod.functions")

                tensor_edge_list = tensor_edge_list_dict[func_name]
                func_preassigned = self.preassigned_placements.get(func_name, {})
                result = self._run_for_function(actual_func, func_name, tensor_edge_list, func_preassigned)
                results[func_name] = result

                if result.success:
                    debug_print(f"[JointPnRILP] Success: max_cong={result.max_congestion}, "
                                f"hops={result.total_hops}")
                else:
                    debug_print(f"[JointPnRILP] Failed: {result.solver_status}")


        return results

    def _run_for_function(
        self,
        func: relay.Function,
        func_name: str,
        tensor_edge_list: List,
        preassigned_call_placements: Optional[Dict[Any, Coord]] = None,
    ) -> JointPnRResult:
        """Run joint PnR for a single function.

        Args:
            func: Relay function
            func_name: Name of the function
            tensor_edge_list: List of TensorEdge from constructTensorEdgeList()
            preassigned_call_placements: Optional dict mapping graph_node_id -> Coord
                for call nodes with pre-determined placements (from Phase 1).
                When provided, these nodes are fixed at their assigned IMCE coords.

        Returns:
            JointPnRResult with mapping and routes
        """
        debug_print(f"[JointPnRILP] Running Joint PnR for function: {func_name}")
        self._current_preassigned = preassigned_call_placements or {}
        if self._current_preassigned:
            debug_print(f"[JointPnRILP] Using {len(self._current_preassigned)} pre-assigned placements")

        # Phase 1: Extract graph info using GraphExtractor
        # - Nodes: extracted from relay function (composite = 1 node)
        # - Commodities: extracted from TensorEdgeList (robust, validated)
        extractor = GraphExtractor()
        self.graph_info = extractor.extract(func, func_name, tensor_edge_list)

        if not self.graph_info.call_nodes:
            # No call nodes to place - trivial case
            return JointPnRResult(
                mapping={},
                routes={},
                commodities=[],
                max_congestion=0,
                total_hops=0,
                solver_status="No call nodes",
                success=True,
                var_to_inode=self.graph_info.var_to_inode,
                funcout_to_inode=self.graph_info.funcout_to_inode,
                const_to_inode=self.graph_info.const_to_inode,
                tensor_edge_to_commodity_id=self.graph_info.tensor_edge_to_commodity_id,
            )

        # Phase 2: Build ILP model
        self._build_model()

        # Phase 3: Solve
        return self._solve()

    def _build_model(self):
        """Build complete ILP model"""
        debug_print(f"[JointPnRILP] Building ILP model...")

        # Initialize topology structures
        self.all_edges = self.topology.get_all_edges()
        self.edge_to_idx = {e: i for i, e in enumerate(self.all_edges)}
        self.all_nodes = self.topology.get_all_nodes()
        self.node_to_idx = {n: i for i, n in enumerate(self.all_nodes)}
        self.imce_nodes = self.topology.get_imce_nodes()
        self.inode_nodes = self.topology.get_inode_nodes()

        # Create problem
        self.prob = pulp.LpProblem("JointPnR", pulp.LpMinimize)

        # Create variables
        self._create_variables()

        # Add constraints
        self._add_placement_constraints()
        self._add_linking_constraints()
        self._add_flow_constraints()
        self._add_multicast_constraints()
        self._add_capacity_constraints()

        # Set objective
        self._set_objective()

        debug_print(f"[JointPnRILP] Model built: {len(self.prob.variables())} variables")

    def _create_variables(self):
        """Create all ILP variables"""
        gi = self.graph_info

        # Identify splits whose producer is a VAR - these go to INODE, not IMCE
        self.splits_preassigned_to_inode = {}  # split_id -> Coord (INODE coord)
        splits_to_exclude = set()
        for split_id in gi.split_nodes:
            split_node = gi.nodes[split_id]
            if split_node.producer and split_node.producer in gi.var_to_inode:
                # Producer is a VAR -> split goes to the same INODE as VAR
                var_inode_coord = gi.var_to_inode[split_node.producer]
                self.splits_preassigned_to_inode[split_id] = var_inode_coord
                splits_to_exclude.add(split_id)
                debug_print(f"[JointPnRILP] Split {split_id} has VAR producer {split_node.producer}, "
                           f"pre-assigning to INODE at ({var_inode_coord.row}, {var_inode_coord.col})")

        # Identify call nodes with pre-assigned IMCE placements
        self.calls_preassigned_to_imce = {}  # call_id -> Coord
        calls_preassigned = set()
        for call_id in gi.call_nodes:
            if call_id in self._current_preassigned:
                coord = self._current_preassigned[call_id]
                self.calls_preassigned_to_imce[call_id] = coord
                calls_preassigned.add(call_id)
                debug_print(f"[JointPnRILP] Call {call_id} pre-assigned to IMCE ({coord.row}, {coord.col})")

        # All placeable nodes (call + split + concat), excluding pre-assigned splits
        placeable_nodes = (
            [c for c in gi.call_nodes if c not in calls_preassigned] +
            [s for s in gi.split_nodes if s not in splits_to_exclude] +
            gi.concat_nodes
        )

        # p[n][v]: placement of call/split/concat node n at IMCE v
        self.p = {}
        for n in placeable_nodes:
            self.p[n] = {}
            for v in self.imce_nodes:
                self.p[n][v] = pulp.LpVariable(
                    f"p_{hash(n) % 100000}_{v.row}_{v.col}",
                    cat=pulp.LpBinary
                )

        # For pre-assigned call nodes, create fixed variables (p[n][assigned_v]=1, rest=0)
        for call_id, coord in self.calls_preassigned_to_imce.items():
            self.p[call_id] = {}
            for v in self.imce_nodes:
                val = 1 if v == coord else 0
                self.p[call_id][v] = pulp.LpVariable(
                    f"p_{hash(call_id) % 100000}_{v.row}_{v.col}",
                    cat=pulp.LpBinary
                )
                self.p[call_id][v].setInitialValue(val)
                self.p[call_id][v].fixValue()

        # x[k][e]: commodity k uses edge e
        self.x = {}
        for k in gi.commodities:
            self.x[k.id] = {}
            for e in self.all_edges:
                self.x[k.id][e] = pulp.LpVariable(
                    f"x_{k.id}_{self.edge_to_idx[e]}",
                    cat=pulp.LpBinary
                )

        # src[k][v], dst[k][v]: source/destination location for commodity k
        self.src = {}
        self.dst = {}
        for k in gi.commodities:
            self.src[k.id] = {}
            self.dst[k.id] = {}
            for v in self.all_nodes:
                self.src[k.id][v] = pulp.LpVariable(
                    f"src_{k.id}_{v.row}_{v.col}",
                    cat=pulp.LpBinary
                )
                self.dst[k.id][v] = pulp.LpVariable(
                    f"dst_{k.id}_{v.row}_{v.col}",
                    cat=pulp.LpBinary
                )

        # Multicast grouping: y[g][e] for commodities with same source
        # Group by (source_node_id, tensor_type) for multicast
        # Non-multicast split outputs get individual groups (unicast routing)
        self.multicast_groups = {}  # group_key -> [commodity_ids]
        for k in gi.commodities:
            if not k.is_multicast and k.split_idx is not None:
                # Non-multicast split output: each goes to separate group for individual routing
                key = (k.source_node_id, k.tensor_type, k.split_idx)
                debug_print(f"[JointPnRILP.multicast] Commodity {k.id} non-multicast split output, separate group key: {key}")
                debug_print(f"  Source node: {k.source_node_id}, tensor_type: {k.tensor_type}, split_idx: {k.split_idx}")
            else:
                key = (k.source_node_id, k.tensor_type)
            if key not in self.multicast_groups:
                self.multicast_groups[key] = []
            self.multicast_groups[key].append(k.id)

        # y[g][e]: multicast group g uses edge e
        self.y = {}
        for gid, (group_key, members) in enumerate(self.multicast_groups.items()):
            if len(members) > 1:  # Only create for actual multicast
                self.y[gid] = {}
                for e in self.all_edges:
                    self.y[gid][e] = pulp.LpVariable(
                        f"y_{gid}_{self.edge_to_idx[e]}",
                        cat=pulp.LpBinary
                    )

        debug_print(f"[JointPnRILP] Created {len(self.p)} placement, "
                   f"{len(self.x)} routing, {len(self.y)} multicast variable sets")

    def _add_placement_constraints(self):
        """Add placement constraints P1, P2, P3, P4"""
        gi = self.graph_info

        # P1: Each placeable node -> exactly one IMCE (skip pre-assigned splits)
        for n in gi.call_nodes + gi.split_nodes + gi.concat_nodes:
            if n in self.splits_preassigned_to_inode:
                continue  # Skip pre-assigned splits (they go to INODE, not IMCE)
            self.prob += (
                pulp.lpSum(self.p[n][v] for v in self.imce_nodes) == 1,
                f"P1_one_location_{hash(n) % 100000}"
            )

        # P2: Each IMCE -> at most one REAL call (split/concat excluded)
        for v in self.imce_nodes:
            real_calls = gi.call_nodes  # Exclude split and concat
            if real_calls:
                self.prob += (
                    pulp.lpSum(self.p[n][v] for n in real_calls) <= 1,
                    f"P2_one_call_per_imce_{v.row}_{v.col}"
                )

        # P3: Split same as producer (skip pre-assigned splits with VAR producers)
        for split_id in gi.split_nodes:
            if split_id in self.splits_preassigned_to_inode:
                continue  # Skip pre-assigned splits (they go to INODE, not IMCE)
            split_node = gi.nodes[split_id]
            if split_node.producer and split_node.producer in self.p:
                for v in self.imce_nodes:
                    self.prob += (
                        self.p[split_id][v] == self.p[split_node.producer][v],
                        f"P3_split_{hash(split_id) % 100000}_{v.row}_{v.col}"
                    )

        # P4: Concat same as last producer
        for concat_id in gi.concat_nodes:
            concat_node = gi.nodes[concat_id]
            if concat_node.last_producer and concat_node.last_producer in self.p:
                for v in self.imce_nodes:
                    self.prob += (
                        self.p[concat_id][v] == self.p[concat_node.last_producer][v],
                        f"P4_concat_{hash(concat_id) % 100000}_{v.row}_{v.col}"
                    )

    def _add_linking_constraints(self):
        """Add source/destination linking constraints L1-L5"""
        gi = self.graph_info

        for k in gi.commodities:
            src_node = gi.nodes.get(k.source_node_id)
            dst_node = gi.nodes.get(k.dest_node_id)

            if src_node is None or dst_node is None:
                continue

            # L1: Call source -> placement linking
            if src_node.node_type in [NodeType.CALL, NodeType.SPLIT, NodeType.CONCAT]:
                if k.source_node_id in self.p:
                    for v in self.imce_nodes:
                        self.prob += (
                            self.src[k.id][v] == self.p[k.source_node_id][v],
                            f"L1_src_{k.id}_{v.row}_{v.col}"
                        )
                    # Source is not at INODE
                    for v in self.inode_nodes:
                        self.prob += (
                            self.src[k.id][v] == 0,
                            f"L1_src_not_inode_{k.id}_{v.row}_{v.col}"
                        )
                elif k.source_node_id in self.splits_preassigned_to_inode:
                    # L1b: Pre-assigned split source (fixed to INODE)
                    split_inode = self.splits_preassigned_to_inode[k.source_node_id]
                    debug_print(f"[L1b] Commodity {k.id}: Pre-assigned split {k.source_node_id} -> INODE ({split_inode.row}, {split_inode.col})")
                    for v in self.all_nodes:
                        if v == split_inode:
                            self.prob += (
                                self.src[k.id][v] == 1,
                                f"L1b_preassigned_split_src_{k.id}_{v.row}_{v.col}"
                            )
                        else:
                            self.prob += (
                                self.src[k.id][v] == 0,
                                f"L1b_preassigned_split_src_not_{k.id}_{v.row}_{v.col}"
                            )

            # L2: Call dest -> placement linking
            if dst_node.node_type in [NodeType.CALL, NodeType.SPLIT, NodeType.CONCAT]:
                if k.dest_node_id in self.p:
                    for v in self.imce_nodes:
                        self.prob += (
                            self.dst[k.id][v] == self.p[k.dest_node_id][v],
                            f"L2_dst_{k.id}_{v.row}_{v.col}"
                        )
                    # Dest is not at INODE
                    for v in self.inode_nodes:
                        self.prob += (
                            self.dst[k.id][v] == 0,
                            f"L2_dst_not_inode_{k.id}_{v.row}_{v.col}"
                        )
                elif k.dest_node_id in self.splits_preassigned_to_inode:
                    # L2b: Pre-assigned split dest (fixed to INODE)
                    split_inode = self.splits_preassigned_to_inode[k.dest_node_id]
                    debug_print(f"[L2b] Commodity {k.id}: Pre-assigned split dest {k.dest_node_id} -> INODE ({split_inode.row}, {split_inode.col})")
                    for v in self.all_nodes:
                        if v == split_inode:
                            self.prob += (
                                self.dst[k.id][v] == 1,
                                f"L2b_preassigned_split_dst_{k.id}_{v.row}_{v.col}"
                            )
                        else:
                            self.prob += (
                                self.dst[k.id][v] == 0,
                                f"L2b_preassigned_split_dst_not_{k.id}_{v.row}_{v.col}"
                            )

            # L3: Var source (fixed to inode from var_to_inode)
            if src_node.node_type == NodeType.VAR:
                var_inode = self.graph_info.var_to_inode.get(k.source_node_id)
                if var_inode is None:
                    raise KeyError(f"VAR source_node_id {k.source_node_id} not found in var_to_inode")
                debug_print(f"[L3] Commodity {k.id}: VAR {k.source_node_id} -> INODE ({var_inode.row}, {var_inode.col})")
                for v in self.all_nodes:
                    if v == var_inode:
                        self.prob += (
                            self.src[k.id][v] == 1,
                            f"L3_var_src_{k.id}_{v.row}_{v.col}"
                        )
                    else:
                        self.prob += (
                            self.src[k.id][v] == 0,
                            f"L3_var_src_not_{k.id}_{v.row}_{v.col}"
                        )

            # L4: Func_out dest (fixed to inode from funcout_to_inode)
            if dst_node.node_type == NodeType.FUNC_OUT:
                funcout_inode = self.graph_info.funcout_to_inode.get(k.dest_node_id)
                if funcout_inode is None:
                    raise KeyError(f"FUNC_OUT dest_node_id {k.dest_node_id} not found in funcout_to_inode")
                debug_print(f"[L4] Commodity {k.id}: FUNC_OUT {k.dest_node_id} -> INODE ({funcout_inode.row}, {funcout_inode.col})")
                for v in self.all_nodes:
                    if v == funcout_inode:
                        self.prob += (
                            self.dst[k.id][v] == 1,
                            f"L4_funcout_dst_{k.id}_{v.row}_{v.col}"
                        )
                    else:
                        self.prob += (
                            self.dst[k.id][v] == 0,
                            f"L4_funcout_dst_not_{k.id}_{v.row}_{v.col}"
                        )

            # L5: Const source (same row INODE as consumer)
            if src_node.node_type == NodeType.CONST:
                if k.dest_node_id in self.p:
                    # For each row, if consumer is in that row, source is at row's INODE
                    for row in range(self.topology.rows):
                        inode = Coord(row, 0)
                        # src[k][inode] >= sum of p[consumer][(row, c)] for c in IMCE cols
                        imce_in_row = [v for v in self.imce_nodes if v.row == row]
                        self.prob += (
                            self.src[k.id][inode] >= pulp.lpSum(
                                self.p[k.dest_node_id][v] for v in imce_in_row
                            ),
                            f"L5_const_row_{k.id}_{row}"
                        )
                    # Not in IMCE
                    for v in self.imce_nodes:
                        self.prob += (
                            self.src[k.id][v] == 0,
                            f"L5_const_not_imce_{k.id}_{v.row}_{v.col}"
                        )

        # Ensure exactly one source and destination per commodity
        for k in gi.commodities:
            self.prob += (
                pulp.lpSum(self.src[k.id][v] for v in self.all_nodes) == 1,
                f"one_src_{k.id}"
            )
            self.prob += (
                pulp.lpSum(self.dst[k.id][v] for v in self.all_nodes) == 1,
                f"one_dst_{k.id}"
            )

    def _add_flow_constraints(self):
        """Add flow conservation constraints F1, F2"""
        gi = self.graph_info

        for k in gi.commodities:
            for v in self.all_nodes:
                # Get incoming and outgoing edges for this node
                in_edges = [e for e in self.all_edges if e.dst == v]
                out_edges = [e for e in self.all_edges if e.src == v]

                in_flow = pulp.lpSum(self.x[k.id][e] for e in in_edges) if in_edges else 0
                out_flow = pulp.lpSum(self.x[k.id][e] for e in out_edges) if out_edges else 0

                # F1: Flow conservation with variable source/dest
                # outflow - inflow = src[k][v] - dst[k][v]
                self.prob += (
                    out_flow - in_flow == self.src[k.id][v] - self.dst[k.id][v],
                    f"F1_flow_{k.id}_{v.row}_{v.col}"
                )

                # F2: Simple path (at most 1 outgoing at transit nodes)
                # This is implicit from flow conservation + binary variables

    def _add_multicast_constraints(self):
        """Add multicast constraints M1"""
        gi = self.graph_info

        # For each multicast group with y variables
        for gid, (group_key, members) in enumerate(self.multicast_groups.items()):
            if len(members) <= 1:
                continue  # Skip unicast

            if gid not in self.y:
                continue

            # M1: y[g][e] >= x[k][e] for all k in group
            for k_id in members:
                for e in self.all_edges:
                    self.prob += (
                        self.y[gid][e] >= self.x[k_id][e],
                        f"M1_mcast_{gid}_{k_id}_{self.edge_to_idx[e]}"
                    )

    def _add_capacity_constraints(self):
        """Add edge capacity constraints C1 (data group only)"""
        gi = self.graph_info

        # Separate commodities by congestion group
        data_commodities = [k for k in gi.commodities if k.get_congestion_group() == 'data']

        # Find multicast groups that are data type
        data_mcast_gids = []
        data_ucast_kids = []

        for gid, (group_key, members) in enumerate(self.multicast_groups.items()):
            # Check if this group is data type
            first_k = next(k for k in gi.commodities if k.id == members[0])
            if first_k.get_congestion_group() == 'data':
                if len(members) > 1 and gid in self.y:
                    data_mcast_gids.append(gid)
                else:
                    data_ucast_kids.extend(members)

        # C1: Data group edge capacity = 1
        for e in self.all_edges:
            mcast_usage = pulp.lpSum(self.y[gid][e] for gid in data_mcast_gids) if data_mcast_gids else 0
            ucast_usage = pulp.lpSum(self.x[k_id][e] for k_id in data_ucast_kids) if data_ucast_kids else 0

            self.prob += (
                mcast_usage + ucast_usage <= 1,
                f"C1_cap_{self.edge_to_idx[e]}"
            )

    def _set_objective(self):
        """Set objective function: minimize congestion + small wirelength penalty"""
        gi = self.graph_info

        # Create max congestion variable
        max_congestion = pulp.LpVariable("max_cong", lowBound=0, cat=pulp.LpInteger)

        # Data group commodities
        data_commodities = [k for k in gi.commodities if k.get_congestion_group() == 'data']
        data_mcast_gids = []
        data_ucast_kids = []

        for gid, (group_key, members) in enumerate(self.multicast_groups.items()):
            first_k = next((k for k in gi.commodities if k.id == members[0]), None)
            if first_k and first_k.get_congestion_group() == 'data':
                if len(members) > 1 and gid in self.y:
                    data_mcast_gids.append(gid)
                else:
                    data_ucast_kids.extend(members)

        # max_congestion >= usage[e] for all data edges
        for e in self.all_edges:
            mcast_usage = pulp.lpSum(self.y[gid][e] for gid in data_mcast_gids) if data_mcast_gids else 0
            ucast_usage = pulp.lpSum(self.x[k_id][e] for k_id in data_ucast_kids) if data_ucast_kids else 0
            self.prob += (
                max_congestion >= mcast_usage + ucast_usage,
                f"obj_cong_{self.edge_to_idx[e]}"
            )

        # Total hops (wirelength)
        total_hops = pulp.lpSum(
            self.x[k.id][e]
            for k in gi.commodities
            for e in self.all_edges
        )

        # Objective: minimize congestion + small wirelength penalty
        self.prob += max_congestion + 0.001 * total_hops

        # Store for result extraction
        self.max_congestion_var = max_congestion
        self.total_hops_expr = total_hops

    def _solve(self) -> JointPnRResult:
        """Solve ILP and extract results"""
        debug_print(f"[JointPnRILP] Solving ILP...")

        # Dump ILP to file for debugging
        self.prob.writeLP("/tmp/joint_pnr_debug.lp")
        debug_print(f"[JointPnRILP] Dumped ILP to /tmp/joint_pnr_debug.lp")

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
        status = self.prob.solve(solver)

        status_str = pulp.LpStatus[status]
        debug_print(f"[JointPnRILP] Solver status: {status_str}")

        if status != pulp.LpStatusOptimal:
            return JointPnRResult(
                mapping={},
                routes={},
                commodities=[],
                max_congestion=0,
                total_hops=0,
                solver_status=f"Infeasible: {status_str}",
                success=False,
                var_to_inode={},
                funcout_to_inode={},
                const_to_inode={},
                tensor_edge_to_commodity_id={},
            )

        # Extract placement
        mapping = {}
        gi = self.graph_info

        # First, add pre-assigned splits (those with VAR producers -> INODE)
        for split_id, inode_coord in self.splits_preassigned_to_inode.items():
            mapping[split_id] = inode_coord
            debug_print(f"[JointPnRILP] Pre-assigned split {split_id} -> INODE ({inode_coord.row}, {inode_coord.col})")

        # Then, extract ILP-solved placements for other nodes
        for n in gi.call_nodes + gi.split_nodes + gi.concat_nodes:
            if n in self.splits_preassigned_to_inode:
                continue  # Already handled above
            for v in self.imce_nodes:
                if pulp.value(self.p[n][v]) > 0.5:
                    mapping[n] = v
                    break

        # Extract routes
        routes = {}
        for k in gi.commodities:
            used_edges = [e for e in self.all_edges if pulp.value(self.x[k.id][e]) > 0.5]
            routes[k.id] = used_edges

        # Extract statistics
        max_congestion = int(pulp.value(self.max_congestion_var)) if hasattr(self, 'max_congestion_var') else 0
        total_hops = sum(len(edges) for edges in routes.values())

        debug_print(f"[JointPnRILP] Extracted {len(mapping)} placements, "
                   f"{len(routes)} routes, max_cong={max_congestion}, hops={total_hops}")

        # Compute const_to_inode based on placement
        # Constants inside composites get INODE at the same row as their composite's IMCE
        const_to_inode = {}
        for commodity in gi.commodities:
            if commodity.source_type == NodeType.CONST:
                # Source is a constant, destination is its composite
                const_id = commodity.source_node_id
                composite_id = commodity.dest_node_id
                if composite_id in mapping:
                    composite_coord = mapping[composite_id]
                    # INODE is at the same row as composite's IMCE
                    const_to_inode[const_id] = Coord(composite_coord.row, 0)

        result = JointPnRResult(
            mapping=mapping,
            routes=routes,
            commodities=gi.commodities,
            max_congestion=max_congestion,
            total_hops=total_hops,
            solver_status=status_str,
            success=True,
            var_to_inode=gi.var_to_inode,
            funcout_to_inode=gi.funcout_to_inode,
            const_to_inode=const_to_inode,
            tensor_edge_to_commodity_id=gi.tensor_edge_to_commodity_id,
        )

        # Dump result for debugging
        debug_print("=" * 70)
        debug_print("[JointPnRILP] RESULT DUMP")
        debug_print("=" * 70)
        debug_print(f"[RESULT] solver_status: {status_str}")
        debug_print(f"[RESULT] max_congestion: {max_congestion}, total_hops: {total_hops}")

        debug_print("\n[RESULT] mapping (graph_node_id -> Coord):")
        for node_id, coord in mapping.items():
            debug_print(f"  {node_id} -> ({coord.row}, {coord.col})")

        debug_print("\n[RESULT] var_to_inode:")
        for node_id, coord in (gi.var_to_inode or {}).items():
            debug_print(f"  {node_id} -> ({coord.row}, {coord.col})")

        debug_print("\n[RESULT] funcout_to_inode:")
        for node_id, coord in (gi.funcout_to_inode or {}).items():
            debug_print(f"  {node_id} -> ({coord.row}, {coord.col})")

        debug_print("\n[RESULT] const_to_inode:")
        for node_id, coord in (const_to_inode or {}).items():
            debug_print(f"  {node_id} -> ({coord.row}, {coord.col})")

        debug_print("\n[RESULT] commodities:")
        for comm in gi.commodities:
            debug_print(f"  Commodity {comm.id}: {comm.source_node_id} ({comm.source_type.name}) "
                       f"-> {comm.dest_node_id} ({comm.dest_type.name}), type={comm.tensor_type}")

        debug_print("\n[RESULT] routes (commodity_id -> edges):")
        for comm_id, edges in routes.items():
            edges_str = ", ".join(f"({e.src.row},{e.src.col})->({e.dst.row},{e.dst.col})" for e in edges)
            debug_print(f"  Commodity {comm_id}: [{edges_str}]")

        debug_print("\n[RESULT] tensor_edge_to_commodity_id:")
        for tensor_edge, comm_id in (gi.tensor_edge_to_commodity_id or {}).items():
            debug_print(f"  {tensor_edge} -> Commodity {comm_id}")

        debug_print("=" * 70)

        return result


# ============================================================
# Integration with existing infrastructure
# ============================================================

def run_joint_pnr(
    mod: tvm.IRModule,
    tensor_edge_list_dict: Dict[str, List],
    preassigned_placements: Optional[Dict[str, Dict[Any, Coord]]] = None,
) -> Dict[str, JointPnRResult]:
    """
    Run joint PnR for all imcflow functions in module.

    This is the main entry point for the joint PnR solver.

    Args:
        mod: TVM module containing imcflow functions
        tensor_edge_list_dict: Dict mapping func_name -> List[TensorEdge]
                               Must be constructed via constructTensorEdgeList() first.
        preassigned_placements: Optional dict mapping func_name -> {graph_node_id -> Coord}
                                for nodes with pre-determined placements.

    Returns:
        Dict mapping function names to JointPnRResult
    """
    solver = JointPnRILP(topology=MeshTopology(4, 5), preassigned_placements=preassigned_placements)
    return solver.run(mod, tensor_edge_list_dict)


def update_hw_node_map(
    results: Dict[str, JointPnRResult],
    hw_node_map: Dict,
    tensor_edge_list_dict: Dict[str, List],
):
    """
    Update HWNodeMap with placement results.

    HWNodeMap key convention:
    - Composite calls: use inner_id (the inner call's ID)
    - Constants inside composites: use inner_id -> INODE (from const_to_inode)
    - Regular calls: use custom_id
    - Vars/FuncOut: use custom_id

    Args:
        results: Results from run_joint_pnr
        hw_node_map: ImcflowDeviceConfig().HWNodeMap to update
        tensor_edge_list_dict: TensorEdgeListDict for composite inner_id mapping
    """
    # CustomIDToNode is imported at module level
    id_to_node = CustomIDToNode()

    for func_name, result in results.items():
        if not result.success:
            raise RuntimeError(f"Joint PnR failed for {func_name}: {result.solver_status}")

        # Build outer_id -> inner_id mapping from TensorEdgeList for IMCE keys
        outer_to_inner_call = {}  # outer_id -> inner_call_id
        tensor_edge_list = tensor_edge_list_dict.get(func_name, [])
        for edge in tensor_edge_list:
            for tensor_id in [edge.src_id, edge.dst_id]:
                gid = tensor_id.graph_node_id
                if isinstance(gid, tuple):
                    outer_id, inner_id = gid[0], gid[1]
                    # Check if inner_id is a Constant using CustomIDToNode
                    node = id_to_node.get(inner_id)
                    if node is None or not isinstance(node, relay.Constant):
                        # Inner call inside composite (not constant)
                        outer_to_inner_call[outer_id] = inner_id

        # IMCE mappings (call/split/concat nodes)
        for graph_node_id, coord in result.mapping.items():
            # For composite nodes, use inner_call_id as key
            inner_id = outer_to_inner_call.get(graph_node_id, graph_node_id)
            # Convert Coord to NodeID
            node_id = NodeID.from_imce_coord(coord.row, coord.col - 1)
            hw_node_map[inner_id] = node_id
            # Also store outer_id -> NodeID for composite nodes (so other inner nodes can find IMCE)
            if graph_node_id in outer_to_inner_call:
                hw_node_map[graph_node_id] = node_id

        # INODE mappings (input vars)
        if result.var_to_inode:
            for var_node_id, coord in result.var_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[var_node_id] = node_id

        # INODE mappings (funcout)
        if result.funcout_to_inode:
            node_out_len = {}
            temp_map = {}
            for funcout_node_id, coord in result.funcout_to_inode.items():
              if funcout_node_id[0] not in node_out_len:
                node_out_len[funcout_node_id[0]] = 0
              node_out_len[funcout_node_id[0]] += 1
            
            temp_map = {k: [None for _ in range(v)] for k,v in node_out_len.items()}
            for funcout_node_id, coord in result.funcout_to_inode.items():
              node_id = NodeID.from_inode_coord(coord.row)
              temp_map[funcout_node_id[0]][funcout_node_id[1]] = node_id
            
            for funcout_node_id, node_id in temp_map.items():
              if len(node_id) == 1:
                hw_node_map[funcout_node_id] = node_id[0]
              else:
                hw_node_map[funcout_node_id] = tuple(node_id)

        # INODE mappings for constants inside composites
        # const_to_inode is computed in _solve() based on composite placement
        if result.const_to_inode:
            for const_id, coord in result.const_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[getInnerNodeID(const_id)] = node_id


def coord_to_node_id(coord: Coord) -> NodeID:
    """Convert Coord to NodeID"""
    if coord.col == 0:
        # INODE
        return NodeID.from_inode_coord(coord.row)
    else:
        # IMCE
        return NodeID.from_imce_coord(coord.row, coord.col - 1)

def extract_hw_commodities_from_noc_paths(
    noc_paths: Dict,
) -> List[HWCommodity]:
    """
    Extract HWCommodity objects from noc_paths for policy table generation.

    noc_paths contains both:
    - TensorEdge paths: TensorEdge -> (src_hwnode, dst_hwnode, split_idx)
    - Instruction paths: NodeID -> (src_hwnode, dst_hwnode, None)

    Args:
        noc_paths: Dict mapping edge (TensorEdge or NodeID) to (src_hwnode, dst_hwnode, split_idx)

    Returns:
        List of HWCommodity objects (only for non-local paths)
    """
    commodities = []
    commodity_id = 0

    for edge, mapping_info in noc_paths.items():
        src_node = mapping_info[0]  # NodeID
        dst_node = mapping_info[1]  # NodeID

        # Convert NodeID to Coord
        src_coord_tuple = NodeID.to_coord(src_node)
        dst_coord_tuple = NodeID.to_coord(dst_node)

        src_coord = Coord(src_coord_tuple[0], src_coord_tuple[1])
        dst_coord = Coord(dst_coord_tuple[0], dst_coord_tuple[1])

        # Skip local edges (same node) - no routing needed
        if src_coord == dst_coord:
            continue

        # Create HW commodity
        commodity = HWCommodity(
            id=commodity_id,
            source=src_coord,
            destination=dst_coord,
            metadata=(edge, mapping_info)
        )
        commodities.append(commodity)
        commodity_id += 1

    debug_print(f"[extract_hw_commodities] Extracted {len(commodities)} commodities "
               f"from {len(noc_paths)} noc_paths entries")
    
    # dump hw commodities
    for c in commodities:
        debug_print(f"[HWCommodity] ID: {c.id}, Source: ({c.source.row}, {c.source.col}), "
                   f"Destination: ({c.destination.row}, {c.destination.col}), Metadata: {c.metadata}")

    return commodities


# ============================================================
# Routes to BaseRoutingResult Conversion
# ============================================================

class JointPnRRoutingResult:
    """
    Adapter class to convert JointPnRResult routes to BaseRoutingResult interface.

    This allows PolicyTableBuilder to use routes from JointPnRILP directly.
    """

    def __init__(self, pnr_result: JointPnRResult, commodities: List[HWCommodity], noc_paths: Dict):
        """
        Initialize from JointPnRResult.

        Args:
            pnr_result: Result from JointPnRILP
            commodities: List of HWCommodity objects (from extract_hw_commodities_from_noc_paths)
            noc_paths: NoCPaths dict
        """
        self._commodities = {}
        self._paths = {}
        self._noc_paths = noc_paths
        self._hw_to_ilp_commodity = {}  # HWCommodity ID -> ILP Commodity map

        # Get TensorEdge -> ILP commodity_id map
        tensor_edge_to_commodity_id = pnr_result.tensor_edge_to_commodity_id or {}

        # Build ILP commodity lookup by ID
        ilp_commodities_by_id = {c.id: c for c in (pnr_result.commodities or [])}

        # Build commodity lookup by id
        for commodity in commodities:
            self._commodities[commodity.id] = commodity

            # Get edge from HWCommodity metadata: (edge, mapping_info)
            edge = commodity.metadata[0] if commodity.metadata else None

            # TensorEdge commodities: use map to find ILP commodity_id and get route
            if edge is not None and edge in tensor_edge_to_commodity_id:
                ilp_commodity_id = tensor_edge_to_commodity_id[edge]

                # Store HW -> ILP commodity mapping
                if ilp_commodity_id in ilp_commodities_by_id:
                    self._hw_to_ilp_commodity[commodity.id] = ilp_commodities_by_id[ilp_commodity_id]

                if pnr_result.routes and ilp_commodity_id in pnr_result.routes:
                    edges = pnr_result.routes[ilp_commodity_id]
                    path = self._edges_to_path(commodity.source, edges)
                    self._paths[commodity.id] = path
                    debug_print(f"[JointPnRRoutingResult.__init__] edge:{edge}, gcomm_id:{ilp_commodity_id}, hcomm_id:{commodity.id}, path:{path}")
                    continue

            # Instruction path (NodeID) or fallback: use direct path
            path = self._generate_direct_path(commodity.source, commodity.destination)
            self._paths[commodity.id] = path
            debug_print(f"[JointPnRRoutingResult.__init__] inst path for hcomm_id:{commodity.id}, path:{path}")
        
    def _edges_to_path(self, start: Coord, edges: List[Edge]) -> List[Coord]:
        """Convert list of edges to path (list of coords)"""
        debug_print(f"[JointPnRRoutingResult._edges_to_path]:")
        debug_print(f"  edges:{edges}")

        if not edges:
            return [start]

        # Build adjacency from edges
        adj = {}
        for e in edges:
            if e.src not in adj:
                adj[e.src] = []
            adj[e.src].append(e.dst)

        # Walk from start
        path = [start]
        current = start

        visited = set()
        while current in adj and current not in visited:
            visited.add(current)
            next_nodes = adj[current]
            if next_nodes:
                next_node = next_nodes[0]  # Take first (should be unique for simple path)
                path.append(next_node)
                current = next_node
            else:
                break
        
        debug_print(f"  generated path:{path}")
        return path

    def _generate_direct_path(self, source: Coord, destination: Coord) -> List[Coord]:
        """Generate a continuous direct path from source to destination.

        For instruction paths (INODE to IMCE), this generates a horizontal path
        where only the column index changes. The path is continuous with each
        step incrementing the column by 1.

        Example: (0,0) to (0,3) -> [(0,0), (0,1), (0,2), (0,3)]

        Args:
            source: Starting coordinate
            destination: Ending coordinate

        Returns:
            List of coordinates forming a continuous path
        """
        path = [source]

        # Assume horizontal movement (same row, columns change)
        if source.row == destination.row:
            # Same row - horizontal path
            col_start = source.col
            col_end = destination.col
            step = 1 if col_end > col_start else -1
            for col in range(col_start + step, col_end + step, step):
                path.append(Coord(source.row, col))
        elif source.col == destination.col:
            # Same column - vertical path
            row_start = source.row
            row_end = destination.row
            step = 1 if row_end > row_start else -1
            for row in range(row_start + step, row_end + step, step):
                path.append(Coord(row, source.col))
        else:
            # Different row and column - use XY routing (horizontal first, then vertical)
            # Horizontal part
            col_start = source.col
            col_end = destination.col
            step = 1 if col_end > col_start else -1
            for col in range(col_start + step, col_end + step, step):
                path.append(Coord(source.row, col))
            # Vertical part
            row_start = source.row
            row_end = destination.row
            step = 1 if row_end > row_start else -1
            for row in range(row_start + step, row_end + step, step):
                path.append(Coord(row, destination.col))

        return path

    def get_all_commodity_ids(self) -> List[int]:
        """Get all commodity IDs"""
        return list(self._commodities.keys())

    def get_commodity(self, commodity_id: int):
        """Get commodity by ID"""
        return self._commodities.get(commodity_id)

    def get_path(self, commodity_id: int) -> List[Coord]:
        """Get path for commodity"""
        return self._paths.get(commodity_id, [])

    def get_noc_paths(self) -> Dict:
        """Get NoCPaths dict"""
        return self._noc_paths

    def get_ilp_commodity(self, hw_commodity_id: int):
        """Get the original ILP Commodity for a HWCommodity ID.

        Args:
            hw_commodity_id: The HWCommodity ID

        Returns:
            The corresponding ILP Commodity, or None if not found
        """
        return self._hw_to_ilp_commodity.get(hw_commodity_id)

    def is_multicast(self, hw_commodity_id: int) -> bool:
        """Check if the commodity is multicast.

        For non-multicast (e.g., DW conv split outputs), chunk selection (ksel)
        should not be applied in routing.

        Args:
            hw_commodity_id: The HWCommodity ID

        Returns:
            True if multicast, False otherwise. Defaults to True if not found.
        """
        ilp_comm = self._hw_to_ilp_commodity.get(hw_commodity_id)
        return ilp_comm.is_multicast if ilp_comm else True


def convert_pnr_result_to_routing_result(
    pnr_result: JointPnRResult,
    noc_paths: Dict,
) -> JointPnRRoutingResult:
    """
    Convert JointPnRResult to a routing result that PolicyTableBuilder can use.

    Args:
        pnr_result: Result from JointPnRILP
        noc_paths: NoCPaths dict containing both TensorEdge and instruction paths

    Returns:
        JointPnRRoutingResult for PolicyTableBuilder
    """
    commodities = extract_hw_commodities_from_noc_paths(noc_paths)
    routing_result = JointPnRRoutingResult(pnr_result, commodities, noc_paths)

    return routing_result


# ============================================================
# Visualization
# ============================================================

def visualize_pnr_mapping(
    pnr_result: JointPnRResult,
    func_name: str,
    output_path: str = None,
    topology: MeshTopology = None,
) -> str:
    """
    Visualize PnR mapping result using matplotlib.

    Creates a mesh grid showing:
    - IMCE nodes with mapped CALL/SPLIT/CONCAT nodes
    - INODE nodes with VAR, CONST, and FUNC_OUT assignments
    - Color coding by node type

    Args:
        pnr_result: Result from JointPnRILP
        func_name: Function name for title
        output_path: Path to save the figure. If None, auto-generates path.
        topology: MeshTopology (default: 4x5)

    Returns:
        Path to saved figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        debug_print("[visualize_pnr_mapping] matplotlib not available, skipping visualization")
        return None

    if topology is None:
        topology = MeshTopology(4, 5)

    rows = topology.rows
    cols = topology.cols

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(cols * 2.5, rows * 2.5))

    # Colors for different node types
    colors = {
        'CALL': '#4CAF50',      # Green
        'SPLIT': '#2196F3',     # Blue
        'CONCAT': '#9C27B0',    # Purple
        'VAR': '#FF9800',       # Orange
        'CONST': '#795548',     # Brown
        'FUNC_OUT': '#F44336',  # Red
        'EMPTY': '#E0E0E0',     # Light gray
        'INODE': '#BDBDBD',     # Gray
    }

    # Build reverse mapping: Coord -> list of (node_id, node_type)
    coord_to_nodes = {}

    # IMCE mappings (CALL, SPLIT, CONCAT)
    for node_id, coord in pnr_result.mapping.items():
        key = (coord.row, coord.col)
        if key not in coord_to_nodes:
            coord_to_nodes[key] = []
        # Determine node type from commodities
        node_type = 'CALL'  # Default
        for comm in pnr_result.commodities:
            if comm.source_node_id == node_id:
                if comm.source_type == NodeType.SPLIT:
                    node_type = 'SPLIT'
                elif comm.source_type == NodeType.CONCAT:
                    node_type = 'CONCAT'
                break
            if comm.dest_node_id == node_id:
                if comm.dest_type == NodeType.SPLIT:
                    node_type = 'SPLIT'
                elif comm.dest_type == NodeType.CONCAT:
                    node_type = 'CONCAT'
                break
        coord_to_nodes[key].append((node_id, node_type))

    # VAR mappings
    if pnr_result.var_to_inode:
        for node_id, coord in pnr_result.var_to_inode.items():
            key = (coord.row, coord.col)
            if key not in coord_to_nodes:
                coord_to_nodes[key] = []
            coord_to_nodes[key].append((node_id, 'VAR'))

    # CONST mappings
    if pnr_result.const_to_inode:
        for node_id, coord in pnr_result.const_to_inode.items():
            key = (coord.row, coord.col)
            if key not in coord_to_nodes:
                coord_to_nodes[key] = []
            coord_to_nodes[key].append((node_id, 'CONST'))

    # FUNC_OUT mappings
    if pnr_result.funcout_to_inode:
        for node_id, coord in pnr_result.funcout_to_inode.items():
            key = (coord.row, coord.col)
            if key not in coord_to_nodes:
                coord_to_nodes[key] = []
            coord_to_nodes[key].append((node_id, 'FUNC_OUT'))

    # Draw mesh grid
    for row in range(rows):
        for col in range(cols):
            x = col
            y = rows - 1 - row  # Flip y-axis so row 0 is at top

            is_inode = (col == 0)
            key = (row, col)
            nodes = coord_to_nodes.get(key, [])

            # Determine background color
            if nodes:
                # Use the type of the first "important" node
                priority = ['CALL', 'SPLIT', 'CONCAT', 'FUNC_OUT', 'VAR', 'CONST']
                bg_color = colors['EMPTY']
                for ptype in priority:
                    for _, ntype in nodes:
                        if ntype == ptype:
                            bg_color = colors[ptype]
                            break
                    if bg_color != colors['EMPTY']:
                        break
            else:
                bg_color = colors['INODE'] if is_inode else colors['EMPTY']

            # Draw cell
            rect = FancyBboxPatch(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=bg_color,
                edgecolor='black',
                linewidth=2 if nodes else 1,
            )
            ax.add_patch(rect)

            # Label: coordinate
            coord_label = f"({'I' if is_inode else 'M'}{row},{col})"
            ax.text(x, y + 0.35, coord_label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='#333333')

            # Label: node IDs
            if nodes:
                # Group by type and show counts or IDs
                node_strs = []
                for node_id, ntype in nodes[:4]:  # Show up to 4 nodes
                    short_type = ntype[0]  # First letter
                    node_strs.append(f"{short_type}:{node_id}")
                if len(nodes) > 4:
                    node_strs.append(f"+{len(nodes)-4} more")

                label = '\n'.join(node_strs)
                ax.text(x, y - 0.1, label, ha='center', va='center',
                       fontsize=7, color='white' if bg_color not in [colors['EMPTY'], colors['INODE']] else 'black')

    # Draw connections (edges between adjacent nodes)
    for row in range(rows):
        for col in range(cols):
            x = col
            y = rows - 1 - row
            # Horizontal edge (to East)
            if col < cols - 1:
                ax.plot([x + 0.45, x + 0.55], [y, y], 'k-', linewidth=0.5, alpha=0.3)
            # Vertical edge (to South)
            if row < rows - 1:
                ax.plot([x, x], [y - 0.45, y - 0.55], 'k-', linewidth=0.5, alpha=0.3)

    # Set axis properties
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    short_name = func_name.split('_')[-1] if '_' in func_name else func_name
    ax.set_title(f"PnR Mapping: {short_name}\n({rows}x{cols} Mesh)", fontsize=12, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['CALL'], edgecolor='black', label='CALL'),
        mpatches.Patch(facecolor=colors['SPLIT'], edgecolor='black', label='SPLIT'),
        mpatches.Patch(facecolor=colors['CONCAT'], edgecolor='black', label='CONCAT'),
        mpatches.Patch(facecolor=colors['VAR'], edgecolor='black', label='VAR'),
        mpatches.Patch(facecolor=colors['CONST'], edgecolor='black', label='CONST'),
        mpatches.Patch(facecolor=colors['FUNC_OUT'], edgecolor='black', label='FUNC_OUT'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        config = ImcflowDeviceConfig()
        eval_dir = config.eval_dir
        output_path = os.path.join(eval_dir, f"pnr_mapping_{short_name}.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    debug_print(f"[visualize_pnr_mapping] Saved mapping visualization to {output_path}")
    return output_path


def visualize_all_pnr_results(
    pnr_results: Dict[str, JointPnRResult],
    output_dir: str = None,
) -> List[str]:
    """
    Visualize all PnR results.

    Args:
        pnr_results: Dict mapping func_name -> JointPnRResult
        output_dir: Directory to save figures

    Returns:
        List of saved figure paths
    """
    if output_dir is None:
        config = ImcflowDeviceConfig()
        output_dir = config.eval_dir

    paths = []
    for func_name, result in pnr_results.items():
        if not result.success:
            continue
        short_name = func_name.split('_')[-1] if '_' in func_name else func_name
        output_path = os.path.join(output_dir, f"pnr_mapping_{short_name}.png")
        path = visualize_pnr_mapping(result, func_name, output_path)
        if path:
            paths.append(path)

    return paths


# ============================================================
# Full Integration Entry Point
# ============================================================

def run_joint_pnr_and_update_config(
    mod: tvm.IRModule,
    tensor_edge_list_dict: Dict[str, List],
    preassigned_placements: Optional[Dict[str, Dict[Any, Coord]]] = None,
) -> Dict[str, JointPnRResult]:
    """
    Run joint PnR and update ImcflowDeviceConfig with results.

    This is the high-level integration function that:
    1. Runs Joint PnR ILP for all imcflow functions
    2. Updates HWNodeMap with placement results

    Args:
        mod: TVM module containing imcflow functions
        tensor_edge_list_dict: Dict mapping func_name -> List[TensorEdge]
                               Must be constructed via constructTensorEdgeList() first.
        preassigned_placements: Optional dict mapping func_name -> {graph_node_id -> Coord}
                                for nodes with pre-determined placements.

    Returns:
        Dict mapping function names to JointPnRResult
    """
    config = ImcflowDeviceConfig()

    # Run joint PnR
    results = run_joint_pnr(mod, tensor_edge_list_dict, preassigned_placements)

    # Update HWNodeMap (pass tensor_edge_list_dict for outer->inner id conversion)
    update_hw_node_map(results, config.HWNodeMap, tensor_edge_list_dict)

    # dump hardware node map
    for k, v in config.HWNodeMap.items():
        debug_print(f"[HWNodeMap] Node {k} -> {v}")

    debug_print(f"[run_joint_pnr_and_update_config] Updated HWNodeMap with "
               f"{len(config.HWNodeMap)} mappings")

    # Visualize PnR mapping
    visualize_all_pnr_results(results, os.environ.get("IMCFLOW_EVAL_DIR", "/tmp/imcflow"))

    return results


def build_policy_tables_from_pnr_result(
    pnr_result: JointPnRResult,
    func_name: str,
    noc_paths: Dict,
    table_capacity: int = 32,
) -> Dict:
    """
    Build policy tables from JointPnRResult using PolicyTableBuilder.

    This function:
    1. Converts PnR result to routing result format (extracts HWCommodities from noc_paths)
    2. Calls PolicyTableBuilder.build() to generate policy tables

    Args:
        pnr_result: Result from JointPnRILP for this function
        func_name: Function name
        noc_paths: NoCPaths dict containing both TensorEdge and instruction paths
        table_capacity: Max entries per node policy table

    Returns:
        Policy tables dict
    """
    from .policy_table_builder import PolicyTableBuilder

    debug_print(f"[build_policy_tables_from_pnr_result] Starting for {func_name}")
    debug_print(f"[build_policy_tables_from_pnr_result]   noc_paths keys: "
               f"{len(noc_paths)} entries")
    tensor_edge_keys = [k for k in noc_paths.keys() if isinstance(k, TensorEdge)]
    inst_path_keys = [k for k in noc_paths.keys() if isinstance(k, NodeID)]
    debug_print(f"[build_policy_tables_from_pnr_result]   TensorEdge paths: {len(tensor_edge_keys)}")
    debug_print(f"[build_policy_tables_from_pnr_result]   Instruction paths (NodeID): {len(inst_path_keys)} -> {[str(k) for k in inst_path_keys]}")
    for te in tensor_edge_keys:
        src, dst, split = noc_paths[te]
        debug_print(f"[build_policy_tables_from_pnr_result]   DATA: {te} -> ({src}, {dst}, {split})")

    # Convert to routing result (extracts HWCommodities from noc_paths)
    routing_result = convert_pnr_result_to_routing_result(pnr_result, noc_paths)

    # Build policy tables
    builder = PolicyTableBuilder(table_capacity=table_capacity)
    policy_tables = builder.build(routing_result, noc_paths, func_name)

    debug_print(f"[build_policy_tables_from_pnr_result] Built policy tables for {func_name}")

    return policy_tables


# ============================================================
# NoCPaths Construction from Joint PnR Results
# ============================================================

def construct_noc_paths_from_pnr_results(
    pnr_results: Dict[str, JointPnRResult],
    tensor_edge_list_dict: Dict[str, List],
) -> Dict[str, Dict]:
    """
    Construct NoCPaths dict from Joint PnR results.

    This replaces the legacy constructNoCPathDict() which used getInnerNodeID()
    to lookup in HWNodeMap (problematic for composite nodes).

    Args:
        pnr_results: Results from run_joint_pnr()
        tensor_edge_list_dict: Dict mapping func_name -> List[TensorEdge]

    Returns:
        Dict mapping func_name -> {tensor_edge: (src_hwnode, dst_hwnode, split_idx)}
    """
    config = ImcflowDeviceConfig()
    noc_paths = config.NoCPaths

    for func_name, pnr_result in pnr_results.items():
        if not pnr_result.success:
            debug_print(f"[construct_noc_paths] Skipping failed function: {func_name}")
            continue

        noc_paths[func_name] = {}

        # Build commodity_id -> tensor_edge mapping from commodities
        commodity_id_to_edge = {}
        for commodity in pnr_result.commodities:
            if commodity.metadata is not None:
                # metadata is the original TensorEdge
                commodity_id_to_edge[commodity.id] = commodity.metadata

        # For each tensor edge, get the src/dst placements
        tensor_edge_list = tensor_edge_list_dict.get(func_name, [])
        hw_node_map = config.HWNodeMap

        # Build node_id -> Commodity mapping to check node types
        commodity_by_src = {}
        commodity_by_dest = {}
        for commodity in pnr_result.commodities:
            commodity_by_src[commodity.source_node_id] = commodity
            commodity_by_dest[commodity.dest_node_id] = commodity

        for tensor_edge in tensor_edge_list:
            src_tensor_id = tensor_edge.src_id
            dst_tensor_id = tensor_edge.dst_id
            split_idx = tensor_edge.split_idx

            # Check if source is a constant inside composite (tuple with negative inner_id)
            src_gid = src_tensor_id.graph_node_id
            dst_gid = dst_tensor_id.graph_node_id

            src_is_const_in_composite = (
                isinstance(src_gid, tuple) and
                len(src_gid) >= 2 and
                src_gid[-1] < 0
            )

            if src_is_const_in_composite:
                # Constant inside composite: use inner_id for src, outer_id for dst
                const_inner_id = src_gid[-1]
                dst_graph_id = getOuterNodeID(dst_gid)

                src_hwnode = hw_node_map.get(const_inner_id)
                dst_hwnode = hw_node_map.get(dst_graph_id)

                if src_hwnode is None:
                    raise KeyError(f"const_inner_id {const_inner_id} not found in HWNodeMap")
                if dst_hwnode is None:
                    raise KeyError(f"dst_graph_id {dst_graph_id} not found in HWNodeMap")

                if isinstance(dst_hwnode, tuple):
                    dst_hwnode = dst_hwnode[split_idx] if split_idx is not None else dst_hwnode[0]
                    split_idx = None
                noc_paths[func_name][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)
                continue

            # Get graph node IDs (outer ID for composites)
            src_graph_id = getOuterNodeID(src_gid)
            dst_graph_id = getOuterNodeID(dst_gid)
            dst_graph_node = CustomIDToNode()[dst_graph_id] 
            if isinstance(dst_graph_node, relay.Function):
              if split_idx is not None:
                dst_graph_id = (getOuterNodeID(dst_gid), split_idx) # func out node with tuple idx
              else:
                dst_graph_id = (getOuterNodeID(dst_gid), 0)

            # Check node types from commodities
            src_commodity = commodity_by_src.get(src_graph_id)
            dst_commodity = commodity_by_dest.get(dst_graph_id)

            src_is_var = (src_commodity is not None and src_commodity.source_type == NodeType.VAR)
            src_is_const = (src_commodity is not None and src_commodity.source_type == NodeType.CONST)
            dst_is_funcout = (dst_commodity is not None and dst_commodity.dest_type == NodeType.FUNC_OUT)

            # Look up source coordinate based on node type
            if src_is_var:
                src_coord = pnr_result.var_to_inode.get(src_graph_id)
                if src_coord is None:
                    raise KeyError(f"VAR src_graph_id {src_graph_id} not found in var_to_inode")
            elif src_is_const:
                src_coord = pnr_result.const_to_inode.get(src_graph_id)
                if src_coord is None:
                    raise KeyError(f"CONST src_graph_id {src_graph_id} not found in const_to_inode")
            else:
                src_coord = pnr_result.mapping.get(src_graph_id)
                if src_coord is None:
                    raise KeyError(f"src_graph_id {src_graph_id} not found in mapping")

            # Look up destination coordinate based on node type
            if dst_is_funcout:
                dst_coord = pnr_result.funcout_to_inode.get(dst_graph_id)
                if dst_coord is None:
                    raise KeyError(f"FUNC_OUT dst_graph_id {dst_graph_id} not found in funcout_to_inode")
            else:
                dst_coord = pnr_result.mapping.get(dst_graph_id)
                if dst_coord is None:
                    raise KeyError(f"dst_graph_id {dst_graph_id} not found in mapping")

            # Convert Coord to NodeID
            src_hwnode = coord_to_node_id(src_coord)
            dst_hwnode = coord_to_node_id(dst_coord)

            noc_paths[func_name][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)

        # Add instruction paths (inode to imce)
        # Each IMCE receives instructions from its corresponding INODE
        # NOTE: We add instruction paths for ALL IMCEs unconditionally (same as legacy constructNoCPathDict)
        # This means even IMCEs not mapped for this function get instruction path entries,
        # which is by design - inodes broadcast instructions to all IMCEs in their row.
        active_imces = set(
            v for v in config.HWNodeMap.values()
            if isinstance(v, NodeID) and v.is_imce()
        )
        debug_print(f"[construct_noc_paths] {func_name}: active IMCEs in HWNodeMap = {sorted([str(x) for x in active_imces])}")
        for dst_hwnode_id in NodeID.imces():
            inode_id = NodeID.from_inode_coord(NodeID.to_coord(dst_hwnode_id)[0])
            noc_paths[func_name][dst_hwnode_id] = (inode_id, dst_hwnode_id, None)
            is_active = dst_hwnode_id in active_imces
            debug_print(f"[construct_noc_paths]   inst path: {inode_id} -> {dst_hwnode_id} (active={is_active})")

        debug_print(f"[construct_noc_paths] {func_name}: {len(noc_paths[func_name])} entries total "
                   f"({len(noc_paths[func_name]) - len(NodeID.imces())} TensorEdge + {len(NodeID.imces())} instruction)")

    return noc_paths