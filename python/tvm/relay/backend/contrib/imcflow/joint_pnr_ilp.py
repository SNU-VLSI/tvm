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
# Debug Utilities
# ============================================================

_DEBUG_ENABLED = None

def _is_debug_enabled():
    global _DEBUG_ENABLED
    if _DEBUG_ENABLED is None:
        debug_var = os.environ.get('IMCFLOW_DEBUG', '0')
        _DEBUG_ENABLED = debug_var == '1' or debug_var.lower() == 'true'
    return _DEBUG_ENABLED

def debug_print(*args, **kwargs):
    """Print debug message only if IMCFLOW_DEBUG is enabled"""
    if _is_debug_enabled():
        print(*args, **kwargs)


# ============================================================
# Data Structures (reuse patterns from mcf_router.py)
# ============================================================

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

    def get_neighbors(self, coord: Coord) -> List[Coord]:
        """Get valid neighbors of a node"""
        neighbors = []
        # North
        if coord.row > 0:
            neighbors.append(Coord(coord.row - 1, coord.col))
        # South
        if coord.row < self.rows - 1:
            neighbors.append(Coord(coord.row + 1, coord.col))
        # West
        if coord.col > 0:
            neighbors.append(Coord(coord.row, coord.col - 1))
        # East
        if coord.col < self.cols - 1:
            neighbors.append(Coord(coord.row, coord.col + 1))
        return neighbors

    def get_all_edges(self) -> List[Edge]:
        """Get all directed edges in the mesh"""
        edges = []
        for node in self.get_all_nodes():
            for neighbor in self.get_neighbors(node):
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
    var_to_inode: Dict[Any, Coord] = None     # var node_id -> INODE Coord
    funcout_to_inode: Dict[Any, Coord] = None # funcout node_id -> INODE Coord


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

        # Build use-def chain first
        self._build_use_def_chain(func)

        # Visit function to extract nodes
        self._visit(func)

        for node_id, node in self.nodes.items():
            debug_print(f"[GraphExtractor] Node ID: {node_id}, Type: {node.node_type}, Topo Order: {node.topo_order}")
            debug_print(f"  Name : {CustomIDToName()[getInnerNodeID(node_id)]}")

        # Extract graph_commodities (data flows)
        self._extract_graph_commodities_from_tensor_edge_list(tensor_edge_list)

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
            self.topo_order += 1
            self.nodes[node_id] = GraphNode(
                id=node_id,
                node_type=NodeType.FUNC_OUT,
                relay_expr=fn,
                topo_order=self.topo_order,
            )
            # Assign to funcout INODE
            if node_id not in self.funcout_to_inode:
                self.funcout_to_inode[node_id] = next(self.funcout_inode_iter)

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

    def _extract_graph_commodities_from_tensor_edge_list(self, tensor_edge_list: List):
        """Extract commodities from TensorEdgeList.

        Uses validated TensorEdgeList instead of parsing relay function.
        This is more robust as the edge information is already validated
        and tensor_type is accurate.

        Args:
            tensor_edge_list: List of TensorEdge from constructTensorEdgeList()
        """
        for edge in tensor_edge_list:
            src_graph_id = edge.src_id.graph_node_id
            dst_graph_id = edge.dst_id.graph_node_id

            # For composite nodes, use outer_id (the composite call's ID)
            # TensorEdgeList uses tuple (outer_id, inner_id) for composite internals
            src_node_id = getOuterNodeID(src_graph_id)
            dst_node_id = getOuterNodeID(dst_graph_id)

            # Look up nodes from GraphExtractor's extracted nodes
            # GraphExtractor now uses custom_id as key for Call nodes
            src_node = self.nodes.get(src_node_id)
            dst_node = self.nodes.get(dst_node_id)

            # Skip if nodes not found (constants, vars, or internal nodes)
            if src_node is None or dst_node is None:
                # Only log if both are positive (should be tracked Call nodes)
                if src_node_id > 0 and dst_node_id > 0:
                    debug_print(f"[_extract_commodities_from_tensor_edge_list] "
                               f"Node not found in GraphExtractor: "
                               f"src_found={src_node is not None}, dst_found={dst_node is not None}, "
                               f"node_ids=({src_node_id}, {dst_node_id})")
                continue

            # Add commodity with validated tensor_type from TensorEdgeList
            self.add_commodity(
                src_node_id, dst_node_id,
                src_node.node_type, dst_node.node_type,
                tensor_type=edge.src_id.tensor_type,
                split_idx=edge.split_idx,
                metadata=edge
            )

        debug_print(f"[_extract_commodities_from_tensor_edge_list] "
                   f"Extracted {len(self.commodities)} commodities from "
                   f"{len(tensor_edge_list)} tensor edges")
        for comm in self.commodities:
            debug_print(f"  Commodity {comm.id}: "
                       f"{comm.source_node_id} ({comm.source_type}) -> "
                       f"{comm.dest_node_id} ({comm.dest_type}), "
                       f"type={comm.tensor_type}")

    def add_commodity(self, src_node_id: Any, dst_node_id: Any,
                      src_type: NodeType, dst_type: NodeType,
                      tensor_type: str, split_idx: Optional[int] = None,
                      metadata: Any = None):
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
        )
        self.commodities.append(commodity)
        self.commodity_id += 1
        return commodity


# ============================================================
# Joint PnR ILP Solver
# ============================================================

class JointPnRILP:
    """Joint node mapping and routing ILP solver"""

    def __init__(self, topology: MeshTopology = None):
        self.topology = topology or MeshTopology(4, 5)

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

                try:
                    tensor_edge_list = tensor_edge_list_dict[func_name]
                    result = self._run_for_function(actual_func, func_name, tensor_edge_list)
                    results[func_name] = result

                    if result.success:
                        debug_print(f"[JointPnRILP] Success: max_cong={result.max_congestion}, "
                                   f"hops={result.total_hops}")
                    else:
                        debug_print(f"[JointPnRILP] Failed: {result.solver_status}")

                except Exception as e:
                    logger.error(f"JointPnRILP failed for {func_name}: {e}")
                    results[func_name] = JointPnRResult(
                        mapping={},
                        routes={},
                        commodities=[],
                        max_congestion=0,
                        total_hops=0,
                        solver_status=f"Error: {e}",
                        success=False,
                        var_to_inode={},
                        funcout_to_inode={},
                    )

        return results

    def _run_for_function(
        self,
        func: relay.Function,
        func_name: str,
        tensor_edge_list: List,
    ) -> JointPnRResult:
        """Run joint PnR for a single function.

        Args:
            func: Relay function
            func_name: Name of the function
            tensor_edge_list: List of TensorEdge from constructTensorEdgeList()

        Returns:
            JointPnRResult with mapping and routes
        """
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

        # All placeable nodes (call + split + concat)
        placeable_nodes = gi.call_nodes + gi.split_nodes + gi.concat_nodes

        # p[n][v]: placement of call/split/concat node n at IMCE v
        self.p = {}
        for n in placeable_nodes:
            self.p[n] = {}
            for v in self.imce_nodes:
                self.p[n][v] = pulp.LpVariable(
                    f"p_{hash(n) % 100000}_{v.row}_{v.col}",
                    cat=pulp.LpBinary
                )

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
        self.multicast_groups = {}  # group_key -> [commodity_ids]
        for k in gi.commodities:
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

        # P1: Each placeable node -> exactly one IMCE
        for n in gi.call_nodes + gi.split_nodes + gi.concat_nodes:
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

        # P3: Split same as producer
        for split_id in gi.split_nodes:
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

            # L3: Var source (fixed to inode)
            if src_node.node_type == NodeType.VAR:
                # Assign to INODE based on hash (round-robin between row 0 and 1)
                var_inode = Coord(0 if hash(k.source_node_id) % 2 == 0 else 1, 0)
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

            # L4: Func_out dest (fixed to inode)
            if dst_node.node_type == NodeType.FUNC_OUT:
                funcout_inode = Coord(3 if hash(k.dest_node_id) % 2 == 0 else 2, 0)
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
            )

        # Extract placement
        mapping = {}
        gi = self.graph_info

        for n in gi.call_nodes + gi.split_nodes + gi.concat_nodes:
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

        return JointPnRResult(
            mapping=mapping,
            routes=routes,
            commodities=gi.commodities,
            max_congestion=max_congestion,
            total_hops=total_hops,
            solver_status=status_str,
            success=True,
            var_to_inode=gi.var_to_inode,
            funcout_to_inode=gi.funcout_to_inode,
        )


# ============================================================
# Integration with existing infrastructure
# ============================================================

def run_joint_pnr(
    mod: tvm.IRModule,
    tensor_edge_list_dict: Dict[str, List],
) -> Dict[str, JointPnRResult]:
    """
    Run joint PnR for all imcflow functions in module.

    This is the main entry point for the joint PnR solver.

    Args:
        mod: TVM module containing imcflow functions
        tensor_edge_list_dict: Dict mapping func_name -> List[TensorEdge]
                               Must be constructed via constructTensorEdgeList() first.

    Returns:
        Dict mapping function names to JointPnRResult
    """
    solver = JointPnRILP(topology=MeshTopology(4, 5))
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
    - Constants inside composites: use inner_id -> INODE
    - Regular calls: use custom_id
    - Vars/FuncOut: use custom_id

    Args:
        results: Results from run_joint_pnr
        hw_node_map: ImcflowDeviceConfig().HWNodeMap to update
        tensor_edge_list_dict: TensorEdgeListDict for composite inner_id mapping
    """
    from tvm.relay.backend.contrib.imcflow import CustomIDToNode

    id_to_node = CustomIDToNode()

    for func_name, result in results.items():
        if not result.success:
            raise RuntimeError(f"Joint PnR failed for {func_name}: {result.solver_status}")

        # Build outer_id -> inner_id mapping from TensorEdgeList
        # Also collect constants inside composites for INODE mapping
        outer_to_inner_call = {}  # outer_id -> inner_call_id
        const_inner_ids = {}      # const_inner_id -> outer_id (for INODE mapping)
        tensor_edge_list = tensor_edge_list_dict.get(func_name, [])
        for edge in tensor_edge_list:
            for tensor_id in [edge.src_id, edge.dst_id]:
                gid = tensor_id.graph_node_id
                if isinstance(gid, tuple):
                    outer_id, inner_id = gid[0], gid[1]
                    # Check if inner_id is a Constant using CustomIDToNode
                    node = id_to_node.get(inner_id)
                    if node is not None and isinstance(node, relay.Constant):
                        const_inner_ids[inner_id] = outer_id
                    else:
                        # Inner call inside composite
                        outer_to_inner_call[outer_id] = inner_id

        # IMCE mappings (call/split/concat nodes)
        for graph_node_id, coord in result.mapping.items():
            # For composite nodes, use inner_call_id as key
            inner_id = outer_to_inner_call.get(graph_node_id, graph_node_id)
            # Convert Coord to NodeID
            node_id = NodeID.from_imce_coord(coord.row, coord.col - 1)
            hw_node_map[inner_id] = node_id

        # INODE mappings (input vars)
        if result.var_to_inode:
            for var_node_id, coord in result.var_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[var_node_id] = node_id

        # INODE mappings (funcout)
        if result.funcout_to_inode:
            for funcout_node_id, coord in result.funcout_to_inode.items():
                node_id = NodeID.from_inode_coord(coord.row)
                hw_node_map[funcout_node_id] = node_id

        # INODE mappings for constants inside composites
        # Constants load data from INODE at the same row as their composite's IMCE
        for const_inner_id, outer_id in const_inner_ids.items():
            if const_inner_id not in hw_node_map and outer_id in result.mapping:
                composite_coord = result.mapping[outer_id]
                node_id = NodeID.from_inode_coord(composite_coord.row)
                hw_node_map[const_inner_id] = node_id


def coord_to_node_id(coord: Coord) -> NodeID:
    """Convert Coord to NodeID"""
    if coord.col == 0:
        # INODE
        return NodeID.from_inode_coord(coord.row)
    else:
        # IMCE
        return NodeID.from_imce_coord(coord.row, coord.col - 1)

def extract_commodities_from_tensor_edge_list(
    tensor_edge_list: Dict,
    hw_node_map: Dict,
) -> Tuple[List['Commodity'], Dict]:
    """
    Extract commodities from TensorEdgeList for routing.

    This function converts TensorEdgeList (from DevConfig().TensorEdgeListDict)
    into Commodity objects that the ILP solver or PolicyTableBuilder can use.

    Args:
        tensor_edge_list: Dict of TensorEdge -> mapping_info from TensorEdgeListDict
        hw_node_map: Current HWNodeMap for resolving node locations

    Returns:
        Tuple of (commodities list, noc_paths dict for PolicyTableBuilder)
    """
    from .mcf_router import Commodity as MCFCommodity

    commodities = []
    noc_paths = {}
    commodity_id = 0

    for edge, mapping_info in tensor_edge_list.items():
        src_node = mapping_info[0]  # NodeID
        dst_node = mapping_info[1]  # NodeID
        split_idx = mapping_info[2] if len(mapping_info) > 2 else None

        # Convert NodeID to Coord
        src_coord_tuple = NodeID.to_coord(src_node)
        dst_coord_tuple = NodeID.to_coord(dst_node)

        src_coord = Coord(src_coord_tuple[0], src_coord_tuple[1])
        dst_coord = Coord(dst_coord_tuple[0], dst_coord_tuple[1])

        # Skip local edges (same node)
        if src_coord == dst_coord:
            noc_paths[edge] = mapping_info
            continue

        # Create commodity
        commodity = MCFCommodity(
            id=commodity_id,
            source=src_coord,
            destination=dst_coord,
            metadata=(edge, mapping_info)
        )
        commodities.append(commodity)
        noc_paths[edge] = mapping_info
        commodity_id += 1

    debug_print(f"[extract_commodities] Extracted {len(commodities)} commodities "
               f"from {len(tensor_edge_list)} tensor edges")

    return commodities, noc_paths


# ============================================================
# Routes to BaseRoutingResult Conversion
# ============================================================

class JointPnRRoutingResult:
    """
    Adapter class to convert JointPnRResult routes to BaseRoutingResult interface.

    This allows PolicyTableBuilder to use routes from JointPnRILP directly.
    """

    def __init__(self, pnr_result: JointPnRResult, commodities: List, noc_paths: Dict):
        """
        Initialize from JointPnRResult.

        Args:
            pnr_result: Result from JointPnRILP
            commodities: List of MCF Commodity objects (from extract_commodities_from_tensor_edge_list)
            noc_paths: NoCPaths dict
        """
        from .mcf_router import Commodity as MCFCommodity

        self._commodities = {}
        self._paths = {}
        self._noc_paths = noc_paths

        # Build commodity lookup by id
        for commodity in commodities:
            self._commodities[commodity.id] = commodity

            # Convert edges to path (list of coords)
            if commodity.id in pnr_result.routes:
                edges = pnr_result.routes[commodity.id]
                path = self._edges_to_path(commodity.source, edges)
                self._paths[commodity.id] = path
            else:
                # No route found - use direct path (shouldn't happen for valid result)
                self._paths[commodity.id] = [commodity.source, commodity.destination]

    def _edges_to_path(self, start: Coord, edges: List[Edge]) -> List[Coord]:
        """Convert list of edges to path (list of coords)"""
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


def convert_pnr_result_to_routing_result(
    pnr_result: JointPnRResult,
    tensor_edge_list: Dict,
    hw_node_map: Dict,
) -> Tuple[JointPnRRoutingResult, Dict]:
    """
    Convert JointPnRResult to a routing result that PolicyTableBuilder can use.

    Args:
        pnr_result: Result from JointPnRILP
        tensor_edge_list: TensorEdgeList dict
        hw_node_map: HWNodeMap dict

    Returns:
        Tuple of (JointPnRRoutingResult, noc_paths)
    """
    commodities, noc_paths = extract_commodities_from_tensor_edge_list(
        tensor_edge_list, hw_node_map
    )

    routing_result = JointPnRRoutingResult(pnr_result, commodities, noc_paths)

    return routing_result, noc_paths


# ============================================================
# Full Integration Entry Point
# ============================================================

def run_joint_pnr_and_update_config(
    mod: tvm.IRModule,
    tensor_edge_list_dict: Dict[str, List],
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

    Returns:
        Dict mapping function names to JointPnRResult
    """
    config = ImcflowDeviceConfig()

    # Run joint PnR
    results = run_joint_pnr(mod, tensor_edge_list_dict)

    # Update HWNodeMap (pass tensor_edge_list_dict for outer->inner id conversion)
    update_hw_node_map(results, config.HWNodeMap, tensor_edge_list_dict)

    # dump hardware node map
    for k, v in config.HWNodeMap.items():
        debug_print(f"[HWNodeMap] Node {k} -> {v}")

    debug_print(f"[run_joint_pnr_and_update_config] Updated HWNodeMap with "
               f"{len(config.HWNodeMap)} mappings")

    return results


def build_policy_tables_from_pnr_result(
    pnr_result: JointPnRResult,
    func_name: str,
    tensor_edge_list: Dict,
    hw_node_map: Dict,
    table_capacity: int = 32,
) -> Dict:
    """
    Build policy tables from JointPnRResult using PolicyTableBuilder.

    This function:
    1. Converts PnR result to routing result format
    2. Calls PolicyTableBuilder.build() to generate policy tables

    Args:
        pnr_result: Result from JointPnRILP for this function
        func_name: Function name
        tensor_edge_list: TensorEdgeList for this function
        hw_node_map: HWNodeMap dict
        table_capacity: Max entries per node policy table

    Returns:
        Policy tables dict
    """
    from .policy_table_builder import PolicyTableBuilder

    # Convert to routing result
    routing_result, noc_paths = convert_pnr_result_to_routing_result(
        pnr_result, tensor_edge_list, hw_node_map
    )

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
        for tensor_edge in tensor_edge_list:
            src_tensor_id = tensor_edge.src_id
            dst_tensor_id = tensor_edge.dst_id
            split_idx = tensor_edge.split_idx

            # Get graph node IDs (outer ID for composites)
            src_graph_id = getOuterNodeID(src_tensor_id.graph_node_id)
            dst_graph_id = getOuterNodeID(dst_tensor_id.graph_node_id)

            # Look up in mapping
            src_coord = pnr_result.mapping.get(src_graph_id)
            dst_coord = pnr_result.mapping.get(dst_graph_id)

            if src_coord is None or dst_coord is None:
                # Node not mapped (could be input var, constant, or funcout)
                # Try to get from HWNodeMap as fallback
                hw_node_map = config.HWNodeMap
                src_hwnode = hw_node_map.get(src_graph_id)
                dst_hwnode = hw_node_map.get(dst_graph_id)

                if src_hwnode is not None and dst_hwnode is not None:
                    # Handle split index for tuple hwnode
                    if isinstance(dst_hwnode, tuple):
                        dst_hwnode = dst_hwnode[split_idx] if split_idx is not None else dst_hwnode[0]
                        split_idx = None
                    noc_paths[func_name][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)
                continue

            # Convert Coord to NodeID
            src_hwnode = coord_to_node_id(src_coord)
            dst_hwnode = coord_to_node_id(dst_coord)

            noc_paths[func_name][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)

        # Add instruction paths (inode to imce)
        # Each IMCE receives instructions from its corresponding INODE
        for dst_hwnode_id in NodeID.imces():
            inode_id = NodeID.from_inode_coord(NodeID.to_coord(dst_hwnode_id)[0])
            noc_paths[func_name][dst_hwnode_id] = (inode_id, dst_hwnode_id, None)

        debug_print(f"[construct_noc_paths] {func_name}: {len(noc_paths[func_name])} entries")

    return noc_paths


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    def run_tests():
        all_passed = True

        print("=" * 70)
        print("Test 1: Basic topology")
        print("=" * 70)

        topo = MeshTopology(4, 5)
        print(f"All nodes: {len(topo.get_all_nodes())}")
        print(f"IMCE nodes: {len(topo.get_imce_nodes())}")
        print(f"INODE nodes: {len(topo.get_inode_nodes())}")
        print(f"All edges: {len(topo.get_all_edges())}")

        assert len(topo.get_all_nodes()) == 20
        assert len(topo.get_imce_nodes()) == 16
        assert len(topo.get_inode_nodes()) == 4
        print("✓ PASS: Topology creation works")

        print("\n" + "=" * 70)
        print("Test 2: Commodity congestion groups")
        print("=" * 70)

        k1 = Commodity(0, 'src', 'dst', NodeType.CALL, NodeType.CALL, 'data')
        k2 = Commodity(1, 'src', 'dst', NodeType.CALL, NodeType.CALL, 'weight')
        k3 = Commodity(2, 'src', 'dst', NodeType.VAR, NodeType.CALL, 'var')

        assert k1.get_congestion_group() == 'data'
        assert k2.get_congestion_group() == 'const'
        assert k3.get_congestion_group() == 'data'
        print("✓ PASS: Congestion group classification")

        print("\n" + "=" * 70)
        print("Test 3: Build and solve ILP with mock graph")
        print("=" * 70)

        # Create a simple mock graph info
        nodes = {
            'call1': GraphNode('call1', NodeType.CALL, topo_order=1),
            'call2': GraphNode('call2', NodeType.CALL, topo_order=2),
            'var1': GraphNode('var1', NodeType.VAR, topo_order=0),
            'func_out': GraphNode('func_out', NodeType.FUNC_OUT, topo_order=3),
        }

        commodities = [
            Commodity(0, 'var1', 'call1', NodeType.VAR, NodeType.CALL, 'data'),
            Commodity(1, 'call1', 'call2', NodeType.CALL, NodeType.CALL, 'data'),
            Commodity(2, 'call2', 'func_out', NodeType.CALL, NodeType.FUNC_OUT, 'data'),
        ]

        graph_info = GraphInfo(
            nodes=nodes,
            commodities=commodities,
            call_nodes=['call1', 'call2'],
            split_nodes=[],
            concat_nodes=[],
            var_nodes=['var1'],
            const_nodes=[],
            funcout_nodes=['func_out'],
        )

        # Create solver and set graph info
        solver = JointPnRILP()
        solver.graph_info = graph_info

        # Build and solve
        solver._build_model()
        result = solver._solve()

        print(f"  Solver status: {result.solver_status}")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Mapping: {result.mapping}")
            print(f"  Routes: {len(result.routes)} commodities")
            print(f"  Max congestion: {result.max_congestion}")
            print(f"  Total hops: {result.total_hops}")
            print("✓ PASS: ILP solved successfully")
        else:
            print("✗ FAIL: ILP failed to solve")
            all_passed = False

        print("\n" + "=" * 70)
        print("Test 4: Larger graph with multiple data flows")
        print("=" * 70)

        # 4 call nodes in a chain
        nodes2 = {
            'v1': GraphNode('v1', NodeType.VAR, topo_order=0),
            'c1': GraphNode('c1', NodeType.CALL, topo_order=1),
            'c2': GraphNode('c2', NodeType.CALL, topo_order=2),
            'c3': GraphNode('c3', NodeType.CALL, topo_order=3),
            'c4': GraphNode('c4', NodeType.CALL, topo_order=4),
            'fo': GraphNode('fo', NodeType.FUNC_OUT, topo_order=5),
        }

        commodities2 = [
            Commodity(0, 'v1', 'c1', NodeType.VAR, NodeType.CALL, 'data'),
            Commodity(1, 'c1', 'c2', NodeType.CALL, NodeType.CALL, 'data'),
            Commodity(2, 'c2', 'c3', NodeType.CALL, NodeType.CALL, 'data'),
            Commodity(3, 'c3', 'c4', NodeType.CALL, NodeType.CALL, 'data'),
            Commodity(4, 'c4', 'fo', NodeType.CALL, NodeType.FUNC_OUT, 'data'),
        ]

        graph_info2 = GraphInfo(
            nodes=nodes2,
            commodities=commodities2,
            call_nodes=['c1', 'c2', 'c3', 'c4'],
            split_nodes=[],
            concat_nodes=[],
            var_nodes=['v1'],
            const_nodes=[],
            funcout_nodes=['fo'],
        )

        solver2 = JointPnRILP()
        solver2.graph_info = graph_info2
        solver2._build_model()
        result2 = solver2._solve()

        print(f"  Solver status: {result2.solver_status}")
        if result2.success:
            print(f"  Placed {len(result2.mapping)} call nodes")
            print(f"  Total hops: {result2.total_hops}")
            print("✓ PASS: Larger graph solved")
        else:
            print("✗ FAIL: Larger graph failed")
            all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("=" * 70)

        return all_passed

    run_tests()
