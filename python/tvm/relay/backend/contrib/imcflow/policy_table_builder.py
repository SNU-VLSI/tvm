"""
Policy Table Builder for NoC Routing

This module provides a unified interface for building policy tables from routing results.
It combines:
- Phase 2: Path tree building (multicast tree construction)
- Phase 3: Policy table generation + EdgeInfo + Memory allocation

Key classes:
- PolicyTableBuilder: Top-level class that orchestrates the entire pipeline
- PathTreeBuilder: Builds multicast trees from routing results
- EdgeInfoGenerator: Generates EdgeInfo from policy tables
- MemoryAllocator: Allocates memory for policy tables
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import tvm
from tvm import relay
from tvm.contrib.imcflow import (
    ImcflowDeviceConfig,
    TensorEdge,
    TensorID,
    NodeID,
    TensorEdgeInfo,
    InstEdgeInfo,
    RouterEntry,
    DataBlock,
)
from tvm.relay.op.contrib.imcflow import CustomIDToName, CustomIDToNode
from tvm.relay.backend.contrib.imcflow.transform import debug_print

from .joint_pnr_ilp import (
    Coord,
    Direction,
    HWCommodity,
)

logger = logging.getLogger(__name__)


# Helper function
def getInnerNodeID(node):
    """Extract inner node ID from potentially nested tuple."""
    if isinstance(node, tuple):
        return node[1]
    else:
        return node


# =============================================================================
# Path Tree Building Classes (from path_tree_builder.py)
# =============================================================================

@dataclass
class PathTreeNode:
    """A node in the multicast path tree.

    Each node represents a position in the NoC mesh. The tree structure
    captures how multiple commodity paths share common prefixes and
    diverge at certain points.

    Attributes:
        coord: The coordinate of this node in the mesh
        commodity_ids: Set of commodity ids that pass through this node
        children: Dictionary mapping directions to child nodes
        is_destination: Whether this node is a destination for any commodity
        destination_info: Dict mapping commodity_id to destination metadata
                         (e.g., chunk_index, ksel) for commodities that end here
    """
    coord: Coord
    commodity_ids: Set[int] = field(default_factory=set)
    children: Dict[Direction, 'PathTreeNode'] = field(default_factory=dict)
    is_destination: bool = False
    destination_info: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def add_child(self, direction: Direction, child: 'PathTreeNode') -> None:
        """Add a child node in the given direction."""
        self.children[direction] = child

    def get_child(self, direction: Direction) -> Optional['PathTreeNode']:
        """Get the child node in the given direction, if exists."""
        return self.children.get(direction)

    def get_directions(self) -> List[Direction]:
        """Get all directions that have children."""
        return list(self.children.keys())

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_divergence_point(self) -> bool:
        """Check if this node is a divergence point (multiple children)."""
        return len(self.children) > 1

    def __repr__(self) -> str:
        child_dirs = [d.value for d in self.children.keys()]
        return f"PathTreeNode({self.coord}, commodities={self.commodity_ids}, children={child_dirs})"


@dataclass
class MulticastTree:
    """A complete multicast tree for a group of commodities sharing the same source and tensor.

    Attributes:
        source: The source coordinate (root of the tree)
        tensor_id: Identifier for the tensor being multicast
        root: The root PathTreeNode
        commodity_ids: All commodity ids in this multicast group
    """
    source: Coord
    tensor_id: Any
    root: PathTreeNode
    commodity_ids: List[int]

    def get_all_nodes(self) -> List[PathTreeNode]:
        """Get all nodes in the tree via BFS traversal."""
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children.values())
        return nodes

    def get_destinations(self) -> List[PathTreeNode]:
        """Get all destination nodes (leaves or nodes marked as destinations)."""
        return [node for node in self.get_all_nodes() if node.is_destination]


@dataclass
class PathTreeBuildResult:
    """Result of building path trees from routing results.

    Attributes:
        trees: Dictionary mapping (source, tensor_id) to MulticastTree
    """
    trees: Dict[Tuple[Coord, Any], MulticastTree]

    def get_tree(self, source: Coord, tensor_id: Any) -> Optional[MulticastTree]:
        """Get the multicast tree for a given source and tensor."""
        return self.trees.get((source, tensor_id))

    def get_all_trees(self) -> List[MulticastTree]:
        """Get all multicast trees."""
        return list(self.trees.values())


class PathTreeBuilder:
    """Builds path trees from routing results.

    This class takes routing results (Phase 1 output) and builds tree structures
    that represent shared multicast paths. The trees are used by the policy table
    generator to efficiently generate policy table entries with path sharing.
    """

    def __init__(self, tensor_id_extractor=None):
        """Initialize the PathTreeBuilder.

        Args:
            tensor_id_extractor: Optional function to extract tensor_id from commodity metadata.
                                If None, uses default extractor that looks for graph_node_id.
        """
        self.tensor_id_extractor = tensor_id_extractor or self._default_tensor_id_extractor

    @staticmethod
    def _default_tensor_id_extractor(metadata: Any) -> Optional[Any]:
        """Default extractor for tensor_id from commodity metadata.

        Expected metadata format: (edge, mapping_info) where edge has src_id.graph_node_id
        """
        if metadata is None:
            return None
        try:
            edge, _ = metadata
            # Handle TensorEdge case
            if hasattr(edge, 'src_id') and hasattr(edge.src_id, 'graph_node_id'):
                return edge.src_id.graph_node_id
            # Handle NodeID case (instruction edge)
            if hasattr(edge, 'name'):
                return f"instruction_{edge.name}"
            return None
        except (TypeError, ValueError, AttributeError):
            return None

    def build(self, routing_result: Any) -> PathTreeBuildResult:
        """Build path trees from routing results.

        Args:
            routing_result: The routing result from Phase 1 (any router)

        Returns:
            PathTreeBuildResult containing all multicast trees
        """
        # Step 1: Group commodities by (source, tensor_id)
        groups = self._group_commodities(routing_result)

        # Step 2: Build tree for each group
        trees: Dict[Tuple[Coord, Any], MulticastTree] = {}
        for (source, tensor_id), commodity_ids in groups.items():
            tree = self._build_tree_for_group(routing_result, source, tensor_id, commodity_ids)
            trees[(source, tensor_id)] = tree

        return PathTreeBuildResult(trees=trees)

    def _group_commodities(
        self,
        routing_result: Any
    ) -> Dict[Tuple[Coord, Any], List[int]]:
        """Group commodities by (source, tensor_id)."""
        groups: Dict[Tuple[Coord, Any], List[int]] = {}

        for cid in routing_result.get_all_commodity_ids():
            commodity = routing_result.get_commodity(cid)
            tensor_id = self.tensor_id_extractor(commodity.metadata)

            key = (commodity.source, tensor_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(cid)

        return groups

    def _build_tree_for_group(
        self,
        routing_result: Any,
        source: Coord,
        tensor_id: Any,
        commodity_ids: List[int]
    ) -> MulticastTree:
        """Build a tree for a group of commodities sharing the same source and tensor."""
        # Create root node at source
        root = PathTreeNode(coord=source, commodity_ids=set(commodity_ids))

        # Add each commodity's path to the tree
        for cid in commodity_ids:
            path = routing_result.get_path(cid)
            commodity = routing_result.get_commodity(cid)
            self._add_path_to_tree(root, path, cid, commodity)

        return MulticastTree(
            source=source,
            tensor_id=tensor_id,
            root=root,
            commodity_ids=commodity_ids
        )

    def _add_path_to_tree(
        self,
        root: PathTreeNode,
        path: List[Coord],
        commodity_id: int,
        commodity: HWCommodity
    ) -> None:
        """Add a single commodity's path to the tree.

        Traverses the path and creates/updates nodes as needed.
        """
        current_node = root

        for i, coord in enumerate(path):
            if i == 0:
                # First coord is the source (root), already handled
                continue

            # Determine direction from previous coord to current
            prev_coord = path[i - 1]
            direction = self._get_direction(prev_coord, coord)

            # Check if child exists in this direction
            child = current_node.get_child(direction)
            if child is None:
                # Create new child node
                child = PathTreeNode(coord=coord, commodity_ids={commodity_id})
                current_node.add_child(direction, child)
            else:
                # Add commodity to existing child
                child.commodity_ids.add(commodity_id)

            current_node = child

        # Mark the last node as destination
        current_node.is_destination = True

        # Store destination metadata (e.g., split_idx, ksel)
        dest_info = self._extract_destination_info(commodity)
        if dest_info:
            current_node.destination_info[commodity_id] = dest_info

    def _get_direction(self, from_coord: Coord, to_coord: Coord) -> Direction:
        """Get the direction from one coordinate to another."""
        if from_coord.col < to_coord.col:
            return Direction.EAST
        elif from_coord.col > to_coord.col:
            return Direction.WEST
        elif from_coord.row < to_coord.row:
            return Direction.SOUTH
        elif from_coord.row > to_coord.row:
            return Direction.NORTH
        return Direction.LOCAL

    def _extract_destination_info(self, commodity: HWCommodity) -> Optional[Dict[str, Any]]:
        """Extract destination-specific info from commodity metadata.

        This includes split_idx (chunk_index) and ksel for the destination node.
        """
        if commodity.metadata is None:
            return None

        try:
            edge, mapping_info = commodity.metadata
            # mapping_info is (src_node, dst_node, split_idx)
            split_idx = mapping_info[2] if len(mapping_info) > 2 else None

            return {
                'split_idx': split_idx,
                'edge': edge,
                'mapping_info': mapping_info,
            }
        except (TypeError, ValueError, IndexError):
            return None


# =============================================================================
# Policy Table Generation Classes (from policy_table_generator_v2.py)
# =============================================================================

@dataclass
class PolicyEntry:
    """A single policy table entry for a router node."""
    local: Dict[str, Any] = field(default_factory=lambda: {
        "enable": False, "chunk_index": 0, "addr": 0, "ksel": 0
    })
    north: Dict[str, Any] = field(default_factory=lambda: {"enable": False, "addr": 0})
    east: Dict[str, Any] = field(default_factory=lambda: {"enable": False, "addr": 0})
    south: Dict[str, Any] = field(default_factory=lambda: {"enable": False, "addr": 0})
    west: Dict[str, Any] = field(default_factory=lambda: {"enable": False, "addr": 0})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format used by hardware."""
        return {
            "Local": self.local.copy(),
            "North": self.north.copy(),
            "East": self.east.copy(),
            "South": self.south.copy(),
            "West": self.west.copy(),
        }

    @staticmethod
    def zero_entry() -> 'PolicyEntry':
        """Create an all-zeros entry (used as first entry in each table)."""
        return PolicyEntry()


class NodeCapacityError(Exception):
    """Raised when a node's policy table capacity is exceeded."""
    def __init__(self, node_id: NodeID, current_count: int, capacity: int):
        self.node_id = node_id
        self.current_count = current_count
        self.capacity = capacity
        super().__init__(
            f"Node {node_id} exceeded capacity: {current_count}/{capacity} entries"
        )


class EdgeInfoGenerator:
    """Generates EdgeInfo objects from policy table generation results.

    This handles the conversion of router entries to TensorEdgeInfo and
    InstEdgeInfo objects, including FIFO ID assignment.
    """

    def __init__(
        self,
        policy_tables: Dict[NodeID, List[Dict]],
        router_entries: Dict[Any, List[Tuple[NodeID, int]]],
        noc_paths: Dict,
    ):
        """Initialize EdgeInfoGenerator.

        Args:
            policy_tables: Generated policy tables from PolicyTableGenerator
            router_entries: Router entry mapping from PolicyTableGenerator
            noc_paths: Original NoCPaths dictionary
        """
        self.policy_tables = policy_tables
        self.router_entries = router_entries
        self.noc_paths = noc_paths

    def generate(self, func_name: str) -> None:
        """Generate EdgeInfo and store in ImcflowDeviceConfig.

        Args:
            func_name: Name of the function being processed
        """
        fifo_id_cnt = {node_id: 2 for node_id in NodeID}
        ID_dict = CustomIDToName()

        for edge, mapping_info in self.noc_paths.items():
            dest_node = mapping_info[1]
            router_entry_list = []

            if edge in self.router_entries:
                for node_id, entry_addr in self.router_entries[edge]:
                    entry = self.policy_tables[node_id][entry_addr]
                    router_entry_list.append(RouterEntry(node_id, entry_addr, entry))

                if isinstance(edge, TensorEdge):
                    self._handle_tensor_edge(
                        edge, router_entry_list, dest_node, fifo_id_cnt, ID_dict
                    )
                else:
                    # Instruction edge
                    edgeinfo = InstEdgeInfo(router_entry_list, None)
                    ImcflowDeviceConfig().add_inst_edge_info(func_name, edge, edgeinfo)
            else:
                # Local edge (src and dst on same node)
                edgeinfo = TensorEdgeInfo([], None, TensorEdgeInfo.LOCAL_FIFO)
                ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

    def _handle_tensor_edge(
        self,
        edge: TensorEdge,
        router_entry_list: List[RouterEntry],
        dest_node: NodeID,
        fifo_id_cnt: Dict[NodeID, int],
        ID_dict: Dict,
    ) -> None:
        """Handle TensorEdge EdgeInfo generation with FIFO ID assignment."""
        if edge.src_id.tensor_type in ["odata", "var"]:
            dst_node_name = ID_dict[getInnerNodeID(edge.dst_id.graph_node_id)]

            if dst_node_name in ["nn.imcflow_qconv", "nn.imcflow_qdwconv"]:
                edgeinfo = TensorEdgeInfo(router_entry_list, None, 0)
            else:
                edgeinfo = TensorEdgeInfo(router_entry_list, None, fifo_id_cnt[dest_node])
                fifo_id_cnt[dest_node] += 1
                if fifo_id_cnt[dest_node] >= 8:
                    raise ValueError("FIFO ID cannot be over 7!")

            ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

        elif edge.src_id.tensor_type in [
            "weight", "bias", "fused_scale", "fused_bias",
            "min", "max", "threshold", "scale", "config"
        ]:
            edgeinfo = TensorEdgeInfo(router_entry_list, None, 1)
            ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
        else:
            raise ValueError(f"Wrong tensor type: {edge.src_id.tensor_type}")


class MemoryAllocator:
    """Allocates memory for policy tables."""

    def __init__(self, policy_tables: Dict[NodeID, List[Dict]]):
        """Initialize MemoryAllocator.

        Args:
            policy_tables: Generated policy tables
        """
        self.policy_tables = policy_tables

    def allocate(self, func_name: str) -> None:
        """Allocate memory for policy tables.

        Args:
            func_name: Name of the function
        """
        for node_id, policy_table in self.policy_tables.items():
            if len(policy_table) == 0:
                continue
            mem_size = len(policy_table) * 32
            mem_block = DataBlock(f"{node_id.name}_policy", mem_size)
            inode_id = node_id.master() if node_id.is_imce() else node_id
            ImcflowDeviceConfig().MemLayout[func_name][f"{inode_id.name}_data"].allocate(
                mem_block, phase="init"
            )


class PolicyTableGenerator:
    """Generates policy tables from path tree structures.

    This is an internal class that handles the actual policy table generation
    from PathTreeBuildResult. Used by PolicyTableBuilder.

    Attributes:
        table_capacity: Maximum entries per node (default: 32)
        policy_tables: Generated policy tables per node
    """

    def __init__(self, table_capacity: int = 32):
        """Initialize PolicyTableGenerator.

        Args:
            table_capacity: Maximum number of entries per node's policy table.
        """
        self.table_capacity = table_capacity
        self.policy_tables: Dict[NodeID, List[Dict]] = {}
        self.router_entry_list: Dict[Any, List[Tuple[NodeID, int]]] = {}

    def generate(
        self,
        tree_result: PathTreeBuildResult,
        noc_paths: Dict,
    ) -> Dict[NodeID, List[Dict]]:
        """Generate policy tables from path tree result.

        Args:
            tree_result: Result from PathTreeBuilder
            noc_paths: Original NoCPaths dictionary for metadata lookup

        Returns:
            Dictionary mapping NodeID to list of policy table entries

        Raises:
            NodeCapacityError: If any node exceeds table_capacity
        """
        # Initialize policy tables with zero entry for all nodes
        self._initialize_tables()

        # Process each multicast tree
        for (source, tensor_id), tree in tree_result.trees.items():
            self._process_tree(tree)

        return self.policy_tables

    def _initialize_tables(self) -> None:
        """Initialize policy tables with zero entry for all nodes."""
        zero_entry = PolicyEntry.zero_entry().to_dict()
        self.policy_tables = {node_id: [zero_entry.copy()] for node_id in NodeID}

    def _check_capacity(self, node_id: NodeID) -> None:
        """Check if node has capacity for another entry.

        Raises:
            NodeCapacityError: If capacity would be exceeded
        """
        current_count = len(self.policy_tables[node_id])
        if current_count >= self.table_capacity:
            raise NodeCapacityError(node_id, current_count, self.table_capacity)

    def _process_tree(self, tree: MulticastTree) -> None:
        """Process a single multicast tree to generate policy entries.

        Traverses the tree and creates shared policy entries where paths overlap.
        """
        # Track entry addresses for path reconstruction
        # Maps (node, tree_node) -> entry_addr for the tree traversal
        entry_addr_map: Dict[Tuple[NodeID, int], int] = {}

        # Get destination info for ksel calculation
        dest_info_map = self._collect_destination_info(tree)

        # BFS traversal to process nodes level by level
        self._traverse_tree(tree.root, tree, dest_info_map)

    def _collect_destination_info(self, tree: MulticastTree) -> Dict[int, Dict[str, Any]]:
        """Collect destination info (ksel, chunk_index) for all commodities in tree."""
        dest_info = {}
        for node in tree.get_all_nodes():
            if node.is_destination:
                for cid, info in node.destination_info.items():
                    dest_info[cid] = info
        return dest_info

    def _traverse_tree(
        self,
        tree_node: PathTreeNode,
        tree: MulticastTree,
        dest_info_map: Dict[int, Dict[str, Any]],
        parent_node_id: Optional[NodeID] = None,
        parent_entry_addr: Optional[int] = None,
        incoming_direction: Optional[Direction] = None,
    ) -> None:
        """Recursively traverse tree and generate policy entries.

        Args:
            tree_node: Current node in the path tree
            tree: The multicast tree being processed
            dest_info_map: Destination info for ksel/chunk_index
            parent_node_id: NodeID of parent (for updating parent's entry)
            parent_entry_addr: Entry address in parent's table
            incoming_direction: Direction from parent to this node
        """
        # Convert Coord to NodeID
        node_id = NodeID.from_coord(tree_node.coord.row, tree_node.coord.col)

        # Calculate ksel from destination info
        ksel = self._get_ksel_for_node(tree_node, dest_info_map)

        # Check capacity before adding entry
        self._check_capacity(node_id)

        # Create entry for this node
        entry = PolicyEntry()
        entry.local["ksel"] = ksel

        # If this is a destination, enable local delivery
        if tree_node.is_destination:
            entry.local["enable"] = True
            # Get chunk_index from first commodity that ends here
            for cid in tree_node.commodity_ids:
                if cid in tree_node.destination_info:
                    info = tree_node.destination_info[cid]
                    if info and info.get('split_idx') is not None:
                        entry.local["chunk_index"] = info['split_idx']
                    break

        # Set up directions to children
        for direction, child_node in tree_node.children.items():
            child_node_id = NodeID.from_coord(child_node.coord.row, child_node.coord.col)
            # Address will be updated after child entry is created
            dir_name = direction.value.lower()
            getattr(entry, dir_name)["enable"] = True

        # Add entry to table
        entry_dict = entry.to_dict()
        entry_addr = len(self.policy_tables[node_id])
        self.policy_tables[node_id].append(entry_dict)

        # Update parent's entry to point to this entry
        if parent_node_id is not None and parent_entry_addr is not None and incoming_direction is not None:
            dir_name = incoming_direction.value
            self.policy_tables[parent_node_id][parent_entry_addr][dir_name]["addr"] = entry_addr

        # Store router entry info for EdgeInfo generation
        self._record_router_entry(tree_node, tree, node_id, entry_addr)

        # Recursively process children
        for direction, child_node in tree_node.children.items():
            self._traverse_tree(
                child_node,
                tree,
                dest_info_map,
                parent_node_id=node_id,
                parent_entry_addr=entry_addr,
                incoming_direction=direction,
            )

    def _get_ksel_for_node(
        self,
        tree_node: PathTreeNode,
        dest_info_map: Dict[int, Dict[str, Any]]
    ) -> int:
        """Get ksel value for a node based on destination info."""
        # Use the first commodity's destination info for ksel
        for cid in tree_node.commodity_ids:
            if cid in dest_info_map:
                info = dest_info_map[cid]
                if info and 'edge' in info:
                    edge = info['edge']
                    mapping_info = info.get('mapping_info')
                    if mapping_info and mapping_info[2] is not None:
                        dst_graph_node = CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)]
                        if isinstance(dst_graph_node, relay.expr.Call) and isinstance(dst_graph_node.op, tvm.ir.Op) and \
                           "conv" in dst_graph_node.op.name:
                          kernel_size = dst_graph_node.attrs['kernel_size'][0].value
                          if kernel_size in [1, 2, 3, 5, 7]:
                              return kernel_size
        return 0

    def _record_router_entry(
        self,
        tree_node: PathTreeNode,
        tree: MulticastTree,
        node_id: NodeID,
        entry_addr: int
    ) -> None:
        """Record router entry for later EdgeInfo generation."""
        # For each commodity passing through this node, record the entry
        for cid in tree_node.commodity_ids:
            # Get the original edge from destination info
            for dest_node in tree.get_destinations():
                if cid in dest_node.destination_info:
                    info = dest_node.destination_info[cid]
                    if info and 'edge' in info:
                        edge = info['edge']
                        if edge not in self.router_entry_list:
                            self.router_entry_list[edge] = []
                        self.router_entry_list[edge].append((node_id, entry_addr))
                    break
        
        # dump
        for edge, entries in self.router_entry_list.items():
            entry_str = ", ".join(f"({nid.name}, {addr})" for nid, addr in entries)
            debug_print(f"[RouterEntry] Edge: {edge}, Entries: {entry_str}")

    def get_router_entries(self) -> Dict[Any, List[Tuple[NodeID, int]]]:
        """Get recorded router entries for EdgeInfo generation."""
        return self.router_entry_list


# =============================================================================
# Top-level PolicyTableBuilder Class
# =============================================================================

class PolicyTableBuilder:
    """Unified Policy Table Builder.

    This is the top-level class that orchestrates the entire pipeline:
    - Phase 2: Tree building (PathTreeBuilder)
    - Phase 3a: Policy table generation (PolicyTableGenerator)
    - Phase 3b: EdgeInfo generation (EdgeInfoGenerator)
    - Phase 3c: Memory allocation (MemoryAllocator)

    Usage:
        builder = PolicyTableBuilder(table_capacity=32)
        policy_tables = builder.build(routing_result, noc_paths, func_name)
    """

    def __init__(self, table_capacity: int = 32):
        """Initialize PolicyTableBuilder.

        Args:
            table_capacity: Maximum number of entries per node's policy table.
        """
        self.table_capacity = table_capacity

    def build(
        self,
        routing_result: Any,
        noc_paths: Dict,
        func_name: str,
    ) -> Dict[NodeID, List[Dict]]:
        """Build policy tables from routing results.

        This method performs the complete pipeline:
        1. Build multicast trees from routing result
        2. Generate policy tables from trees
        3. Generate EdgeInfo
        4. Allocate policy table memory
        5. Store in DevConfig

        Args:
            routing_result: Routing result from Phase 1 (any router)
            noc_paths: NoCPaths dictionary
            func_name: Name of the function being processed

        Returns:
            Dictionary mapping NodeID to list of policy table entries
        """
        # Phase 2: Build multicast trees from routing result
        tree_builder = PathTreeBuilder()
        tree_result = tree_builder.build(routing_result)

        # Phase 3a: Generate policy tables
        generator = PolicyTableGenerator(table_capacity=self.table_capacity)
        policy_tables = generator.generate(tree_result, noc_paths)

        # Phase 3b: Generate EdgeInfo
        edge_info_gen = EdgeInfoGenerator(
            policy_tables,
            generator.get_router_entries(),
            noc_paths,
        )
        edge_info_gen.generate(func_name)

        # Phase 3c: Allocate memory
        allocator = MemoryAllocator(policy_tables)
        allocator.allocate(func_name)

        # Store in device config
        ImcflowDeviceConfig().PolicyTableDict[func_name] = policy_tables

        return policy_tables


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

def build_path_trees(
    routing_result: Any,
    tensor_id_extractor=None
) -> PathTreeBuildResult:
    """Convenience function to build path trees from routing results.

    Args:
        routing_result: The routing result from Phase 1
        tensor_id_extractor: Optional function to extract tensor_id from metadata

    Returns:
        PathTreeBuildResult containing all multicast trees
    """
    builder = PathTreeBuilder(tensor_id_extractor=tensor_id_extractor)
    return builder.build(routing_result)
