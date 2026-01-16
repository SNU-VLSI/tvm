"""
Policy Table Generator for NoC Routing (Phase 3)

This module generates policy tables from path tree structures.
It takes the output of Phase 2 (PathTreeBuilder) and creates
hardware-compatible policy table entries.

Key responsibilities:
- Traverse multicast trees to generate policy entries
- Handle node capacity constraints (max 32 entries per node)
- Generate RouterEntry objects for EdgeInfo
- Manage FIFO ID assignment
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

from .mcf_router import Coord, Direction, Edge
from .path_tree_builder import (
    PathTreeNode,
    MulticastTree,
    PathTreeBuildResult,
)

logger = logging.getLogger(__name__)


# Helper function (moved from transform.py)
def getInnerNodeID(node):
    """Extract inner node ID from potentially nested tuple."""
    if isinstance(node, tuple):
        return node[1]
    else:
        return node


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


class PolicyTableGenerator:
    """Generates policy tables from path tree structures.

    This is Phase 3 of the routing pipeline:
    - Phase 1 (Router): Generates paths for commodities
    - Phase 2 (TreeBuilder): Builds multicast trees
    - Phase 3 (This): Generates policy table entries

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
            tree_result: Result from Phase 2 (PathTreeBuilder)
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
                        try:
                            dst_graph_node = CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)]
                            kernel_size = dst_graph_node.attrs['kernel_size'][0].value
                            if kernel_size in [1, 2, 3, 5, 7]:
                                return kernel_size
                        except (KeyError, AttributeError, TypeError):
                            pass
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

    def get_router_entries(self) -> Dict[Any, List[Tuple[NodeID, int]]]:
        """Get recorded router entries for EdgeInfo generation."""
        return self.router_entry_list


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

            if dst_node_name == "nn.imcflow_qconv":
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


def generate_policy_tables(
    tree_result: PathTreeBuildResult,
    noc_paths: Dict,
    func_name: str,
    table_capacity: int = 32,
) -> Dict[NodeID, List[Dict]]:
    """Convenience function to generate policy tables and update device config.

    This is the main entry point for Phase 3.

    Args:
        tree_result: Result from Phase 2 (PathTreeBuilder)
        noc_paths: Original NoCPaths dictionary
        func_name: Name of the function being processed
        table_capacity: Maximum entries per node

    Returns:
        Generated policy tables

    Raises:
        NodeCapacityError: If any node exceeds capacity
    """
    # Phase 3a: Generate policy tables
    generator = PolicyTableGenerator(table_capacity=table_capacity)
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
