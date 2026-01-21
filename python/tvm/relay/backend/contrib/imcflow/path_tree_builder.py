"""
Path Tree Builder for NoC Routing (Phase 2)

This module builds tree structures from routing results to enable efficient
policy table generation with multicast path sharing.

Given routing results where multiple commodities may share the same source
and tensor (multicast scenario), this module:
1. Groups commodities by (source, tensor_id)
2. Builds a tree structure representing shared paths
3. Identifies divergence points where paths split

The output tree structure is consumed by Phase 3 (PolicyTableGenerator).
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .mcf_router import (
    BaseRoutingResult,
    RoutingResult,
    Commodity,
    Coord,
    Edge,
    Direction,
)


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
        single_paths: List of commodity ids that are not part of any multicast group
                     (unique source+tensor combinations)
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
    that represent shared multicast paths. The trees are used by Phase 3
    (PolicyTableGenerator) to efficiently generate policy table entries with
    path sharing.
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

    def build(self, routing_result: BaseRoutingResult) -> PathTreeBuildResult:
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
        routing_result: BaseRoutingResult
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
        routing_result: BaseRoutingResult,
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
        commodity: Commodity
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

    def _extract_destination_info(self, commodity: Commodity) -> Optional[Dict[str, Any]]:
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


def build_path_trees(
    routing_result: BaseRoutingResult,
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
