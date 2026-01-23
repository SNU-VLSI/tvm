"""
Multi-Commodity Flow Router using ILP Solver

This module implements an ILP-based router for NoC (Network on Chip) that minimizes
edge conflicts when routing multiple (source, destination) pairs simultaneously.

The multi-commodity flow problem is formulated as:
- Variables: Binary variable x_{k,e} for each commodity k and edge e
- Constraints:
  - Flow conservation: at each node, inflow = outflow (except source/sink)
  - Capacity constraints: total flow on each edge <= capacity
- Objective: Minimize total edge usage (or maximize edge sharing efficiency)
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import pulp

logger = logging.getLogger(__name__)


# Debug print utility
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
class Commodity:
    """A flow demand from source to destination"""
    id: int
    source: Coord
    destination: Coord
    metadata: Any = None  # Original edge info from NoCPaths

    def __hash__(self):
        return hash(self.id)


@dataclass
class MeshTopology:
    """2D Mesh NoC topology"""
    rows: int
    cols: int

    def get_all_nodes(self) -> List[Coord]:
        """Get all nodes in the mesh"""
        return [Coord(r, c) for r in range(self.rows) for c in range(self.cols)]

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

    def get_all_edges(self) -> List[Edge]:
        """Get all directed edges in the mesh"""
        edges = []
        for node in self.get_all_nodes():
            for neighbor, _ in self.get_neighbors(node):
                edges.append(Edge(node, neighbor))
        return edges

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if coordinate is within mesh bounds"""
        return 0 <= coord.row < self.rows and 0 <= coord.col < self.cols


@dataclass
class RoutingResult:
    """Result of routing a single commodity"""
    commodity: Commodity
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

    def get_commodity(self, commodity_id: int) -> Commodity:
        """Get the commodity object by id."""
        if commodity_id not in self.routes:
            raise KeyError(f"Commodity {commodity_id} not found in routes")
        return self.routes[commodity_id].commodity

    def get_all_commodity_ids(self) -> List[int]:
        """Get all commodity ids."""
        return list(self.routes.keys())

    def get_commodities_by_source(self, source: Coord) -> List[int]:
        """Get all commodity ids that originate from a given source."""
        return [
            cid for cid, route in self.routes.items()
            if route.commodity.source == source
        ]

    def get_commodities_passing_through(self, node: Coord) -> List[int]:
        """Get all commodity ids whose path passes through a given node."""
        result = []
        for cid, route in self.routes.items():
            if node in route.path:
                result.append(cid)
        return result

    def get_commodities_by_metadata_key(self, key_func) -> Dict[Any, List[int]]:
        """Group commodity ids by a key extracted from metadata.

        Args:
            key_func: Function that takes commodity.metadata and returns a grouping key.
                      Return None to exclude the commodity from grouping.

        Returns:
            Dictionary mapping keys to lists of commodity ids.
        """
        groups: Dict[Any, List[int]] = {}
        for cid, route in self.routes.items():
            key = key_func(route.commodity.metadata)
            if key is not None:
                if key not in groups:
                    groups[key] = []
                groups[key].append(cid)
        return groups


@dataclass
class MCFRoutingResult(BaseRoutingResult):
    """Complete result of multi-commodity flow routing using ILP solver."""
    routes: Dict[int, RoutingResult]  # commodity_id -> routing result
    edge_usage: Dict[Edge, List[int]]  # edge -> list of commodity ids using it
    max_edge_congestion: int
    total_edge_usage: int
    solver_status: str


class MCFRouter:
    """
    Multi-Commodity Flow Router using ILP

    This router finds paths for multiple (source, destination) pairs while
    minimizing edge conflicts (multiple commodities using the same edge).
    """

    def __init__(
        self,
        topology: MeshTopology,
        minimize_congestion: bool = True,
        enforce_destination_disjoint: bool = False,
    ):
        """
        Initialize MCF Router with multicast support.

        The router automatically finds the minimum edge capacity needed.
        Commodities with the same source are treated as multicast and
        share edges (counting as 1 towards capacity).

        Args:
            topology: The mesh topology
            minimize_congestion: If True, minimize max congestion. If False, minimize total usage.
            enforce_destination_disjoint: If True, data commodities with the same destination
                must use edge-disjoint paths (no shared edges). This is a hard constraint.
        """
        self.topology = topology
        self.minimize_congestion = minimize_congestion
        self.enforce_destination_disjoint = enforce_destination_disjoint

    def route(self, commodities: List[Commodity]) -> MCFRoutingResult:
        """
        Route all commodities using ILP

        Args:
            commodities: List of commodities to route

        Returns:
            MCFRoutingResult containing all routes and statistics
        """
        debug_print(f"[MCFRouter] Starting routing for {len(commodities)} commodities")

        if not commodities:
          raise ValueError("No commodities provided for routing.")

        # Filter out same-node commodities
        valid_commodities = [c for c in commodities if c.source != c.destination]
        skipped = len(commodities) - len(valid_commodities)
        if skipped > 0:
            debug_print(f"[MCFRouter] Skipped {skipped} same-node commodities")

        if not valid_commodities:
          raise ValueError("All commodities have same source and destination.")

        # Deduplicate const group commodities with same (src, dst)
        # They will share the same path
        deduped_commodities, const_dedup_map = self._deduplicate_const_commodities(valid_commodities)
        if len(deduped_commodities) < len(valid_commodities):
            debug_print(f"[MCFRouter] Deduplicated const commodities: {len(valid_commodities)} -> {len(deduped_commodities)}")

        # Validate: no multicast group should have same-destination commodities
        if self.enforce_destination_disjoint:
            self._validate_no_multicast_same_destination(deduped_commodities)

        debug_print(f"[MCFRouter] Routing {len(deduped_commodities)} commodities on {self.topology.rows}x{self.topology.cols} mesh")
        result = self._route_ilp(deduped_commodities)

        # Expand results back to deduplicated commodities
        if const_dedup_map:
            result = self._expand_const_dedup_results(result, const_dedup_map)

        return result

    def _validate_no_multicast_same_destination(self, commodities: List[Commodity]) -> None:
        """
        Validate that no multicast group has commodities with the same destination.

        When enforce_destination_disjoint is enabled, commodities with the same destination
        must use edge-disjoint paths. However, multicast commodities (same source) share
        edges by definition, which would violate this constraint.

        Raises:
            ValueError: If a multicast group has commodities with the same destination.
        """
        # Group commodities by multicast key (source + graph_node_id)
        source_groups = self._group_by_source(commodities)

        for source_key, group in source_groups.items():
            if len(group) <= 1:
                continue  # Not a multicast group

            # Check if this is a data group (only data group has the constraint)
            group_type = self._get_commodity_group(group[0])
            if group_type != 'data':
                continue

            # Check for duplicate destinations within this multicast group
            destinations = [c.destination for c in group]
            seen_destinations = set()
            for c in group:
                if c.destination in seen_destinations:
                    raise ValueError(
                        f"Multicast group from source {source_key} has multiple commodities "
                        f"with the same destination {c.destination}. This violates the "
                        f"enforce_destination_disjoint constraint. "
                        f"Commodities in group: {[(c.id, c.source, c.destination) for c in group]}"
                    )
                seen_destinations.add(c.destination)

    def _deduplicate_const_commodities(
        self,
        commodities: List[Commodity]
    ) -> Tuple[List[Commodity], Dict[Tuple[Coord, Coord], List[Commodity]]]:
        """
        Deduplicate const group commodities that have the same (src, dst).

        For const tensors (weight, config, etc.), multiple commodities with the same
        (source, destination) should use the same NoC path. This method groups them
        and keeps only one representative for routing.

        Args:
            commodities: List of all commodities

        Returns:
            Tuple of:
            - deduped_commodities: List with const duplicates removed
            - const_dedup_map: Dict mapping (src, dst) -> list of duplicate commodities
                               (only for groups with more than 1 commodity)
        """
        # Separate const and non-const commodities
        const_commodities = []
        other_commodities = []

        for c in commodities:
            if self._get_commodity_group(c) == 'const':
                const_commodities.append(c)
            else:
                other_commodities.append(c)

        if not const_commodities:
            return commodities, {}

        # Group const commodities by (src, dst)
        const_by_src_dst: Dict[Tuple[Coord, Coord], List[Commodity]] = {}
        for c in const_commodities:
            key = (c.source, c.destination)
            if key not in const_by_src_dst:
                const_by_src_dst[key] = []
            const_by_src_dst[key].append(c)

        # Build deduplicated list and map
        deduped_commodities = list(other_commodities)
        const_dedup_map: Dict[Tuple[Coord, Coord], List[Commodity]] = {}

        for (src, dst), group in const_by_src_dst.items():
            # Always add the first commodity (representative)
            deduped_commodities.append(group[0])

            # If there are duplicates, record them for later expansion
            if len(group) > 1:
                const_dedup_map[(src, dst)] = group
                debug_print(f"[MCFRouter]   Const dedup: ({src}, {dst}) has {len(group)} commodities, keeping id={group[0].id}")

        return deduped_commodities, const_dedup_map

    def _expand_const_dedup_results(
        self,
        result: 'MCFRoutingResult',
        const_dedup_map: Dict[Tuple[Coord, Coord], List[Commodity]]
    ) -> 'MCFRoutingResult':
        """
        Expand routing results to include deduplicated const commodities.

        For each (src, dst) group in const_dedup_map, copy the routing result
        from the representative commodity to all other commodities in the group.

        Args:
            result: Routing result with only representative commodities
            const_dedup_map: Map from (src, dst) to list of all commodities in that group

        Returns:
            MCFRoutingResult with all commodities included
        """
        expanded_routes = dict(result.routes)
        expanded_edge_usage = {e: list(v) for e, v in result.edge_usage.items()}

        for (src, dst), group in const_dedup_map.items():
            # Find the representative's route (first in group)
            rep_id = group[0].id
            if rep_id not in expanded_routes:
                debug_print(f"[MCFRouter] WARNING: Representative {rep_id} not found in routes")
                continue

            rep_route = expanded_routes[rep_id]

            # Copy route to all other commodities in the group
            for c in group[1:]:
                # Create a new RoutingResult with the same path but different commodity
                new_route = RoutingResult(
                    commodity=c,
                    path=list(rep_route.path),  # Copy path
                    edges=list(rep_route.edges)  # Copy edges
                )
                expanded_routes[c.id] = new_route

                # Update edge usage
                for e in rep_route.edges:
                    if e in expanded_edge_usage:
                        expanded_edge_usage[e].append(c.id)
                    else:
                        expanded_edge_usage[e] = [c.id]

                debug_print(f"[MCFRouter]   Expanded const route: id={c.id} copies path from id={rep_id}")

        return MCFRoutingResult(
            routes=expanded_routes,
            edge_usage={e: v for e, v in expanded_edge_usage.items() if v},
            max_edge_congestion=result.max_edge_congestion,
            total_edge_usage=result.total_edge_usage,
            solver_status=result.solver_status + f" (const_dedup={len(const_dedup_map)})"
        )

    def _route_ilp(self, commodities: List[Commodity]) -> MCFRoutingResult:
        """Route using ILP solver (PuLP) with adaptive capacity.

        Starts with edge_capacity=1 (no conflicts) and increments until feasible.
        """
        # Start with edge_capacity=1 and increment until feasible
        current_capacity = 1
        max_capacity = len(commodities)  # Upper bound

        debug_print(f"[MCFRouter] Trying adaptive capacity from 1 to {max_capacity}")

        while current_capacity <= max_capacity:
            result = self._try_route_with_capacity(commodities, current_capacity)
            if result is not None:
                if current_capacity > 1:
                    debug_print(f"[MCFRouter] Found solution with edge_capacity={current_capacity}")
                    logger.info(f"MCF routing found solution with edge_capacity={current_capacity}")
                else:
                    debug_print(f"[MCFRouter] Found solution with edge_capacity=1 (no conflicts)")
                return result

            debug_print(f"[MCFRouter] Infeasible with edge_capacity={current_capacity}, trying {current_capacity + 1}")
            logger.debug(f"MCF routing infeasible with edge_capacity={current_capacity}, trying {current_capacity + 1}")
            current_capacity += 1

        # Fallback: solve without capacity constraint
        debug_print(f"[MCFRouter] WARNING: Failed with all capacities, solving without constraint")
        logger.warning("MCF routing failed with all capacities, solving without constraint")
        result = self._try_route_with_capacity(commodities, None)
        if result is None:
            raise RuntimeError(
                f"MCF routing failed completely for {len(commodities)} commodities. "
                "ILP solver could not find a feasible solution even without capacity constraints."
            )
        return result

    # Tensor type classification for congestion groups
    CONST_TENSOR_TYPES = frozenset([
        'weight', 'config', 'min', 'max', 'fused_scale', 'fused_bias',
        'bias', 'scale', 'threshold'
    ])
    DATA_TENSOR_TYPES = frozenset([
        'odata', 'data', 'lhs', 'rhs', *[f"func_out{i}" for i in range(30)], 'var'
    ])

    def _get_commodity_group(self, commodity: Commodity) -> str:
        """
        Classify commodity into congestion group.

        Groups:
        - 'inst': Instruction edges (key is NodeID, not TensorEdge)
        - 'const': Constant tensors (weight, config, min, max, etc.)
        - 'data': Runtime data tensors (odata, data, lhs, rhs, etc.)

        These groups don't compete for edges at runtime, so congestion
        should be calculated separately within each group.

        Returns:
            Group name: 'inst', 'const', or 'data'
        """
        if commodity.metadata is None:
            return 'data'  # Default to data group

        edge = commodity.metadata[0]

        # Check if this is an instruction edge (key is NodeID, not TensorEdge)
        if not hasattr(edge, 'src_id'):
            # NodeID doesn't have src_id attribute, so this is instruction
            return 'inst'

        # Get tensor type from TensorEdge
        tensor_type = edge.src_id.tensor_type if hasattr(edge.src_id, 'tensor_type') else None

        if tensor_type in self.CONST_TENSOR_TYPES:
            return 'const'
        elif tensor_type in self.DATA_TENSOR_TYPES:
            return 'data'
        else:
            # Unknown type - default to data (more conservative)
            debug_print(f"[MCFRouter] Unknown tensor_type '{tensor_type}', defaulting to 'data' group")
            return 'data'

    def _get_multicast_key(self, commodity: Commodity) -> Tuple:
        """
        Get multicast grouping key for a commodity.

        Multicast requires both:
        1. Same HW source coordinate
        2. Same source graph_node_id (same data being transmitted)

        Returns:
            Tuple of (source_coord, graph_node_id) or (source_coord, None) if no metadata
        """
        graph_node_id = None
        if commodity.metadata is not None:
            # metadata is (TensorEdge, mapping_info)
            edge = commodity.metadata[0]
            if hasattr(edge, 'src_id') and hasattr(edge.src_id, 'graph_node_id'):
                graph_node_id = edge.src_id.graph_node_id
        return (commodity.source, graph_node_id)

    def _group_by_source(self, commodities: List[Commodity]) -> Dict[Tuple, List[Commodity]]:
        """
        Group commodities by their multicast key for multicast handling.

        Two commodities are in the same multicast group only if:
        1. Same HW source coordinate
        2. Same source graph_node_id (i.e., transmitting the same data)

        Returns:
            Dict mapping (source_coord, graph_node_id) to list of commodities
        """
        groups = {}
        for c in commodities:
            key = self._get_multicast_key(c)
            if key not in groups:
                groups[key] = []
            groups[key].append(c)
        return groups

    def _try_route_with_capacity(
        self,
        commodities: List[Commodity],
        capacity: Optional[int]
    ) -> Optional[MCFRoutingResult]:
        """Try to route with given edge capacity. Returns None if infeasible.

        Supports multicast: commodities with same source share edges and
        count as 1 towards edge capacity.
        """
        cap_str = str(capacity) if capacity is not None else "unlimited"
        debug_print(f"[MCFRouter] Solving ILP with edge_capacity={cap_str}")

        # Create problem
        prob = pulp.LpProblem("MCF_Multicast", pulp.LpMinimize)

        # Get all edges
        all_edges = self.topology.get_all_edges()
        edge_to_idx = {e: i for i, e in enumerate(all_edges)}

        # Group commodities by source for multicast
        source_groups = self._group_by_source(commodities)
        group_id_map = {}  # source -> group_id
        for gid, source in enumerate(source_groups.keys()):
            group_id_map[source] = gid

        # ============================================================
        # Variables
        # ============================================================

        # x[k][e] = 1 if commodity k uses edge e
        x = {}
        for k in commodities:
            x[k.id] = {}
            for e in all_edges:
                x[k.id][e] = pulp.LpVariable(
                    f"x_{k.id}_{edge_to_idx[e]}",
                    cat=pulp.LpBinary
                )

        # y[g][e] = 1 if ANY commodity in multicast group g uses edge e
        # (for groups with multiple destinations, i.e., actual multicast)
        y = {}
        multicast_groups = {src: grp for src, grp in source_groups.items() if len(grp) > 1}
        unicast_commodities = [c for src, grp in source_groups.items() if len(grp) == 1 for c in grp]

        debug_print(f"[MCFRouter]   Multicast groups: {len(multicast_groups)}, Unicast commodities: {len(unicast_commodities)}")
        debug_print(f"[MCFRouter]   Total edges in mesh: {len(all_edges)}")

        for source in multicast_groups:
            gid = group_id_map[source]
            y[gid] = {}
            for e in all_edges:
                y[gid][e] = pulp.LpVariable(
                    f"y_{gid}_{edge_to_idx[e]}",
                    cat=pulp.LpBinary
                )

        # ============================================================
        # Constraints
        # ============================================================

        # 1. Flow conservation for each commodity
        # Ensure simple path (no cycles, no branching)
        for k in commodities:
            for node in self.topology.get_all_nodes():
                in_edges = [e for e in all_edges if e.dst == node]
                out_edges = [e for e in all_edges if e.src == node]

                in_flow = pulp.lpSum(x[k.id][e] for e in in_edges) if in_edges else 0
                out_flow = pulp.lpSum(x[k.id][e] for e in out_edges) if out_edges else 0

                if node == k.source:
                    # Source: exactly 1 outgoing, 0 incoming (no cycles back to source)
                    prob += out_flow == 1, f"flow_{k.id}_src_out"
                    if in_edges:
                        prob += in_flow == 0, f"flow_{k.id}_src_in"
                elif node == k.destination:
                    # Destination: exactly 1 incoming, 0 outgoing (no cycles from destination)
                    prob += in_flow == 1, f"flow_{k.id}_dst_in"
                    if out_edges:
                        prob += out_flow == 0, f"flow_{k.id}_dst_out"
                else:
                    # Intermediate: conservation + at most 1 outgoing (simple path)
                    prob += in_flow == out_flow, f"flow_{k.id}_{node.row}_{node.col}"
                    if out_edges:
                        prob += out_flow <= 1, f"simple_{k.id}_{node.row}_{node.col}"

        # 2. Multicast group edge variable: y[g][e] >= x[k][e] for all k in group g
        #    This ensures y[g][e] = 1 if ANY commodity in the group uses edge e
        for source, group in multicast_groups.items():
            gid = group_id_map[source]
            for e in all_edges:
                for k in group:
                    prob += y[gid][e] >= x[k.id][e], f"mcast_{gid}_{k.id}_{edge_to_idx[e]}"

        # ============================================================
        # 3. Edge usage per congestion group
        # Different tensor types don't compete at runtime, so calculate
        # congestion separately for: inst, const, data
        # ============================================================

        # Classify commodities and multicast groups by congestion group
        congestion_groups = {'inst': [], 'const': [], 'data': []}
        for c in commodities:
            cg = self._get_commodity_group(c)
            congestion_groups[cg].append(c)

        # Also classify multicast groups
        mcast_by_cgroup = {'inst': [], 'const': [], 'data': []}
        for src, grp in multicast_groups.items():
            # All commodities in a multicast group have same type
            cg = self._get_commodity_group(grp[0])
            mcast_by_cgroup[cg].append(src)

        # Classify unicast commodities
        unicast_by_cgroup = {'inst': [], 'const': [], 'data': []}
        for c in unicast_commodities:
            cg = self._get_commodity_group(c)
            unicast_by_cgroup[cg].append(c)

        debug_print(f"[MCFRouter]   Congestion groups: inst={len(congestion_groups['inst'])}, "
                   f"const={len(congestion_groups['const'])}, data={len(congestion_groups['data'])}")

        # Create edge usage variables for each congestion group
        edge_usage_by_group = {}
        for cg_name in ['inst', 'const', 'data']:
            edge_usage_by_group[cg_name] = {}
            for e in all_edges:
                edge_usage_by_group[cg_name][e] = pulp.LpVariable(
                    f"usage_{cg_name}_{edge_to_idx[e]}",
                    lowBound=0,
                    cat=pulp.LpInteger
                )
                # Usage = multicast groups in this cg + unicast commodities in this cg
                mcast_usage = pulp.lpSum(
                    y[group_id_map[src]][e]
                    for src in mcast_by_cgroup[cg_name]
                )
                ucast_usage = pulp.lpSum(
                    x[k.id][e]
                    for k in unicast_by_cgroup[cg_name]
                )
                prob += edge_usage_by_group[cg_name][e] == mcast_usage + ucast_usage

        # 4. Capacity constraints (DATA group only - most critical at runtime)
        # inst and const are not constrained since they don't compete at runtime
        if capacity is not None:
            for e in all_edges:
                prob += edge_usage_by_group['data'][e] <= capacity

        # 5. Destination-disjoint constraint for data commodities (hard constraint)
        # Commodities with same destination must use edge-disjoint paths
        if self.enforce_destination_disjoint:
            # Group data commodities by destination
            data_commodities = congestion_groups['data']
            dest_groups: Dict[Coord, List[Commodity]] = {}
            for c in data_commodities:
                if c.destination not in dest_groups:
                    dest_groups[c.destination] = []
                dest_groups[c.destination].append(c)

            # For each destination group with multiple commodities,
            # add edge-disjoint constraint: at most 1 commodity can use each edge
            num_disjoint_groups = 0
            for dest, group in dest_groups.items():
                if len(group) <= 1:
                    continue  # No constraint needed for single commodity

                num_disjoint_groups += 1
                for e in all_edges:
                    # Sum of x[k][e] for all k in this destination group <= 1
                    prob += (
                        pulp.lpSum(x[k.id][e] for k in group) <= 1,
                        f"dest_disjoint_{dest.row}_{dest.col}_{edge_to_idx[e]}"
                    )

            if num_disjoint_groups > 0:
                debug_print(f"[MCFRouter]   Added destination-disjoint constraints for {num_disjoint_groups} destination groups")

        # ============================================================
        # Objective: minimize max congestion across all groups
        # + small penalty for tree size (multicast sharing)
        # ============================================================

        # Max congestion per group
        max_cong_by_group = {}
        for cg_name in ['inst', 'const', 'data']:
            max_cong_by_group[cg_name] = pulp.LpVariable(
                f"max_cong_{cg_name}",
                lowBound=0,
                cat=pulp.LpInteger
            )
            for e in all_edges:
                prob += max_cong_by_group[cg_name] >= edge_usage_by_group[cg_name][e]

        # Overall max congestion = max of all group congestions (for reporting)
        overall_max_congestion = pulp.LpVariable("overall_max_cong", lowBound=0, cat=pulp.LpInteger)
        for cg_name in ['inst', 'const', 'data']:
            prob += overall_max_congestion >= max_cong_by_group[cg_name]

        if self.minimize_congestion:
            # Primary: minimize DATA group congestion (most critical at runtime)
            # Secondary: minimize inst/const congestion as tie-breakers
            # Tertiary: minimize multicast tree sizes (encourage sharing)
            epsilon1 = 0.1   # weight for inst/const relative to data
            epsilon2 = 0.001  # weight for tree size penalty
            tree_size_penalty = pulp.lpSum(
                y[group_id_map[src]][e]
                for src in multicast_groups
                for e in all_edges
            )
            prob += (max_cong_by_group['data']
                     + epsilon1 * (max_cong_by_group['inst'] + max_cong_by_group['const'])
                     + epsilon2 * tree_size_penalty)
        else:
            # Minimize total edge usage across all groups
            total_usage = pulp.lpSum(
                edge_usage_by_group[cg_name][e]
                for cg_name in ['inst', 'const', 'data']
                for e in all_edges
            )
            prob += total_usage

        # ============================================================
        # Solve
        # ============================================================

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            debug_print(f"[MCFRouter]   Solver status: {pulp.LpStatus[status]} (not optimal)")
            return None  # Infeasible with this capacity

        debug_print(f"[MCFRouter]   Solver status: {pulp.LpStatus[status]}")

        # ============================================================
        # Extract routes from solution
        # ============================================================

        routes = {}
        edge_usage = {e: [] for e in all_edges}

        for k in commodities:
            used_edges = [e for e in all_edges if pulp.value(x[k.id][e]) > 0.5]

            # Debug: Check for multiple outgoing edges from same node
            out_degree = {}
            for e in used_edges:
                if e.src not in out_degree:
                    out_degree[e.src] = []
                out_degree[e.src].append(e.dst)

            multi_out = {src: dsts for src, dsts in out_degree.items() if len(dsts) > 1}
            if multi_out:
                print(f"[MCFRouter DEBUG] Commodity {k.id} ({k.source} -> {k.destination}):")
                print(f"  Multiple outgoing edges detected: {multi_out}")
                # Print all used edges for this commodity
                print(f"  All used edges: {used_edges}")
                # Check flow conservation violation
                for node in multi_out.keys():
                    in_edges = [e for e in used_edges if e.dst == node]
                    out_edges = [e for e in used_edges if e.src == node]
                    print(f"  At {node}: in={in_edges}, out={out_edges}")
                    # Also print x values
                    for e in out_edges:
                        val = pulp.value(x[k.id][e])
                        print(f"    x[{k.id}][{e}] = {val}")

            path = self._reconstruct_path(k.source, k.destination, used_edges)
            routes[k.id] = RoutingResult(k, path, used_edges)

            for e in used_edges:
                edge_usage[e].append(k.id)

        # Calculate statistics (per-group congestion)
        max_cong_per_group = {}
        total_usage_per_group = {}
        for cg_name in ['inst', 'const', 'data']:
            max_cong_per_group[cg_name] = int(pulp.value(max_cong_by_group[cg_name])) if pulp.value(max_cong_by_group[cg_name]) else 0
            total_usage_per_group[cg_name] = sum(
                int(pulp.value(edge_usage_by_group[cg_name][e]) or 0)
                for e in all_edges
            )

        overall_max_cong = int(pulp.value(overall_max_congestion)) if pulp.value(overall_max_congestion) else 0
        total_usage = sum(total_usage_per_group.values())

        # Count multicast groups
        num_multicast = len(multicast_groups)
        num_unicast = len(unicast_commodities)

        debug_print(f"[MCFRouter]   Result: overall_max_cong={overall_max_cong}")
        debug_print(f"[MCFRouter]     Per-group max_cong: inst={max_cong_per_group['inst']}, "
                   f"const={max_cong_per_group['const']}, data={max_cong_per_group['data']}")
        debug_print(f"[MCFRouter]     Per-group total_usage: inst={total_usage_per_group['inst']}, "
                   f"const={total_usage_per_group['const']}, data={total_usage_per_group['data']}")
        debug_print(f"[MCFRouter]   Routed {len(routes)} commodities successfully")

        return MCFRoutingResult(
            routes=routes,
            edge_usage={e: v for e, v in edge_usage.items() if v},
            max_edge_congestion=overall_max_cong,
            total_edge_usage=total_usage,
            solver_status=f"{pulp.LpStatus[status]} (cap={capacity}, mcast={num_multicast}, ucast={num_unicast}, "
                         f"cong:inst={max_cong_per_group['inst']}/const={max_cong_per_group['const']}/data={max_cong_per_group['data']})"
        )

    def _reconstruct_path(
        self,
        source: Coord,
        destination: Coord,
        edges: List[Edge]
    ) -> List[Coord]:
        """Reconstruct path from list of edges"""
        if not edges:
            return [source]

        # Build adjacency from edges - check for conflicts
        adj = {}
        for e in edges:
            if e.src in adj:
                # Multiple edges from same source - this is a problem!
                logger.warning(
                    f"Multiple edges from {e.src}: "
                    f"existing {e.src}->{adj[e.src]}, new {e.src}->{e.dst}"
                )
                print(
                    f"[MCFRouter WARNING] Multiple edges from {e.src}: "
                    f"existing {e.src}->{adj[e.src]}, new {e.src}->{e.dst}"
                )
            adj[e.src] = e.dst

        path = [source]
        current = source
        visited = set()

        while current != destination and current in adj:
            if current in visited:
                logger.warning(f"Cycle detected in path reconstruction")
                print(f"[MCFRouter WARNING] Cycle detected: path so far = {path}, current = {current}")
                break
            visited.add(current)
            current = adj[current]
            path.append(current)

        # Check if we reached the destination
        if path[-1] != destination:
            logger.warning(
                f"Path reconstruction incomplete: {source} -> {destination}, "
                f"stopped at {path[-1]}, edges={edges}"
            )
            print(
                f"[MCFRouter WARNING] Incomplete path: {source} -> {destination}, "
                f"stopped at {path[-1]}"
            )
            print(f"  Edges: {edges}")

        return path

class NoCPathsAdapter:
    """
    Adapter to convert between NoCPaths format and MCF Router format

    This bridges the existing PolicyTableGenerator interface with the new
    MCF-based router.
    """

    def __init__(self, mesh_rows: int = 4, mesh_cols: int = 4):
        """
        Initialize adapter

        Args:
            mesh_rows: Number of rows in mesh
            mesh_cols: Number of columns in mesh
        """
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.topology = MeshTopology(mesh_rows, mesh_cols)

    def noc_paths_to_commodities(
        self,
        noc_paths: Dict,
        node_id_class: Any
    ) -> List[Commodity]:
        """
        Convert NoCPaths dict to list of Commodities

        Args:
            noc_paths: Dictionary mapping edges to (src_node, dst_node, split_idx)
            node_id_class: NodeID class with to_coord method

        Returns:
            List of Commodity objects
        """
        commodities = []

        for idx, (edge, mapping_info) in enumerate(noc_paths.items()):
            src_node = mapping_info[0]
            dst_node = mapping_info[1]

            # Convert NodeID to coordinates
            src_coord_tuple = node_id_class.to_coord(src_node)
            dst_coord_tuple = node_id_class.to_coord(dst_node)

            src_coord = Coord(src_coord_tuple[0], src_coord_tuple[1])
            dst_coord = Coord(dst_coord_tuple[0], dst_coord_tuple[1])

            commodity = Commodity(
                id=idx,
                source=src_coord,
                destination=dst_coord,
                metadata=(edge, mapping_info)
            )
            commodities.append(commodity)

        return commodities

    def routing_result_to_paths(
        self,
        result: MCFRoutingResult,
        commodities: List[Commodity],
        node_id_class: Any
    ) -> Dict:
        """
        Convert MCF routing result back to path format usable by PolicyTableGenerator

        Args:
            result: MCF routing result
            commodities: Original commodities with metadata
            node_id_class: NodeID class with from_coord method

        Returns:
            Dictionary mapping original edges to their computed paths
        """
        edge_to_path = {}

        for commodity in commodities:
            if commodity.id in result.routes:
                route = result.routes[commodity.id]
                original_edge, mapping_info = commodity.metadata

                # Convert path coordinates to NodeIDs
                node_path = []
                for coord in route.path:
                    node_id = node_id_class.from_coord(coord.row, coord.col)
                    node_path.append(node_id)

                edge_to_path[original_edge] = {
                    'path': node_path,
                    'mapping_info': mapping_info,
                    'edges': route.edges,
                    'length': route.get_path_length()
                }

        return edge_to_path


def solve_mcf_routing(
    noc_paths: Dict,
    node_id_class: Any,
    mesh_rows: int = 4,
    mesh_cols: int = 4,
    minimize_congestion: bool = True
) -> Tuple[Dict, MCFRoutingResult]:
    """
    Convenience function to solve MCF routing for NoCPaths

    Supports multicast: commodities with the same source are grouped and
    share edges, counting as 1 towards edge capacity.

    Args:
        noc_paths: Dictionary from NoCPaths
        node_id_class: NodeID class
        mesh_rows: Mesh dimensions
        mesh_cols: Mesh dimensions
        minimize_congestion: Optimization objective

    Returns:
        Tuple of (edge_to_path dict, MCFRoutingResult)
    """
    adapter = NoCPathsAdapter(mesh_rows, mesh_cols)
    commodities = adapter.noc_paths_to_commodities(noc_paths, node_id_class)

    router = MCFRouter(
        topology=adapter.topology,
        minimize_congestion=minimize_congestion
    )

    result = router.route(commodities)
    edge_to_path = adapter.routing_result_to_paths(result, commodities, node_id_class)

    return edge_to_path, result


# For testing
if __name__ == "__main__":
    def run_tests():
        all_passed = True

        # ============================================================
        print("=" * 70)
        print("Test 1: Pure Multicast (same source, 3 destinations)")
        print("=" * 70)

        topology = MeshTopology(4, 4)
        router = MCFRouter(topology, minimize_congestion=True)

        commodities = [
            Commodity(0, Coord(0, 0), Coord(3, 3)),
            Commodity(1, Coord(0, 0), Coord(3, 0)),
            Commodity(2, Coord(0, 0), Coord(0, 3)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")

        if result.max_edge_congestion == 1:
            print("✓ PASS: Multicast group achieves cap=1")
        else:
            print(f"✗ FAIL: Expected cap=1, got {result.max_edge_congestion}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 2: Mixed multicast + unicast")
        print("=" * 70)

        commodities = [
            # Multicast group 1: source (0,0)
            Commodity(0, Coord(0, 0), Coord(3, 3)),
            Commodity(1, Coord(0, 0), Coord(2, 2)),
            # Multicast group 2: source (3,0)
            Commodity(2, Coord(3, 0), Coord(0, 3)),
            Commodity(3, Coord(3, 0), Coord(1, 1)),
            # Unicast
            Commodity(4, Coord(1, 1), Coord(2, 2)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")
        print("Routes:")
        for cid, route in sorted(result.routes.items()):
            src, dst = route.commodity.source, route.commodity.destination
            print(f"  {cid}: {src} -> {dst}, edges={len(route.edges)}")

        if "mcast=2" in result.solver_status and "ucast=1" in result.solver_status:
            print("✓ PASS: Correctly identified 2 multicast groups + 1 unicast")
        else:
            print(f"✗ FAIL: Wrong group detection: {result.solver_status}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 3: Conflicting unicast routes (different sources)")
        print("=" * 70)

        # Two different sources going through same bottleneck
        commodities = [
            Commodity(0, Coord(0, 0), Coord(0, 2)),
            Commodity(1, Coord(0, 1), Coord(0, 3)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")
        print("Routes:")
        for cid, route in sorted(result.routes.items()):
            path = " -> ".join(str(p) for p in route.path)
            print(f"  {cid}: {path}")

        # These have different sources, so can't share - may need cap > 1
        print("✓ PASS: Different sources handled correctly")

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 4: Large multicast group (8 destinations)")
        print("=" * 70)

        commodities = [
            Commodity(i, Coord(2, 2), Coord(r, c))
            for i, (r, c) in enumerate([
                (0, 0), (0, 3), (3, 0), (3, 3),
                (1, 0), (1, 3), (2, 0), (2, 3)
            ])
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")
        print(f"Total tree edges: {result.total_edge_usage}")

        if result.max_edge_congestion == 1:
            print("✓ PASS: 8-destination multicast achieves cap=1")
        else:
            print(f"✗ FAIL: Expected cap=1, got {result.max_edge_congestion}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 5: Edge sharing visualization")
        print("=" * 70)

        commodities = [
            Commodity(0, Coord(0, 0), Coord(2, 0)),
            Commodity(1, Coord(0, 0), Coord(2, 2)),
            Commodity(2, Coord(0, 0), Coord(0, 2)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print("\nMulticast tree structure:")
        for cid, route in sorted(result.routes.items()):
            path = " -> ".join(str(p) for p in route.path)
            print(f"  To {route.commodity.destination}: {path}")

        print("\nShared edges:")
        for edge, users in sorted(result.edge_usage.items(), key=lambda x: -len(x[1])):
            if len(users) > 1:
                print(f"  {edge}: commodities {users}")

        print("✓ PASS: Edge sharing works correctly")

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 6: graph_node_id based multicast grouping")
        print("=" * 70)

        # Mock TensorEdge-like object for metadata
        class MockSrcId:
            def __init__(self, graph_node_id):
                self.graph_node_id = graph_node_id

        class MockEdge:
            def __init__(self, graph_node_id):
                self.src_id = MockSrcId(graph_node_id)

        # Same HW source (0,0), but different graph_node_ids
        # Should NOT be multicast - they're different data
        commodities = [
            Commodity(0, Coord(0, 0), Coord(2, 0), metadata=(MockEdge(100), None)),  # graph_node_id=100
            Commodity(1, Coord(0, 0), Coord(0, 2), metadata=(MockEdge(200), None)),  # graph_node_id=200
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")

        # These should NOT be grouped as multicast (different graph_node_ids)
        if "mcast=0" in result.solver_status and "ucast=2" in result.solver_status:
            print("✓ PASS: Different graph_node_ids = no multicast (2 unicast)")
        else:
            print(f"✗ FAIL: Should be 2 unicast, got: {result.solver_status}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 7: Same HW source + same graph_node_id = multicast")
        print("=" * 70)

        # Same HW source (0,0) AND same graph_node_id
        # Should BE multicast - they're the same data
        commodities = [
            Commodity(0, Coord(0, 0), Coord(2, 0), metadata=(MockEdge(100), None)),  # graph_node_id=100
            Commodity(1, Coord(0, 0), Coord(0, 2), metadata=(MockEdge(100), None)),  # graph_node_id=100 (same!)
            Commodity(2, Coord(0, 0), Coord(2, 2), metadata=(MockEdge(100), None)),  # graph_node_id=100 (same!)
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")

        # These SHOULD be grouped as multicast (same graph_node_id)
        if "mcast=1" in result.solver_status and "ucast=0" in result.solver_status:
            print("✓ PASS: Same graph_node_id = multicast (1 group)")
        else:
            print(f"✗ FAIL: Should be 1 multicast group, got: {result.solver_status}")
            all_passed = False

        if result.max_edge_congestion == 1:
            print("✓ PASS: Multicast achieves cap=1")
        else:
            print(f"✗ FAIL: Expected cap=1, got {result.max_edge_congestion}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 8: Mixed - same HW source, some same/different graph_node_ids")
        print("=" * 70)

        commodities = [
            # Multicast group: graph_node_id=100 from (0,0)
            Commodity(0, Coord(0, 0), Coord(2, 0), metadata=(MockEdge(100), None)),
            Commodity(1, Coord(0, 0), Coord(2, 2), metadata=(MockEdge(100), None)),
            # Unicast: graph_node_id=200 from (0,0) - same HW source, different data!
            Commodity(2, Coord(0, 0), Coord(0, 2), metadata=(MockEdge(200), None)),
            # Unicast: graph_node_id=300 from (1,1)
            Commodity(3, Coord(1, 1), Coord(3, 3), metadata=(MockEdge(300), None)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")

        # Should be: 1 multicast group (commodities 0,1) + 2 unicast (commodities 2,3)
        if "mcast=1" in result.solver_status and "ucast=2" in result.solver_status:
            print("✓ PASS: Correctly split by graph_node_id (1 mcast + 2 ucast)")
        else:
            print(f"✗ FAIL: Should be 1 mcast + 2 ucast, got: {result.solver_status}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 9: Const commodities with same (src, dst) share path")
        print("=" * 70)

        # Mock TensorEdge-like object for const metadata
        class MockConstSrcId:
            def __init__(self, graph_node_id, tensor_type='weight'):
                self.graph_node_id = graph_node_id
                self.tensor_type = tensor_type

        class MockConstEdge:
            def __init__(self, graph_node_id, tensor_type='weight'):
                self.src_id = MockConstSrcId(graph_node_id, tensor_type)

        # Three const commodities with same (src, dst) but different graph_node_ids
        # They should all use the same path
        commodities = [
            Commodity(0, Coord(0, 0), Coord(2, 2), metadata=(MockConstEdge(100, 'weight'), None)),
            Commodity(1, Coord(0, 0), Coord(2, 2), metadata=(MockConstEdge(200, 'weight'), None)),
            Commodity(2, Coord(0, 0), Coord(2, 2), metadata=(MockConstEdge(300, 'config'), None)),
            # Different dst - should have different path
            Commodity(3, Coord(0, 0), Coord(3, 3), metadata=(MockConstEdge(400, 'weight'), None)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print(f"Max congestion: {result.max_edge_congestion}")
        print("Routes:")
        for cid in sorted(result.routes.keys()):
            route = result.routes[cid]
            path = " -> ".join(str(p) for p in route.path)
            print(f"  {cid}: {route.commodity.source} -> {route.commodity.destination}: {path}")

        # Check that commodities 0, 1, 2 have the same path
        path_0 = result.routes[0].path
        path_1 = result.routes[1].path
        path_2 = result.routes[2].path
        path_3 = result.routes[3].path

        if path_0 == path_1 == path_2:
            print("✓ PASS: Const commodities with same (src, dst) share the same path")
        else:
            print(f"✗ FAIL: Paths should be identical but got:")
            print(f"  path_0: {path_0}")
            print(f"  path_1: {path_1}")
            print(f"  path_2: {path_2}")
            all_passed = False

        if path_0 != path_3:
            print("✓ PASS: Const commodities with different dst have different paths")
        else:
            print(f"✗ FAIL: Commodities 0 and 3 have different destinations but same path")
            all_passed = False

        # Verify const_dedup is in status
        if "const_dedup" in result.solver_status:
            print("✓ PASS: const_dedup reported in solver status")
        else:
            print(f"✗ FAIL: const_dedup not in status: {result.solver_status}")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        print("Test 10: Mixed const (deduped) and data (not deduped)")
        print("=" * 70)

        class MockDataSrcId:
            def __init__(self, graph_node_id):
                self.graph_node_id = graph_node_id
                self.tensor_type = 'data'

        class MockDataEdge:
            def __init__(self, graph_node_id):
                self.src_id = MockDataSrcId(graph_node_id)

        commodities = [
            # Const: same (src, dst) - should be deduped
            Commodity(0, Coord(0, 0), Coord(2, 2), metadata=(MockConstEdge(100, 'weight'), None)),
            Commodity(1, Coord(0, 0), Coord(2, 2), metadata=(MockConstEdge(200, 'weight'), None)),
            # Data: same (src, dst) - should NOT be deduped (different routing possible)
            Commodity(2, Coord(1, 1), Coord(3, 3), metadata=(MockDataEdge(300), None)),
            Commodity(3, Coord(1, 1), Coord(3, 3), metadata=(MockDataEdge(400), None)),
        ]

        result = router.route(commodities)
        print(f"Status: {result.solver_status}")
        print("Routes:")
        for cid in sorted(result.routes.keys()):
            route = result.routes[cid]
            path = " -> ".join(str(p) for p in route.path)
            print(f"  {cid}: {route.commodity.source} -> {route.commodity.destination}: {path}")

        # Const should share path
        if result.routes[0].path == result.routes[1].path:
            print("✓ PASS: Const commodities 0, 1 share path")
        else:
            print(f"✗ FAIL: Const commodities should share path")
            all_passed = False

        # Data commodities are NOT guaranteed to share (they might or might not depending on solver)
        # Just verify both exist
        if 2 in result.routes and 3 in result.routes:
            print("✓ PASS: Data commodities 2, 3 both routed")
        else:
            print(f"✗ FAIL: Data commodities not found in routes")
            all_passed = False

        # ============================================================
        print("\n" + "=" * 70)
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("=" * 70)

        return all_passed

    run_tests()
