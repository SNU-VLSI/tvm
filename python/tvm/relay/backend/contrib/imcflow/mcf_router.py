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

logger = logging.getLogger(__name__)


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
class MCFRoutingResult:
    """Complete result of multi-commodity flow routing"""
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
        edge_capacity: int = 1,
        minimize_congestion: bool = True,
    ):
        """
        Initialize MCF Router

        Args:
            topology: The mesh topology
            edge_capacity: Maximum number of commodities that can share an edge
                          (use large number for soft constraint)
            minimize_congestion: If True, minimize max congestion. If False, minimize total usage.
        """
        self.topology = topology
        self.edge_capacity = edge_capacity
        self.minimize_congestion = minimize_congestion
        self._solver = None

    def _check_pulp_available(self) -> bool:
        """Check if PuLP is available"""
        try:
            import pulp
            return True
        except ImportError:
            return False

    def route(self, commodities: List[Commodity]) -> MCFRoutingResult:
        """
        Route all commodities using ILP

        Args:
            commodities: List of commodities to route

        Returns:
            MCFRoutingResult containing all routes and statistics
        """
        if not commodities:
            return MCFRoutingResult(
                routes={},
                edge_usage={},
                max_edge_congestion=0,
                total_edge_usage=0,
                solver_status="empty"
            )

        # Filter out same-node commodities
        valid_commodities = [c for c in commodities if c.source != c.destination]

        if not valid_commodities:
            return MCFRoutingResult(
                routes={c.id: RoutingResult(c, [c.source], []) for c in commodities},
                edge_usage={},
                max_edge_congestion=0,
                total_edge_usage=0,
                solver_status="all_local"
            )

        if not self._check_pulp_available():
            logger.warning("PuLP not available, falling back to greedy routing")
            return self._route_greedy(valid_commodities)

        return self._route_ilp(valid_commodities)

    def _route_ilp(self, commodities: List[Commodity]) -> MCFRoutingResult:
        """Route using ILP solver (PuLP)"""
        import pulp

        # Create problem
        if self.minimize_congestion:
            prob = pulp.LpProblem("MCF_MinCongestion", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem("MCF_MinTotalUsage", pulp.LpMinimize)

        # Get all edges
        all_edges = self.topology.get_all_edges()
        edge_to_idx = {e: i for i, e in enumerate(all_edges)}

        # Create binary variables x[k][e] = 1 if commodity k uses edge e
        x = {}
        for k in commodities:
            x[k.id] = {}
            for e in all_edges:
                x[k.id][e] = pulp.LpVariable(
                    f"x_{k.id}_{edge_to_idx[e]}",
                    cat=pulp.LpBinary
                )

        # Create congestion variable for each edge (if minimizing congestion)
        if self.minimize_congestion:
            max_congestion = pulp.LpVariable("max_congestion", lowBound=0, cat=pulp.LpInteger)

        # Edge usage count variables
        edge_usage_vars = {}
        for e in all_edges:
            edge_usage_vars[e] = pulp.LpVariable(
                f"usage_{edge_to_idx[e]}",
                lowBound=0,
                cat=pulp.LpInteger
            )
            # Edge usage = sum of all commodities using this edge
            prob += edge_usage_vars[e] == pulp.lpSum(x[k.id][e] for k in commodities)

        # Objective function
        if self.minimize_congestion:
            # Minimize maximum congestion
            prob += max_congestion
            # Constraint: max_congestion >= usage of each edge
            for e in all_edges:
                prob += max_congestion >= edge_usage_vars[e]
        else:
            # Minimize total edge usage
            prob += pulp.lpSum(edge_usage_vars[e] for e in all_edges)

        # Flow conservation constraints
        for k in commodities:
            for node in self.topology.get_all_nodes():
                # Calculate in-flow and out-flow
                in_edges = [e for e in all_edges if e.dst == node]
                out_edges = [e for e in all_edges if e.src == node]

                in_flow = pulp.lpSum(x[k.id][e] for e in in_edges) if in_edges else 0
                out_flow = pulp.lpSum(x[k.id][e] for e in out_edges) if out_edges else 0

                if node == k.source:
                    # Source: out_flow - in_flow = 1
                    prob += out_flow - in_flow == 1, f"flow_conservation_{k.id}_source"
                elif node == k.destination:
                    # Destination: in_flow - out_flow = 1
                    prob += in_flow - out_flow == 1, f"flow_conservation_{k.id}_dest"
                else:
                    # Intermediate: in_flow = out_flow
                    prob += in_flow == out_flow, f"flow_conservation_{k.id}_{node.row}_{node.col}"

        # Optional: Hard capacity constraints
        if self.edge_capacity < len(commodities):
            for e in all_edges:
                prob += edge_usage_vars[e] <= self.edge_capacity

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            logger.warning(f"ILP solver status: {pulp.LpStatus[status]}, falling back to greedy")
            return self._route_greedy(commodities)

        # Extract routes from solution
        routes = {}
        edge_usage = {e: [] for e in all_edges}

        for k in commodities:
            # Find edges used by this commodity
            used_edges = [e for e in all_edges if pulp.value(x[k.id][e]) > 0.5]

            # Reconstruct path from edges
            path = self._reconstruct_path(k.source, k.destination, used_edges)
            routes[k.id] = RoutingResult(k, path, used_edges)

            for e in used_edges:
                edge_usage[e].append(k.id)

        # Calculate statistics
        max_cong = max(len(v) for v in edge_usage.values()) if edge_usage else 0
        total_usage = sum(len(v) for v in edge_usage.values())

        return MCFRoutingResult(
            routes=routes,
            edge_usage={e: v for e, v in edge_usage.items() if v},
            max_edge_congestion=max_cong,
            total_edge_usage=total_usage,
            solver_status=pulp.LpStatus[status]
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

        # Build adjacency from edges
        adj = {}
        for e in edges:
            adj[e.src] = e.dst

        path = [source]
        current = source
        visited = set()

        while current != destination and current in adj:
            if current in visited:
                logger.warning(f"Cycle detected in path reconstruction")
                break
            visited.add(current)
            current = adj[current]
            path.append(current)

        return path

    def _route_greedy(self, commodities: List[Commodity]) -> MCFRoutingResult:
        """
        Fallback greedy routing when ILP is not available

        Uses shortest path with congestion-aware cost function
        """
        routes = {}
        edge_usage: Dict[Edge, List[int]] = {}

        # Sort commodities by Manhattan distance (shorter first)
        sorted_commodities = sorted(
            commodities,
            key=lambda c: abs(c.source.row - c.destination.row) +
                         abs(c.source.col - c.destination.col)
        )

        for k in sorted_commodities:
            # Find path using BFS with congestion awareness
            path = self._find_path_bfs(k.source, k.destination, edge_usage)

            if path:
                edges = []
                for i in range(len(path) - 1):
                    e = Edge(path[i], path[i + 1])
                    edges.append(e)
                    if e not in edge_usage:
                        edge_usage[e] = []
                    edge_usage[e].append(k.id)

                routes[k.id] = RoutingResult(k, path, edges)
            else:
                # Fallback to X-Y routing
                path = self._xy_routing(k.source, k.destination)
                edges = []
                for i in range(len(path) - 1):
                    e = Edge(path[i], path[i + 1])
                    edges.append(e)
                    if e not in edge_usage:
                        edge_usage[e] = []
                    edge_usage[e].append(k.id)
                routes[k.id] = RoutingResult(k, path, edges)

        max_cong = max(len(v) for v in edge_usage.values()) if edge_usage else 0
        total_usage = sum(len(v) for v in edge_usage.values())

        return MCFRoutingResult(
            routes=routes,
            edge_usage={e: v for e, v in edge_usage.items() if v},
            max_edge_congestion=max_cong,
            total_edge_usage=total_usage,
            solver_status="greedy"
        )

    def _find_path_bfs(
        self,
        source: Coord,
        destination: Coord,
        current_usage: Dict[Edge, List[int]]
    ) -> Optional[List[Coord]]:
        """BFS with congestion-aware edge cost"""
        from collections import deque

        queue = deque([(source, [source], 0)])
        visited = {source: 0}

        best_path = None
        best_cost = float('inf')

        while queue:
            current, path, cost = queue.popleft()

            if current == destination:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                continue

            if cost >= best_cost:
                continue

            for neighbor, _ in self.topology.get_neighbors(current):
                edge = Edge(current, neighbor)
                edge_cost = 1 + len(current_usage.get(edge, []))
                new_cost = cost + edge_cost

                if neighbor not in visited or visited[neighbor] > new_cost:
                    visited[neighbor] = new_cost
                    queue.append((neighbor, path + [neighbor], new_cost))

        return best_path

    def _xy_routing(self, source: Coord, destination: Coord) -> List[Coord]:
        """Simple X-Y routing (move horizontally first, then vertically)"""
        path = [source]
        current = source

        # Move horizontally (X direction)
        while current.col != destination.col:
            step = 1 if current.col < destination.col else -1
            current = Coord(current.row, current.col + step)
            path.append(current)

        # Move vertically (Y direction)
        while current.row != destination.row:
            step = 1 if current.row < destination.row else -1
            current = Coord(current.row + step, current.col)
            path.append(current)

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
    edge_capacity: int = 32,
    minimize_congestion: bool = True
) -> Tuple[Dict, MCFRoutingResult]:
    """
    Convenience function to solve MCF routing for NoCPaths

    Args:
        noc_paths: Dictionary from NoCPaths
        node_id_class: NodeID class
        mesh_rows: Mesh dimensions
        mesh_cols: Mesh dimensions
        edge_capacity: Max commodities per edge
        minimize_congestion: Optimization objective

    Returns:
        Tuple of (edge_to_path dict, MCFRoutingResult)
    """
    adapter = NoCPathsAdapter(mesh_rows, mesh_cols)
    commodities = adapter.noc_paths_to_commodities(noc_paths, node_id_class)

    router = MCFRouter(
        topology=adapter.topology,
        edge_capacity=edge_capacity,
        minimize_congestion=minimize_congestion
    )

    result = router.route(commodities)
    edge_to_path = adapter.routing_result_to_paths(result, commodities, node_id_class)

    return edge_to_path, result


# For testing
if __name__ == "__main__":
    # Create a simple test case
    topology = MeshTopology(4, 4)
    router = MCFRouter(topology, minimize_congestion=True)

    # Create test commodities
    commodities = [
        Commodity(0, Coord(0, 0), Coord(3, 3)),
        Commodity(1, Coord(0, 1), Coord(3, 2)),
        Commodity(2, Coord(1, 0), Coord(2, 3)),
        Commodity(3, Coord(0, 0), Coord(2, 2)),
    ]

    result = router.route(commodities)

    print(f"Solver status: {result.solver_status}")
    print(f"Max congestion: {result.max_edge_congestion}")
    print(f"Total edge usage: {result.total_edge_usage}")
    print("\nRoutes:")
    for cid, route in result.routes.items():
        print(f"  Commodity {cid}: {route.path}")

    print("\nCongested edges:")
    for edge, users in result.edge_usage.items():
        if len(users) > 1:
            print(f"  {edge}: used by commodities {users}")
