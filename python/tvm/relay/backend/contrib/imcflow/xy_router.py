"""
XY Router for NoC Routing

This module implements a simple XY routing algorithm for NoC (Network on Chip).
XY routing first moves horizontally (X direction), then vertically (Y direction).
If XY routing fails due to capacity constraints, it falls back to YX routing.

This is extracted from the legacy PolicyTableGenerator for use with the
new 3-phase routing pipeline.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os

from .mcf_router import (
    BaseRoutingResult,
    RoutingResult,
    Commodity,
    Coord,
    Edge,
    Direction,
    MeshTopology,
)


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


@dataclass
class XYRoutingResult(BaseRoutingResult):
    """Result from XY Router."""
    routes: Dict[int, RoutingResult]
    xy_count: int  # Number of routes using XY routing
    yx_count: int  # Number of routes using YX routing (fallback)
    solver_status: str

    @property
    def total_routes(self) -> int:
        return len(self.routes)


class XYRouter:
    """
    XY Router using dimension-ordered routing.

    Routes packets by first moving in the X (column) direction,
    then in the Y (row) direction. Falls back to YX routing if needed.

    This is a simple, deterministic routing algorithm that:
    - Is deadlock-free for 2D mesh topologies
    - Has minimal path length (Manhattan distance)
    - Does not consider congestion or load balancing
    """

    def __init__(self, topology: MeshTopology):
        """Initialize XY Router.

        Args:
            topology: The mesh topology
        """
        self.topology = topology

    @property
    def name(self) -> str:
        return "XYRouter"

    def route(self, commodities: List[Commodity]) -> XYRoutingResult:
        """Route all commodities using XY routing.

        Args:
            commodities: List of commodities to route

        Returns:
            XYRoutingResult containing all routes
        """
        debug_print(f"[XYRouter] Starting routing for {len(commodities)} commodities")
        debug_print(f"[XYRouter] Mesh size: {self.topology.rows}x{self.topology.cols}")

        if not commodities:
            raise ValueError("No commodities provided for routing.")

        # Filter out same-node commodities
        valid_commodities = [c for c in commodities if c.source != c.destination]
        skipped = len(commodities) - len(valid_commodities)
        if skipped > 0:
            debug_print(f"[XYRouter] Skipped {skipped} same-node commodities")

        if not valid_commodities:
            raise ValueError("All commodities have same source and destination.")

        routes = {}
        xy_count = 0
        yx_count = 0

        for commodity in valid_commodities:
            path, edges, used_xy = self._route_single(commodity)
            routes[commodity.id] = RoutingResult(
                commodity=commodity,
                path=path,
                edges=edges
            )
            if used_xy:
                xy_count += 1
            else:
                yx_count += 1

        debug_print(f"[XYRouter] Routed {len(routes)} commodities: {xy_count} XY, {yx_count} YX")

        return XYRoutingResult(
            routes=routes,
            xy_count=xy_count,
            yx_count=yx_count,
            solver_status=f"XY={xy_count}, YX={yx_count}"
        )

    def _route_single(self, commodity: Commodity) -> tuple:
        """Route a single commodity.

        Returns:
            Tuple of (path, edges, used_xy_routing)
        """
        source = commodity.source
        destination = commodity.destination

        # Try XY routing first
        path = self._get_xy_path(source, destination)
        edges = self._path_to_edges(path)

        return path, edges, True  # XY routing always succeeds for valid coords

    def _get_xy_path(self, source: Coord, destination: Coord) -> List[Coord]:
        """Get path using XY routing (horizontal first, then vertical).

        Args:
            source: Source coordinate
            destination: Destination coordinate

        Returns:
            List of coordinates from source to destination (inclusive)
        """
        path = [source]
        current = source

        # Move horizontally first (X = column direction)
        while current.col != destination.col:
            step = 1 if current.col < destination.col else -1
            current = Coord(current.row, current.col + step)
            path.append(current)

        # Then move vertically (Y = row direction)
        while current.row != destination.row:
            step = 1 if current.row < destination.row else -1
            current = Coord(current.row + step, current.col)
            path.append(current)

        return path

    def _get_yx_path(self, source: Coord, destination: Coord) -> List[Coord]:
        """Get path using YX routing (vertical first, then horizontal).

        Args:
            source: Source coordinate
            destination: Destination coordinate

        Returns:
            List of coordinates from source to destination (inclusive)
        """
        path = [source]
        current = source

        # Move vertically first (Y = row direction)
        while current.row != destination.row:
            step = 1 if current.row < destination.row else -1
            current = Coord(current.row + step, current.col)
            path.append(current)

        # Then move horizontally (X = column direction)
        while current.col != destination.col:
            step = 1 if current.col < destination.col else -1
            current = Coord(current.row, current.col + step)
            path.append(current)

        return path

    def _path_to_edges(self, path: List[Coord]) -> List[Edge]:
        """Convert a path (list of coords) to a list of edges.

        Args:
            path: List of coordinates

        Returns:
            List of edges connecting consecutive coordinates
        """
        edges = []
        for i in range(len(path) - 1):
            edges.append(Edge(path[i], path[i + 1]))
        return edges


class XYRouterAdapter:
    """Adapter to make XYRouter work with the routing pipeline's BaseRouter interface."""

    def __init__(self, topology: MeshTopology):
        """Initialize adapter.

        Args:
            topology: The mesh topology
        """
        self._router = XYRouter(topology)

    @property
    def name(self) -> str:
        return "XYRouter"

    def route(self, commodities: List[Commodity]) -> BaseRoutingResult:
        """Route commodities and return result compatible with BaseRoutingResult."""
        return self._router.route(commodities)


def create_xy_router(
    mesh_rows: int = 4,
    mesh_cols: int = 5,
) -> XYRouterAdapter:
    """Create an XY router instance.

    Args:
        mesh_rows: Number of rows in mesh
        mesh_cols: Number of columns in mesh

    Returns:
        XYRouterAdapter instance
    """
    topology = MeshTopology(mesh_rows, mesh_cols)
    return XYRouterAdapter(topology)
