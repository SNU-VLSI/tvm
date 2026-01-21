"""
NoC Routing Pipeline with Pluggable Routers

This module provides a 3-phase routing pipeline for generating policy tables:
- Phase 1 (Router): Generate paths - PLUGGABLE (MCF, XY, custom, etc.)
- Phase 2 (Tree Builder): Build multicast trees - REUSABLE
- Phase 3 (PolicyTableBuilder): Generate policy entries - REUSABLE

Architecture:
                    ┌─────────────────┐
                    │   BaseRouter    │  ← Abstract interface
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │MCFRouter │   │ XYRouter │   │ Future   │
        └──────────┘   └──────────┘   └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ BaseRoutingResult│
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ PathTreeBuilder │  ← Reusable
                    └────────┬────────┘
                             ▼
                    ┌───────────────────┐
                    │PolicyTableBuilder │  ← Reusable
                    └───────────────────┘

Usage:
    # With MCF Router (default)
    pipeline = RoutingPipeline(router=MCFRouter(topology))
    pipeline.run(mod, noc_paths)

    # With custom router
    my_router = MyCustomRouter(topology)
    pipeline = RoutingPipeline(router=my_router)
    pipeline.run(mod, noc_paths)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, NodeID

from .mcf_router import (
    BaseRoutingResult,
    MCFRouter,
    MeshTopology,
    Commodity,
    Coord,
)
from .xy_router import (
    XYRouter,
    XYRouterAdapter,
    create_xy_router,
)
from .path_tree_builder import (
    PathTreeBuilder,
    PathTreeBuildResult,
)
from .policy_table_generator import (
    generate_policy_tables,
    NodeCapacityError,
)


class BaseRouter(ABC):
    """Abstract base class for all routers.

    All router implementations must inherit from this class and implement
    the route() method. This ensures consistent interface across different
    routing algorithms.
    """

    @abstractmethod
    def route(self, commodities: List[Commodity]) -> BaseRoutingResult:
        """Route all commodities and return routing result.

        Args:
            commodities: List of commodities (source, destination pairs) to route

        Returns:
            BaseRoutingResult containing paths for all commodities
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this router for logging/debugging."""
        pass


class MCFRouterAdapter(BaseRouter):
    """Adapter to make MCFRouter conform to BaseRouter interface."""

    def __init__(
        self,
        topology: MeshTopology,
        minimize_congestion: bool = True,
    ):
        self._router = MCFRouter(topology, minimize_congestion=minimize_congestion, enforce_destination_disjoint=True)
        self._topology = topology
        self._minimize_congestion = minimize_congestion

    def route(self, commodities: List[Commodity]) -> BaseRoutingResult:
        return self._router.route(commodities)

    @property
    def name(self) -> str:
        return "MCFRouter"


@dataclass
class PipelineConfig:
    """Configuration for the routing pipeline."""
    mesh_rows: int = 4
    mesh_cols: int = 5  # 4 rows x 5 cols: inode at col 0, imce at cols 1-4
    table_capacity: int = 32
    tensor_id_extractor: Optional[Callable] = None


class RoutingPipeline:
    """
    3-Phase Routing Pipeline with pluggable router.

    This is the main entry point for policy table generation. It orchestrates:
    - Phase 1: Routing (using provided router)
    - Phase 2: Tree building (PathTreeBuilder)
    - Phase 3: Policy table generation (PolicyTableBuilder)

    The router is pluggable - you can use MCFRouter, XYRouter, or any custom
    router that implements the BaseRouter interface.
    """

    def __init__(
        self,
        router: BaseRouter,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize the routing pipeline.

        Args:
            router: Router instance implementing BaseRouter interface
            config: Pipeline configuration (optional)
        """
        self.router = router
        self.config = config or PipelineConfig()
        self.PolicyTable_2D = {}

    def run(self, mod, noc_paths_dict: Dict[str, Dict]):
        """Run the routing pipeline for all functions in the module.

        Args:
            mod: TVM module containing imcflow functions
            noc_paths_dict: Dictionary mapping function names to their NoCPaths

        Returns:
            The module (unchanged)
        """
        for gv, func in mod.functions.items():
            if (isinstance(func, relay.Function) and
                hasattr(func.attrs, "Compiler") and
                func.attrs["Compiler"] == "imcflow"):

                func_name = gv.name_hint
                if func_name not in noc_paths_dict:
                    continue

                noc_paths = noc_paths_dict[func_name]

                try:
                    policy_tables = self._run_for_function(func_name, noc_paths)
                    self.PolicyTable_2D[func_name] = policy_tables
                except NodeCapacityError as e:
                    raise RuntimeError(f"Routing failed for {func_name}: {e}")

        return mod

    def _run_for_function(self, func_name: str, noc_paths: Dict) -> Dict:
        """Run pipeline for a single function."""
        # Phase 1: Route
        commodities = self._noc_paths_to_commodities(noc_paths)
        if not commodities:
            # No commodities to route - return empty tables
            return {node_id: [self._zero_entry()] for node_id in NodeID}

        routing_result = self.router.route(commodities)

        # Dump routing results for debugging
        self._dump_routing_results(func_name, routing_result, noc_paths)

        # Phase 2: Build trees
        tree_builder = PathTreeBuilder(
            tensor_id_extractor=self.config.tensor_id_extractor or self._default_tensor_id_extractor
        )
        tree_result = tree_builder.build(routing_result)

        # Phase 3: Generate policy tables
        policy_tables = generate_policy_tables(
            tree_result,
            noc_paths,
            func_name,
            table_capacity=self.config.table_capacity,
        )

        return policy_tables

    def _dump_routing_results(self, func_name: str, routing_result, noc_paths: Dict) -> None:
        """Dump routing results for debugging.

        Creates a file showing each noc_path edge and the corresponding route.
        """
        import os

        # Get output directory from environment or use default
        output_dir = os.environ.get('IMCFLOW_DEBUG_DIR', '/tmp')
        output_file = os.path.join(output_dir, f'routing_result_{func_name}.txt')

        with open(output_file, 'w') as f:
            f.write(f"Routing Results for {func_name}\n")
            f.write("=" * 80 + "\n\n")

            # Build a map from (src_coord, dst_coord) to edges for lookup
            coord_to_edges = {}
            for edge, mapping_info in noc_paths.items():
                src_node = mapping_info[0]
                dst_node = mapping_info[1]
                src_coord = NodeID.to_coord(src_node)
                dst_coord = NodeID.to_coord(dst_node)
                key = (src_coord, dst_coord)
                if key not in coord_to_edges:
                    coord_to_edges[key] = []
                coord_to_edges[key].append((edge, mapping_info))

            # Dump each commodity's route
            for cid in routing_result.get_all_commodity_ids():
                commodity = routing_result.get_commodity(cid)
                path = routing_result.get_path(cid)

                src = commodity.source
                dst = commodity.destination

                # Find corresponding edge
                edge_info = ""
                if commodity.metadata:
                    edge, mapping_info = commodity.metadata
                    src_node = mapping_info[0]
                    dst_node = mapping_info[1]
                    edge_info = f"{src_node.name} -> {dst_node.name}"
                    if isinstance(edge, TensorEdge):
                        edge_info += f" [{edge.src_id.tensor_type}]"

                # Format path
                path_str = ' -> '.join(f'({c.row},{c.col})' for c in path)

                # Check for detours
                detour_info = ""
                expected_path = self._compute_xy_path(src, dst)
                if len(path) != len(expected_path):
                    detour_info = f" [DETOUR: expected {len(expected_path)} hops, got {len(path)}]"

                # Check directions used
                directions_used = []
                for i in range(len(path) - 1):
                    curr, next_ = path[i], path[i + 1]
                    if next_.row < curr.row:
                        directions_used.append('N')
                    elif next_.row > curr.row:
                        directions_used.append('S')
                    elif next_.col > curr.col:
                        directions_used.append('E')
                    elif next_.col < curr.col:
                        directions_used.append('W')

                f.write(f"Commodity {cid}: {edge_info}\n")
                f.write(f"  Source: ({src.row},{src.col}), Dest: ({dst.row},{dst.col})\n")
                f.write(f"  Path: {path_str}{detour_info}\n")
                f.write(f"  Directions: {' '.join(directions_used)}\n")
                f.write("\n")

            # Summary: group by source node
            f.write("\n" + "=" * 80 + "\n")
            f.write("Summary by Source Node\n")
            f.write("=" * 80 + "\n\n")

            source_groups = {}
            for cid in routing_result.get_all_commodity_ids():
                commodity = routing_result.get_commodity(cid)
                src = (commodity.source.row, commodity.source.col)
                if src not in source_groups:
                    source_groups[src] = []
                source_groups[src].append(cid)

            for src, cids in sorted(source_groups.items()):
                f.write(f"Source ({src[0]},{src[1]}):\n")
                for cid in cids:
                    commodity = routing_result.get_commodity(cid)
                    path = routing_result.get_path(cid)
                    dst = commodity.destination
                    directions = []
                    for i in range(len(path) - 1):
                        curr, next_ = path[i], path[i + 1]
                        if next_.row < curr.row:
                            directions.append('N')
                        elif next_.row > curr.row:
                            directions.append('S')
                        elif next_.col > curr.col:
                            directions.append('E')
                        elif next_.col < curr.col:
                            directions.append('W')
                    f.write(f"  -> ({dst.row},{dst.col}): {' '.join(directions)}\n")
                f.write("\n")

        print(f"[DEBUG] Routing results dumped to: {output_file}")

    @staticmethod
    def _compute_xy_path(src: Coord, dst: Coord) -> List[Coord]:
        """Compute XY routing path (X/col first, then Y/row)."""
        path = [src]
        r, c = src.row, src.col
        dr, dc = dst.row, dst.col

        # X (column) first
        while c != dc:
            c += 1 if c < dc else -1
            path.append(Coord(r, c))

        # Y (row) second
        while r != dr:
            r += 1 if r < dr else -1
            path.append(Coord(r, c))

        return path

    def _noc_paths_to_commodities(self, noc_paths: Dict) -> List[Commodity]:
        """Convert NoCPaths to list of Commodity objects."""
        commodities = []

        for idx, (edge, mapping_info) in enumerate(noc_paths.items()):
            src_node = mapping_info[0]
            dst_node = mapping_info[1]

            src_coord_tuple = NodeID.to_coord(src_node)
            dst_coord_tuple = NodeID.to_coord(dst_node)

            src_coord = Coord(src_coord_tuple[0], src_coord_tuple[1])
            dst_coord = Coord(dst_coord_tuple[0], dst_coord_tuple[1])

            if src_coord == dst_coord:
                continue

            commodity = Commodity(
                id=idx,
                source=src_coord,
                destination=dst_coord,
                metadata=(edge, mapping_info)
            )
            commodities.append(commodity)

        return commodities

    @staticmethod
    def _default_tensor_id_extractor(metadata) -> Any:
        """Default tensor_id extractor for multicast grouping."""
        if metadata is None:
            return None
        try:
            edge, mapping_info = metadata
            source_node = mapping_info[0]

            if isinstance(edge, TensorEdge):
                tensor_id = edge.src_id.graph_node_id
            elif hasattr(edge, 'name'):
                tensor_id = f"instruction_{edge.name}"
            else:
                tensor_id = None

            return (source_node, tensor_id)
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _zero_entry() -> Dict:
        """Create a zero policy entry."""
        return {
            "Local": {"enable": False, "chunk_index": 0, "addr": 0, "ksel": 0},
            "North": {"enable": False, "addr": 0},
            "East": {"enable": False, "addr": 0},
            "South": {"enable": False, "addr": 0},
            "West": {"enable": False, "addr": 0},
        }


class PolicyTableGenerator:
    """
    High-level Policy Table Generator.

    This is the main class to use for generating policy tables.
    It creates a routing pipeline with the specified router.

    Usage:
        # Default (MCF Router)
        generator = PolicyTableGenerator(NoCPaths)
        generator.run(mod)

        # With custom router
        generator = PolicyTableGenerator(NoCPaths, router=my_router)
        generator.run(mod)
    """

    def __init__(
        self,
        NoCPaths: Dict,
        router: Optional[BaseRouter] = None,
        mesh_rows: int = 4,
        mesh_cols: int = 5,  # 4 rows x 5 cols: inode at col 0, imce at cols 1-4
        table_capacity: int = 32,
        minimize_congestion: bool = True,
    ):
        """Initialize PolicyTableGenerator.

        Args:
            NoCPaths: Dictionary mapping function names to their NoCPaths
            router: Optional router instance. If None, uses MCFRouter.
            mesh_rows: Mesh topology rows
            mesh_cols: Mesh topology columns
            table_capacity: Max entries per node
            minimize_congestion: MCF optimization objective (if using MCF)
        """
        self.NoCPaths = NoCPaths

        # Create router if not provided
        if router is None:
            topology = MeshTopology(mesh_rows, mesh_cols)
            router = MCFRouterAdapter(topology, minimize_congestion=minimize_congestion)

        # Create pipeline
        config = PipelineConfig(
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            table_capacity=table_capacity,
        )
        self._pipeline = RoutingPipeline(router=router, config=config)

    def run(self, mod):
        """Run policy table generation.

        Args:
            mod: TVM module containing imcflow functions

        Returns:
            The module (unchanged)
        """
        result = self._pipeline.run(mod, self.NoCPaths)
        self.PolicyTable_2D = self._pipeline.PolicyTable_2D
        return result


# Factory functions for creating routers
def create_mcf_router(
    mesh_rows: int = 4,
    mesh_cols: int = 5,  # 4 rows x 5 cols: inode at col 0, imce at cols 1-4
    minimize_congestion: bool = True,
) -> BaseRouter:
    """Create an MCF router instance."""
    topology = MeshTopology(mesh_rows, mesh_cols)
    return MCFRouterAdapter(topology, minimize_congestion=minimize_congestion)


# Convenience function for backward compatibility
def create_policy_table_generator(
    noc_paths: Dict,
    router_type: str = "mcf",
    **kwargs
) -> PolicyTableGenerator:
    """Create a PolicyTableGenerator with the specified router type.

    Args:
        noc_paths: NoCPaths dictionary
        router_type: Router type ("mcf" or "xy")
        **kwargs: Additional arguments for the generator

    Returns:
        PolicyTableGenerator instance
    """
    mesh_rows = kwargs.pop('mesh_rows', 4)
    mesh_cols = kwargs.pop('mesh_cols', 5)

    if router_type == "mcf":
        return PolicyTableGenerator(noc_paths, router=None, mesh_rows=mesh_rows, mesh_cols=mesh_cols, **kwargs)
    elif router_type == "xy":
        router = create_xy_router(mesh_rows=mesh_rows, mesh_cols=mesh_cols)
        return PolicyTableGenerator(noc_paths, router=router, mesh_rows=mesh_rows, mesh_cols=mesh_cols, **kwargs)
    else:
        raise ValueError(f"Unknown router type: {router_type}. Supported: 'mcf', 'xy'")
