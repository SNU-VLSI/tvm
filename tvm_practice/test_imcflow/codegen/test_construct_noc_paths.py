"""
Unit tests for construct_noc_paths_from_pnr_results function.

Tests the conversion from JointPnRResult to NoCPaths dict.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum, auto

# Mock NodeID for testing without full imcflow import
class MockNodeID(Enum):
    inode_0_0 = auto()
    inode_1_0 = auto()
    inode_2_0 = auto()
    inode_3_0 = auto()
    imce_0_1 = auto()
    imce_0_2 = auto()
    imce_1_1 = auto()
    imce_1_2 = auto()
    imce_2_1 = auto()
    imce_3_1 = auto()

    @staticmethod
    def to_coord(node_id):
        """Convert NodeID to (row, col) tuple"""
        name = node_id.name
        parts = name.split('_')
        return (int(parts[1]), int(parts[2]))

    @staticmethod
    def from_coord(row, col):
        """Convert (row, col) to NodeID"""
        if col == 0:
            return getattr(MockNodeID, f"inode_{row}_0")
        else:
            return getattr(MockNodeID, f"imce_{row}_{col}")


@dataclass(frozen=True)
class Coord:
    row: int
    col: int


@dataclass(frozen=True)
class TensorID:
    graph_node_id: Any
    tensor_type: str


@dataclass(frozen=True)
class TensorEdge:
    src_id: TensorID
    dst_id: TensorID
    split_idx: Optional[int] = None


class NodeType(Enum):
    CALL = auto()
    VAR = auto()
    CONST = auto()
    FUNC_OUT = auto()
    SPLIT = auto()
    CONCAT = auto()


@dataclass
class Commodity:
    id: int
    source_node_id: Any
    dest_node_id: Any
    source_type: NodeType
    dest_type: NodeType
    tensor_type: str
    split_idx: Optional[int] = None
    metadata: Any = None


@dataclass
class Edge:
    src: Coord
    dst: Coord


@dataclass
class JointPnRResult:
    mapping: Dict[Any, Coord]
    routes: Dict[int, List[Edge]]
    commodities: List[Commodity]
    max_congestion: int
    total_hops: int
    solver_status: str
    success: bool = True
    var_to_inode: Dict[Any, Coord] = None
    funcout_to_inode: Dict[Any, Coord] = None
    const_to_inode: Dict[Any, Coord] = None
    tensor_edge_to_commodity_id: Dict[Any, int] = None


def getOuterNodeID(node):
    if isinstance(node, tuple):
        return node[0]
    return node


def coord_to_node_id(coord: Coord):
    """Convert Coord to MockNodeID"""
    return MockNodeID.from_coord(coord.row, coord.col)


def construct_noc_paths_from_pnr_results_original(
    pnr_result: JointPnRResult,
    tensor_edge_list: List[TensorEdge],
    hw_node_map: Dict,
) -> Dict:
    """
    Original implementation (with bug).
    """
    noc_paths = {}

    for tensor_edge in tensor_edge_list:
        src_tensor_id = tensor_edge.src_id
        dst_tensor_id = tensor_edge.dst_id
        split_idx = tensor_edge.split_idx

        src_gid = src_tensor_id.graph_node_id
        dst_gid = dst_tensor_id.graph_node_id

        # Get graph node IDs (outer ID for composites)
        src_graph_id = getOuterNodeID(src_gid)
        dst_graph_id = getOuterNodeID(dst_gid)

        # Look up in mapping
        src_coord = pnr_result.mapping.get(src_graph_id)
        dst_coord = pnr_result.mapping.get(dst_graph_id)

        if src_coord is None or dst_coord is None:
            # Node not mapped - try HWNodeMap as fallback
            src_hwnode = hw_node_map.get(src_graph_id)
            dst_hwnode = hw_node_map.get(dst_graph_id)

            if src_hwnode is not None and dst_hwnode is not None:
                if isinstance(dst_hwnode, tuple):
                    dst_hwnode = dst_hwnode[split_idx] if split_idx is not None else dst_hwnode[0]
                    split_idx = None
                noc_paths[tensor_edge] = (src_hwnode, dst_hwnode, split_idx)
            continue

        # Convert Coord to NodeID
        src_hwnode = coord_to_node_id(src_coord)
        dst_hwnode = coord_to_node_id(dst_coord)

        noc_paths[tensor_edge] = (src_hwnode, dst_hwnode, split_idx)

    return noc_paths


def construct_noc_paths_from_pnr_results_fixed(
    pnr_result: JointPnRResult,
    tensor_edge_list: List[TensorEdge],
    hw_node_map: Dict,
) -> Dict:
    """
    Fixed implementation.

    Key fixes:
    - For VAR sources, use var_to_inode instead of mapping
    - For FUNC_OUT destinations, use funcout_to_inode instead of mapping
    - For CONST sources, use const_to_inode instead of mapping
    """
    noc_paths = {}

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

        src_gid = src_tensor_id.graph_node_id
        dst_gid = dst_tensor_id.graph_node_id

        # Get graph node IDs (outer ID for composites)
        src_graph_id = getOuterNodeID(src_gid)
        dst_graph_id = getOuterNodeID(dst_gid)

        # Check node types from commodities
        src_commodity = commodity_by_src.get(src_graph_id)
        dst_commodity = commodity_by_dest.get(dst_graph_id)

        src_is_var = (src_commodity is not None and src_commodity.source_type == NodeType.VAR)
        src_is_const = (src_commodity is not None and src_commodity.source_type == NodeType.CONST)
        dst_is_funcout = (dst_commodity is not None and dst_commodity.dest_type == NodeType.FUNC_OUT)

        # Look up source coordinate
        if src_is_var:
            src_coord = pnr_result.var_to_inode.get(src_graph_id) if pnr_result.var_to_inode else None
        elif src_is_const:
            src_coord = pnr_result.const_to_inode.get(src_graph_id) if pnr_result.const_to_inode else None
        else:
            src_coord = pnr_result.mapping.get(src_graph_id)

        # Look up destination coordinate
        if dst_is_funcout:
            dst_coord = pnr_result.funcout_to_inode.get(dst_graph_id) if pnr_result.funcout_to_inode else None
        else:
            dst_coord = pnr_result.mapping.get(dst_graph_id)

        if src_coord is None or dst_coord is None:
            # Node not mapped - try HWNodeMap as fallback
            src_hwnode = hw_node_map.get(src_graph_id)
            dst_hwnode = hw_node_map.get(dst_graph_id)

            if src_hwnode is not None and dst_hwnode is not None:
                if isinstance(dst_hwnode, tuple):
                    dst_hwnode = dst_hwnode[split_idx] if split_idx is not None else dst_hwnode[0]
                    split_idx = None
                noc_paths[tensor_edge] = (src_hwnode, dst_hwnode, split_idx)
            continue

        # Convert Coord to NodeID
        src_hwnode = coord_to_node_id(src_coord)
        dst_hwnode = coord_to_node_id(dst_coord)

        noc_paths[tensor_edge] = (src_hwnode, dst_hwnode, split_idx)

    return noc_paths


def test_funcout_routing():
    """
    Test case: IMCE (1,1) -> FUNC_OUT -> INODE (3,0)

    Scenario:
    - Node 10 is a CALL mapped to IMCE (1,1)
    - Node 11 is a FUNC_OUT that should route to INODE (3,0)
    - TensorEdge: (10, odata) -> (11, func_out0)

    Expected: noc_path should be (imce_1_1, inode_3_0, None)
    """
    print("=" * 70)
    print("Test: FUNC_OUT routing")
    print("=" * 70)

    # Create tensor edge: node 10 (odata) -> node 11 (func_out0)
    tensor_edge = TensorEdge(
        src_id=TensorID(graph_node_id=10, tensor_type='odata'),
        dst_id=TensorID(graph_node_id=11, tensor_type='func_out0'),
        split_idx=None
    )

    # Create commodity for this edge
    commodity = Commodity(
        id=0,
        source_node_id=10,
        dest_node_id=11,
        source_type=NodeType.CALL,
        dest_type=NodeType.FUNC_OUT,
        tensor_type='odata',
        metadata=tensor_edge
    )

    # Create PnR result
    # - Node 10 (CALL) is mapped to IMCE (1,1)
    # - Node 11 (FUNC_OUT) should go to INODE (3,0)
    pnr_result = JointPnRResult(
        mapping={10: Coord(1, 1)},  # CALL -> IMCE
        routes={
            0: [
                Edge(Coord(1, 1), Coord(1, 0)),
                Edge(Coord(1, 0), Coord(2, 0)),
                Edge(Coord(2, 0), Coord(3, 0)),
            ]
        },
        commodities=[commodity],
        max_congestion=1,
        total_hops=3,
        solver_status="Optimal",
        success=True,
        var_to_inode={},
        funcout_to_inode={11: Coord(3, 0)},  # FUNC_OUT -> INODE
        const_to_inode={},
        tensor_edge_to_commodity_id={tensor_edge: 0},
    )

    tensor_edge_list = [tensor_edge]
    hw_node_map = {}  # Empty - should use pnr_result

    # Test original implementation
    print("\n--- Original Implementation ---")
    noc_paths_original = construct_noc_paths_from_pnr_results_original(
        pnr_result, tensor_edge_list, hw_node_map
    )

    if tensor_edge in noc_paths_original:
        src, dst, split = noc_paths_original[tensor_edge]
        print(f"Result: {src.name} -> {dst.name}, split_idx={split}")
        print(f"Expected: imce_1_1 -> inode_3_0")

        if dst.name == "inode_3_0":
            print("✓ PASS")
        else:
            print("✗ FAIL: Destination should be inode_3_0, got", dst.name)
    else:
        print("✗ FAIL: TensorEdge not in noc_paths")
        print("  This happens because mapping.get(11) returns None (FUNC_OUT not in mapping)")
        print("  And hw_node_map is empty, so fallback also fails")

    # Test fixed implementation
    print("\n--- Fixed Implementation ---")
    noc_paths_fixed = construct_noc_paths_from_pnr_results_fixed(
        pnr_result, tensor_edge_list, hw_node_map
    )

    if tensor_edge in noc_paths_fixed:
        src, dst, split = noc_paths_fixed[tensor_edge]
        print(f"Result: {src.name} -> {dst.name}, split_idx={split}")
        print(f"Expected: imce_1_1 -> inode_3_0")

        if dst.name == "inode_3_0":
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL: Destination should be inode_3_0, got", dst.name)
            return False
    else:
        print("✗ FAIL: TensorEdge not in noc_paths")
        return False


def test_var_to_call():
    """
    Test case: VAR (INODE 0,0) -> CALL (IMCE 1,1)
    """
    print("\n" + "=" * 70)
    print("Test: VAR -> CALL routing")
    print("=" * 70)

    tensor_edge = TensorEdge(
        src_id=TensorID(graph_node_id=5, tensor_type='data'),
        dst_id=TensorID(graph_node_id=10, tensor_type='data'),
        split_idx=None
    )

    commodity = Commodity(
        id=0,
        source_node_id=5,
        dest_node_id=10,
        source_type=NodeType.VAR,
        dest_type=NodeType.CALL,
        tensor_type='data',
        metadata=tensor_edge
    )

    pnr_result = JointPnRResult(
        mapping={10: Coord(1, 1)},
        routes={0: [Edge(Coord(0, 0), Coord(0, 1)), Edge(Coord(0, 1), Coord(1, 1))]},
        commodities=[commodity],
        max_congestion=1,
        total_hops=2,
        solver_status="Optimal",
        success=True,
        var_to_inode={5: Coord(0, 0)},
        funcout_to_inode={},
        const_to_inode={},
        tensor_edge_to_commodity_id={tensor_edge: 0},
    )

    hw_node_map = {5: MockNodeID.inode_0_0}  # VAR is in HWNodeMap
    tensor_edge_list = [tensor_edge]

    noc_paths = construct_noc_paths_from_pnr_results_fixed(
        pnr_result, tensor_edge_list, hw_node_map
    )

    if tensor_edge in noc_paths:
        src, dst, split = noc_paths[tensor_edge]
        print(f"Result: {src.name} -> {dst.name}, split_idx={split}")
        print(f"Expected: inode_0_0 -> imce_1_1")

        if src.name == "inode_0_0" and dst.name == "imce_1_1":
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL")
            return False
    else:
        print("✗ FAIL: TensorEdge not in noc_paths")
        return False


def test_call_to_call():
    """
    Test case: CALL (IMCE 0,1) -> CALL (IMCE 1,1)
    """
    print("\n" + "=" * 70)
    print("Test: CALL -> CALL routing")
    print("=" * 70)

    tensor_edge = TensorEdge(
        src_id=TensorID(graph_node_id=8, tensor_type='odata'),
        dst_id=TensorID(graph_node_id=10, tensor_type='data'),
        split_idx=None
    )

    commodity = Commodity(
        id=0,
        source_node_id=8,
        dest_node_id=10,
        source_type=NodeType.CALL,
        dest_type=NodeType.CALL,
        tensor_type='odata',
        metadata=tensor_edge
    )

    pnr_result = JointPnRResult(
        mapping={8: Coord(0, 1), 10: Coord(1, 1)},
        routes={0: [Edge(Coord(0, 1), Coord(1, 1))]},
        commodities=[commodity],
        max_congestion=1,
        total_hops=1,
        solver_status="Optimal",
        success=True,
        var_to_inode={},
        funcout_to_inode={},
        const_to_inode={},
        tensor_edge_to_commodity_id={tensor_edge: 0},
    )

    hw_node_map = {}
    tensor_edge_list = [tensor_edge]

    noc_paths = construct_noc_paths_from_pnr_results_fixed(
        pnr_result, tensor_edge_list, hw_node_map
    )

    if tensor_edge in noc_paths:
        src, dst, split = noc_paths[tensor_edge]
        print(f"Result: {src.name} -> {dst.name}, split_idx={split}")
        print(f"Expected: imce_0_1 -> imce_1_1")

        if src.name == "imce_0_1" and dst.name == "imce_1_1":
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL")
            return False
    else:
        print("✗ FAIL: TensorEdge not in noc_paths")
        return False


if __name__ == "__main__":
    print("Unit Tests for construct_noc_paths_from_pnr_results")
    print("=" * 70)

    results = []
    results.append(("FUNC_OUT routing", test_funcout_routing()))
    results.append(("VAR -> CALL routing", test_var_to_call()))
    results.append(("CALL -> CALL routing", test_call_to_call()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    sys.exit(0 if all_passed else 1)
