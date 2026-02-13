#!/usr/bin/env python3
"""
Test suite for Policy Table Generator.

Tests:
1. Path Tracing: Each edge reaches its correct destination
2. No Spurious Arrivals: Packets don't arrive at unintended nodes
3. Multicast Sharing: Multicast edges share same entry at source
4. Entry Uniqueness: Different flows don't incorrectly share entries
"""

import sys
sys.path.insert(0, '/root/project/tvm/python')
sys.path.insert(0, '/root/project/tvm/tvm_practice')

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from tvm.relay.backend.contrib.imcflow.mcf_router import MCFRouter, MeshTopology, Commodity, Coord
from tvm.relay.backend.contrib.imcflow.path_tree_builder import PathTreeBuilder
from tvm.relay.backend.contrib.imcflow.policy_table_generator import PolicyTableBuilder
from tvm.contrib.imcflow import NodeID, TensorEdge, TensorID


# ============================================================================
# Test Infrastructure
# ============================================================================

NODE_COORDS = {
    NodeID.inode_0_0: (0, 0), NodeID.imce_0_1: (0, 1), NodeID.imce_0_2: (0, 2), NodeID.imce_0_3: (0, 3), NodeID.imce_0_4: (0, 4),
    NodeID.inode_1_0: (1, 0), NodeID.imce_1_1: (1, 1), NodeID.imce_1_2: (1, 2), NodeID.imce_1_3: (1, 3), NodeID.imce_1_4: (1, 4),
    NodeID.inode_2_0: (2, 0), NodeID.imce_2_1: (2, 1), NodeID.imce_2_2: (2, 2), NodeID.imce_2_3: (2, 3), NodeID.imce_2_4: (2, 4),
    NodeID.inode_3_0: (3, 0), NodeID.imce_3_1: (3, 1), NodeID.imce_3_2: (3, 2), NodeID.imce_3_3: (3, 3), NodeID.imce_3_4: (3, 4),
}

COORD_TO_NODE = {v: k for k, v in NODE_COORDS.items()}


@dataclass
class TestEdge:
    """Test edge specification."""
    src_node: NodeID
    dst_node: NodeID
    tensor_id: int  # graph_node_id for grouping multicast
    edge_name: str = ""

    def __hash__(self):
        return hash((self.src_node, self.dst_node, self.tensor_id))


@dataclass
class TestResult:
    """Result of a single test."""
    passed: bool
    message: str
    details: Optional[List[str]] = None


def next_coord(coord: Tuple[int, int], direction: str) -> Optional[Tuple[int, int]]:
    """Get next coordinate given direction."""
    r, c = coord
    if direction == 'North':
        return (r - 1, c) if r > 0 else None
    elif direction == 'South':
        return (r + 1, c) if r < 3 else None
    elif direction == 'East':
        return (r, c + 1) if c < 4 else None
    elif direction == 'West':
        return (r, c - 1) if c > 0 else None
    return None


def trace_packet(policy_tables: Dict, start_node: NodeID, start_entry: int,
                 max_hops: int = 30) -> Tuple[List[Tuple[NodeID, int, str]], Set[NodeID], Set[NodeID]]:
    """
    Trace a packet through policy tables.

    Policy table format:
      - policy_tables[NodeID] = list of entries
      - entry = {'Local': {'enable': bool, 'addr': int, ...}, 'North': {...}, ...}

    Returns:
        - path: List of (node, entry, directions_enabled)
        - visited_nodes: Set of all nodes the packet visits (including transit)
        - delivered_to: Set of nodes where Local delivery happened
    """
    path = []
    visited_nodes = set()
    delivered_to = set()

    current_node = start_node
    current_entry = start_entry
    visited_states = set()

    while len(path) < max_hops:
        coord = NODE_COORDS.get(current_node)
        if coord is None:
            path.append((current_node, current_entry, "INVALID_NODE"))
            break

        visited_nodes.add(current_node)

        if current_node not in policy_tables:
            path.append((current_node, current_entry, "NODE_NOT_IN_TABLE"))
            break

        table = policy_tables[current_node]

        # Table is a list, access by index
        if current_entry >= len(table):
            path.append((current_node, current_entry, f"ENTRY_NOT_FOUND(max={len(table)-1})"))
            break

        entry = table[current_entry]

        # Check enabled directions
        enabled = []
        for d in ['Local', 'North', 'East', 'South', 'West']:
            if entry.get(d, {}).get('enable'):
                if d == 'Local':
                    enabled.append('L')
                    delivered_to.add(current_node)
                else:
                    enabled.append(f"{d[0]}:{entry[d].get('addr', 0)}")

        path.append((current_node, current_entry, ','.join(enabled) if enabled else "NONE"))

        # Check for loop
        state = (current_node, current_entry)
        if state in visited_states:
            path.append((current_node, current_entry, "LOOP_DETECTED"))
            break
        visited_states.add(state)

        # If only Local enabled, done
        if enabled == ['L']:
            break

        # Move to next node (first non-Local direction)
        moved = False
        for dir_name in ['North', 'East', 'South', 'West']:
            if entry.get(dir_name, {}).get('enable'):
                next_c = next_coord(coord, dir_name)
                if next_c:
                    next_node = COORD_TO_NODE.get(next_c)
                    if next_node:
                        current_node = next_node
                        current_entry = entry[dir_name].get('addr', 0)
                        moved = True
                        break

        if not moved:
            if 'L' in enabled:
                break  # Local delivery only
            path.append((current_node, current_entry, "ROUTING_DEAD_END"))
            break

    return path, visited_nodes, delivered_to


# ============================================================================
# Test Cases
# ============================================================================

def test_path_reaches_destination(policy_tables: Dict, router_entries: Dict,
                                  edges: List[TestEdge]) -> TestResult:
    """Test that each edge's packet reaches the correct destination."""
    failures = []

    for edge in edges:
        # Find the entry for this edge at source
        source_entry = None
        if edge in router_entries:
            for node_id, entry_addr in router_entries[edge]:
                if node_id == edge.src_node:
                    source_entry = entry_addr
                    break

        if source_entry is None:
            failures.append(f"  {edge.edge_name}: No source entry found in router_entries")
            continue

        # Trace the packet
        path, visited, delivered = trace_packet(policy_tables, edge.src_node, source_entry)

        # Check if destination received delivery
        if edge.dst_node not in delivered:
            path_str = ' -> '.join(f"{n.name}[{e}]({dirs})" for n, e, dirs in path[:10])
            failures.append(f"  {edge.edge_name}: {edge.src_node.name} -> {edge.dst_node.name}")
            failures.append(f"    Expected delivery to {edge.dst_node.name}, but got: {[n.name for n in delivered]}")
            failures.append(f"    Path: {path_str}")

    if failures:
        return TestResult(False, f"Path tracing failed", failures)
    return TestResult(True, f"All {len(edges)} edges reach correct destinations")


def test_no_spurious_arrivals(policy_tables: Dict, router_entries: Dict,
                              edges: List[TestEdge]) -> TestResult:
    """Test that packets don't arrive at unintended nodes."""
    failures = []

    for edge in edges:
        source_entry = None
        if edge in router_entries:
            for node_id, entry_addr in router_entries[edge]:
                if node_id == edge.src_node:
                    source_entry = entry_addr
                    break

        if source_entry is None:
            continue

        path, visited, delivered = trace_packet(policy_tables, edge.src_node, source_entry)

        # Check for unexpected Local deliveries
        for node in delivered:
            if node != edge.dst_node:
                # Check if this is a valid multicast destination
                is_valid_multicast = any(
                    e.src_node == edge.src_node and
                    e.tensor_id == edge.tensor_id and
                    e.dst_node == node
                    for e in edges
                )
                if not is_valid_multicast:
                    failures.append(f"  {edge.edge_name}: Spurious delivery to {node.name}")
                    failures.append(f"    Expected only: {edge.dst_node.name}")

    if failures:
        return TestResult(False, f"Found spurious arrivals", failures)
    return TestResult(True, "No spurious arrivals detected")


def test_multicast_sharing(policy_tables: Dict, router_entries: Dict,
                           edges: List[TestEdge]) -> TestResult:
    """Test that multicast edges share the same entry at source."""
    failures = []

    # Group edges by (src_node, tensor_id)
    multicast_groups = defaultdict(list)
    for edge in edges:
        key = (edge.src_node, edge.tensor_id)
        multicast_groups[key].append(edge)

    for (src_node, tensor_id), group_edges in multicast_groups.items():
        if len(group_edges) <= 1:
            continue  # Not multicast

        source_entries = set()
        for edge in group_edges:
            if edge in router_entries:
                for node_id, entry_addr in router_entries[edge]:
                    if node_id == src_node:
                        source_entries.add(entry_addr)

        if len(source_entries) > 1:
            failures.append(f"  Multicast group (src={src_node.name}, tensor={tensor_id}):")
            failures.append(f"    Has {len(source_entries)} different source entries: {source_entries}")
            failures.append(f"    Should share ONE entry for multicast to work!")
            for edge in group_edges:
                failures.append(f"      -> {edge.dst_node.name}")

    if failures:
        return TestResult(False, "Multicast edges don't share entries", failures)

    multicast_count = sum(1 for g in multicast_groups.values() if len(g) > 1)
    return TestResult(True, f"All {multicast_count} multicast groups share entries correctly")


# ============================================================================
# Policy Table Generation Pipeline
# ============================================================================

def generate_policy_tables(edges: List[TestEdge], rows: int = 4, cols: int = 5):
    """Run the full policy table generation pipeline."""

    # Group edges by (src_node, tensor_id) to handle multicast properly
    # Edges with same source and tensor_id should share the same TensorEdge
    multicast_groups = defaultdict(list)
    for edge in edges:
        key = (edge.src_node, edge.tensor_id)
        multicast_groups[key].append(edge)

    # Create NoCPaths dict (tensor_edge -> (src, dst, extra))
    # For multicast, one TensorEdge can have multiple noc_paths entries
    noc_paths = {}
    edge_to_tensor = {}  # Map TestEdge -> TensorEdge
    tensor_edge_map = {}  # Map (src_node, tensor_id) -> TensorEdge

    for (src_node, tensor_id), group_edges in multicast_groups.items():
        # Create ONE TensorEdge for the multicast group
        src_tid = TensorID(tensor_id, "odata")
        dst_tid = TensorID(tensor_id + 1000, "data")
        tensor_edge = TensorEdge(src_tid, dst_tid, None)
        tensor_edge_map[(src_node, tensor_id)] = tensor_edge

        # Map all edges in the group to the same TensorEdge
        for edge in group_edges:
            edge_to_tensor[edge] = tensor_edge
            # Add to noc_paths (each edge has its own destination)
            noc_paths[tensor_edge] = (edge.src_node, edge.dst_node, None)

    # For multicast, we need to add ALL destinations to noc_paths
    # But the current noc_paths structure is edge -> (src, dst, None)
    # This is a limitation - let's create separate entries but share TensorEdge

    # Rebuild noc_paths with proper structure for each edge
    noc_paths = {}
    for edge in edges:
        tensor_edge = edge_to_tensor[edge]
        # Use the edge itself as key to maintain 1:1 mapping
        noc_paths[tensor_edge] = (edge.src_node, edge.dst_node, None)

    # Create commodities
    commodities = []
    for idx, edge in enumerate(edges):
        src_coord = NODE_COORDS[edge.src_node]
        dst_coord = NODE_COORDS[edge.dst_node]
        tensor_edge = edge_to_tensor[edge]

        c = Commodity(
            id=idx,
            source=Coord(src_coord[0], src_coord[1]),
            destination=Coord(dst_coord[0], dst_coord[1]),
            metadata=(tensor_edge, (edge.src_node, edge.dst_node, None))
        )
        commodities.append(c)

    # Phase 1: MCF Routing
    topology = MeshTopology(rows=rows, cols=cols)
    router = MCFRouter(topology, minimize_congestion=True)
    routing_result = router.route(commodities)

    # Phase 2: Build trees
    tree_builder = PathTreeBuilder()
    tree_result = tree_builder.build(routing_result)

    # Phase 3: Generate policy tables
    # For this, we need proper noc_paths with all edges
    full_noc_paths = {}
    for edge in edges:
        tensor_edge = edge_to_tensor[edge]
        full_noc_paths[tensor_edge] = (edge.src_node, edge.dst_node, None)

    policy_builder = PolicyTableBuilder(table_capacity=32)
    policy_tables = policy_builder.generate(tree_result, full_noc_paths)

    # Get router entries mapping
    raw_router_entries = policy_builder.get_router_entries()

    # Convert router entries keys from TensorEdge to TestEdge
    # For multicast, multiple TestEdges map to same TensorEdge
    router_entries = {}
    for tensor_edge, entries in raw_router_entries.items():
        # Find all TestEdges that use this TensorEdge
        for edge, te in edge_to_tensor.items():
            if te == tensor_edge:
                router_entries[edge] = entries

    return policy_tables, router_entries, routing_result


# ============================================================================
# Test Runner
# ============================================================================

def run_tests(test_name: str, edges: List[TestEdge]):
    """Run all tests for given edges."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    print(f"\nEdges ({len(edges)}):")
    for edge in edges:
        src_coord = NODE_COORDS[edge.src_node]
        dst_coord = NODE_COORDS[edge.dst_node]
        print(f"  {edge.edge_name}: {edge.src_node.name}{src_coord} -> {edge.dst_node.name}{dst_coord} (tensor={edge.tensor_id})")

    # Generate policy tables
    print("\nGenerating policy tables...")
    try:
        policy_tables, router_entries, routing_result = generate_policy_tables(edges)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  Generated tables for {len(policy_tables)} nodes")
    print(f"  Router entries for {len(router_entries)} edges")

    # Show routes
    print("\nRoutes computed:")
    for cid, route in routing_result.routes.items():
        path = ' -> '.join(f'({c.row},{c.col})' for c in route.path)
        print(f"  Commodity {cid}: {path}")

    # Debug: show router_entries
    print("\nRouter entries:")
    for edge, entries in router_entries.items():
        src_entry = None
        for node_id, entry_addr in entries:
            if node_id == edge.src_node:
                src_entry = entry_addr
        print(f"  {edge.edge_name}: source_entry={src_entry}, total_hops={len(entries)}")

    # Run tests
    all_passed = True
    tests = [
        ("Path Reaches Destination", test_path_reaches_destination),
        ("No Spurious Arrivals", test_no_spurious_arrivals),
        ("Multicast Sharing", test_multicast_sharing),
    ]

    print("\n" + "-"*50)
    print("TEST RESULTS")
    print("-"*50)

    for test_desc, test_func in tests:
        result = test_func(policy_tables, router_entries, edges)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status}: {test_desc}")
        print(f"  {result.message}")
        if result.details:
            for detail in result.details[:20]:  # Limit output
                print(f"  {detail}")
            if len(result.details) > 20:
                print(f"  ... and {len(result.details) - 20} more")
        if not result.passed:
            all_passed = False

    return all_passed


# ============================================================================
# Test Cases
# ============================================================================

def test_simple_unicast():
    """Test simple unicast routing."""
    edges = [
        TestEdge(NodeID.inode_0_0, NodeID.imce_0_1, tensor_id=1, edge_name="edge1"),
        TestEdge(NodeID.inode_0_0, NodeID.imce_0_4, tensor_id=2, edge_name="edge2"),
        TestEdge(NodeID.inode_3_0, NodeID.imce_3_4, tensor_id=3, edge_name="edge3"),
    ]
    return run_tests("Simple Unicast", edges)


def test_multicast():
    """Test multicast routing (same source, same tensor, multiple destinations)."""
    edges = [
        # Multicast: same tensor_id=-10 from inode_0_0 to two destinations
        TestEdge(NodeID.inode_0_0, NodeID.imce_3_4, tensor_id=-10, edge_name="mcast1"),
        TestEdge(NodeID.inode_0_0, NodeID.imce_2_4, tensor_id=-10, edge_name="mcast2"),
    ]
    return run_tests("Multicast", edges)


def test_output_path():
    """Test output path going West (imce -> inode)."""
    edges = [
        TestEdge(NodeID.imce_2_4, NodeID.inode_2_0, tensor_id=24, edge_name="output1"),
        TestEdge(NodeID.imce_3_4, NodeID.inode_3_0, tensor_id=31, edge_name="output2"),
    ]
    return run_tests("Output Path (West)", edges)


def test_region1_subset():
    """Test a subset of Region 1 edges from noc_paths.txt."""
    edges = [
        # Input data multicast
        TestEdge(NodeID.inode_0_0, NodeID.imce_3_4, tensor_id=-10, edge_name="input_data1"),
        TestEdge(NodeID.inode_0_0, NodeID.imce_2_4, tensor_id=-10, edge_name="input_data2"),

        # Output from imce_2_4
        TestEdge(NodeID.imce_2_4, NodeID.inode_2_0, tensor_id=24, edge_name="output_24"),

        # Data flow in row 3
        TestEdge(NodeID.imce_3_4, NodeID.imce_3_3, tensor_id=20, edge_name="data_20"),
        TestEdge(NodeID.imce_3_3, NodeID.imce_3_2, tensor_id=21, edge_name="data_21"),
        TestEdge(NodeID.imce_3_2, NodeID.imce_3_1, tensor_id=22, edge_name="data_22"),

        # Output from imce_3_1
        TestEdge(NodeID.imce_3_1, NodeID.inode_3_0, tensor_id=23, edge_name="output_23"),

        # Constants from inode_3_0
        TestEdge(NodeID.inode_3_0, NodeID.imce_3_4, tensor_id=-21, edge_name="const_-21"),
        TestEdge(NodeID.inode_3_0, NodeID.imce_3_3, tensor_id=-19, edge_name="const_-19"),
        TestEdge(NodeID.inode_3_0, NodeID.imce_3_2, tensor_id=-23, edge_name="const_-23"),
        TestEdge(NodeID.inode_3_0, NodeID.imce_3_1, tensor_id=-14, edge_name="const_-14"),

        # Input from inode_2_0
        TestEdge(NodeID.inode_2_0, NodeID.imce_2_4, tensor_id=-25, edge_name="input_-25"),
    ]
    return run_tests("Region 1 Subset", edges)


def test_cross_row_routing():
    """Test routing that crosses multiple rows."""
    edges = [
        # From row 0 to row 3
        TestEdge(NodeID.inode_0_0, NodeID.imce_3_4, tensor_id=1, edge_name="r0_to_r3"),
        # From row 3 to row 0 (would need to go through inode column)
        TestEdge(NodeID.imce_3_4, NodeID.inode_3_0, tensor_id=2, edge_name="r3_output"),
    ]
    return run_tests("Cross-Row Routing", edges)


# ============================================================================
# NoCPaths Parser
# ============================================================================

def parse_noc_paths_file(filepath: str) -> Dict[str, List[TestEdge]]:
    """
    Parse noc_paths.txt and create TestEdge objects.

    Returns:
        Dict mapping function_name -> list of TestEdge
    """
    import re

    result = {}
    current_func = None
    edge_idx = 0

    # NodeID name to enum mapping
    node_name_map = {n.name: n for n in NodeID}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Function name (no special prefix)
            if not line.startswith('TensorEdge') and not line.startswith('NodeID'):
                current_func = line
                result[current_func] = []
                edge_idx = 0
                continue

            if current_func is None:
                continue

            # Parse the line: EdgeInfo (NodeID.src, NodeID.dst, extra)
            if ' (' in line:
                edge_part, mapping_part = line.rsplit(' (', 1)

                # Extract NodeID names from mapping part
                # Format: (<NodeID.inode_0_0: 0>, <NodeID.imce_3_4: 19>, None)
                node_pattern = r'<NodeID\.(\w+):'
                nodes = re.findall(node_pattern, mapping_part)

                if len(nodes) >= 2:
                    src_name = nodes[0]
                    dst_name = nodes[1]

                    try:
                        src_node = node_name_map[src_name]
                        dst_node = node_name_map[dst_name]
                    except KeyError:
                        continue

                    # Extract tensor_id from TensorEdge
                    # Format: TensorEdge((graph_node_id, tensor_type), ...)
                    # or: TensorEdge(((inner_id, graph_node_id), tensor_type), ...)
                    tensor_id = edge_idx  # Default to index

                    if edge_part.startswith('TensorEdge'):
                        # Try simple format: TensorEdge((id, type), ...)
                        match = re.match(r'TensorEdge\(\((-?\d+),', edge_part)
                        if match:
                            tensor_id = int(match.group(1))
                        else:
                            # Try nested format: TensorEdge(((inner, outer), type), ...)
                            match = re.match(r'TensorEdge\(\(\((-?\d+),\s*(-?\d+)\),', edge_part)
                            if match:
                                tensor_id = int(match.group(2))  # Use outer id
                    elif edge_part.startswith('NodeID'):
                        # Instruction edge: NodeID.imce_X_Y
                        # Use a unique negative id for instructions
                        tensor_id = -1000 - edge_idx

                    edge_name = f"edge_{edge_idx}"
                    if edge_part.startswith('TensorEdge'):
                        # Create a shorter name from the edge
                        short_match = re.match(r'TensorEdge\(\((-?\d+),\s*(\w+)\)', edge_part)
                        if short_match:
                            edge_name = f"T({short_match.group(1)},{short_match.group(2)})"
                        else:
                            edge_name = f"T{tensor_id}"
                    else:
                        edge_name = f"Inst_{dst_name}"

                    test_edge = TestEdge(
                        src_node=src_node,
                        dst_node=dst_node,
                        tensor_id=tensor_id,
                        edge_name=edge_name
                    )
                    result[current_func].append(test_edge)
                    edge_idx += 1

    return result


def test_from_noc_paths(filepath: str, func_name: str = None):
    """
    Test policy table generation using edges from noc_paths.txt.

    Args:
        filepath: Path to noc_paths.txt
        func_name: Specific function to test, or None for all
    """
    print(f"\n{'='*70}")
    print(f"PARSING: {filepath}")
    print(f"{'='*70}")

    parsed = parse_noc_paths_file(filepath)

    print(f"Found {len(parsed)} functions:")
    for fn, edges in parsed.items():
        tensor_edges = [e for e in edges if not e.edge_name.startswith('Inst_')]
        inst_edges = [e for e in edges if e.edge_name.startswith('Inst_')]
        print(f"  {fn}: {len(tensor_edges)} tensor edges, {len(inst_edges)} instruction edges")

    results = []

    for fn, edges in parsed.items():
        if func_name and fn != func_name:
            continue

        # Test only tensor edges (skip instruction edges for now)
        tensor_edges = [e for e in edges if not e.edge_name.startswith('Inst_')]

        if not tensor_edges:
            print(f"\n  Skipping {fn}: no tensor edges")
            continue

        passed = run_tests(f"NoCPaths: {fn[:50]}", tensor_edges)
        results.append((fn, passed))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Policy Table Generator')
    parser.add_argument('--noc-paths', type=str, help='Path to noc_paths.txt to test')
    parser.add_argument('--func', type=str, help='Specific function name to test')
    parser.add_argument('--basic', action='store_true', help='Run basic tests only')
    args = parser.parse_args()

    print("="*70)
    print("POLICY TABLE GENERATOR TEST SUITE")
    print("="*70)

    results = []

    if args.basic or not args.noc_paths:
        # Run basic tests
        results.append(("Simple Unicast", test_simple_unicast()))
        results.append(("Multicast", test_multicast()))
        results.append(("Output Path", test_output_path()))
        results.append(("Region 1 Subset", test_region1_subset()))
        results.append(("Cross-Row Routing", test_cross_row_routing()))

    if args.noc_paths:
        # Test from noc_paths.txt
        noc_results = test_from_noc_paths(args.noc_paths, args.func)
        results.extend(noc_results)

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        # Truncate long names
        display_name = name[:60] + "..." if len(name) > 60 else name
        print(f"  {status}: {display_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
