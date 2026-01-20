#!/usr/bin/env python3
"""
NoC Path Analyzer

Analyzes noc_paths.txt files to identify:
1. Nodes receiving multiple data edges (potential conflicts)
2. Multicast sources (one source sending to multiple destinations)

Usage:
    python noc_path_analyzer.py <noc_paths.txt>
    python noc_path_analyzer.py <noc_paths.txt> --function <func_name>
"""

import re
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# Data tensor types (from mcf_router.py)
DATA_TENSOR_TYPES = frozenset([
    'odata', 'data', 'lhs', 'rhs', 'var',
    *[f"func_out{i}" for i in range(30)]
])

# Constant tensor types (for reference)
CONST_TENSOR_TYPES = frozenset([
    'weight', 'config', 'min', 'max', 'fused_scale', 'fused_bias',
    'bias', 'scale', 'threshold'
])


@dataclass
class TensorEdgeInfo:
    """Parsed tensor edge information."""
    src_id: str           # Source tensor ID (e.g., "(15, 12)" or "14")
    src_type: str         # Source tensor type (e.g., "odata")
    dst_id: str           # Destination tensor ID
    dst_type: str         # Destination tensor type (e.g., "data", "lhs")
    src_node: str         # Source NodeID (e.g., "imce_3_3")
    dst_node: str         # Destination NodeID (e.g., "imce_3_1")
    split_idx: Optional[int]  # Split index if present
    raw_line: str         # Original line for reference


@dataclass
class FunctionAnalysis:
    """Analysis result for a single function."""
    func_name: str
    tensor_edges: List[TensorEdgeInfo] = field(default_factory=list)
    inst_edges: List[Tuple[str, str]] = field(default_factory=list)  # (src_node, dst_node)

    # Analysis results
    multi_data_receivers: Dict[str, List[TensorEdgeInfo]] = field(default_factory=dict)
    multicast_sources: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)


def parse_node_id(node_str: str) -> str:
    """Extract node name from NodeID string like '<NodeID.imce_3_3: 18>'."""
    match = re.search(r'NodeID\.(\w+)', node_str)
    if match:
        return match.group(1)
    return node_str


def parse_tensor_id(tensor_str: str) -> Tuple[str, str]:
    """
    Parse tensor ID and type from string like:
    - "((15, 12), odata)" -> ("(15, 12)", "odata")
    - "(14, odata)" -> ("14", "odata")
    - "(43, func_out0)" -> ("43", "func_out0")
    """
    tensor_str = tensor_str.strip()

    # Pattern for nested tuple: ((x, y), type)
    nested_match = re.match(r'\(\(([^)]+)\),\s*(\w+)\)', tensor_str)
    if nested_match:
        return f"({nested_match.group(1)})", nested_match.group(2)

    # Pattern for simple tuple: (id, type)
    simple_match = re.match(r'\(([^,]+),\s*(\w+)\)', tensor_str)
    if simple_match:
        return simple_match.group(1).strip(), simple_match.group(2)

    return tensor_str, "unknown"


def parse_tensor_edge_line(line: str) -> Optional[TensorEdgeInfo]:
    """
    Parse a TensorEdge line like:
    TensorEdge(((15, 12), odata), (16, func_out0)) (<NodeID.imce_3_3: 18>, <NodeID.inode_3_0: 15>, None)
    TensorEdge((14, odata), ((15, 11), data)) (<NodeID.imce_3_4: 19>, <NodeID.imce_3_3: 18>, None)
    """
    if not line.startswith('TensorEdge('):
        return None

    # Split into TensorEdge part and NodeID part
    # Find the matching parenthesis for TensorEdge(...)
    paren_count = 0
    tensor_end = -1
    for i, ch in enumerate(line):
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            paren_count -= 1
            if paren_count == 0:
                tensor_end = i
                break

    if tensor_end == -1:
        return None

    tensor_part = line[:tensor_end + 1]
    node_part = line[tensor_end + 1:].strip()

    # Parse TensorEdge(src, dst) or TensorEdge(src, dst, split_idx)
    # Remove "TensorEdge(" prefix and ")" suffix
    inner = tensor_part[len('TensorEdge('):-1]

    # Find the comma that separates src and dst (handling nested parens)
    paren_count = 0
    comma_positions = []
    for i, ch in enumerate(inner):
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            paren_count -= 1
        elif ch == ',' and paren_count == 0:
            comma_positions.append(i)

    if len(comma_positions) < 1:
        return None

    src_str = inner[:comma_positions[0]].strip()
    if len(comma_positions) >= 2:
        dst_str = inner[comma_positions[0] + 1:comma_positions[1]].strip()
        split_str = inner[comma_positions[1] + 1:].strip()
        try:
            split_idx = int(split_str)
        except ValueError:
            split_idx = None
    else:
        dst_str = inner[comma_positions[0] + 1:].strip()
        split_idx = None

    src_id, src_type = parse_tensor_id(src_str)
    dst_id, dst_type = parse_tensor_id(dst_str)

    # Parse NodeID part: (<NodeID.xxx: n>, <NodeID.yyy: m>, split)
    node_match = re.search(r'\(<NodeID\.(\w+):\s*\d+>,\s*<NodeID\.(\w+):\s*\d+>', node_part)
    if not node_match:
        return None

    src_node = node_match.group(1)
    dst_node = node_match.group(2)

    return TensorEdgeInfo(
        src_id=src_id,
        src_type=src_type,
        dst_id=dst_id,
        dst_type=dst_type,
        src_node=src_node,
        dst_node=dst_node,
        split_idx=split_idx,
        raw_line=line
    )


def parse_inst_edge_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse instruction edge line like:
    NodeID.imce_0_1 (<NodeID.inode_0_0: 0>, <NodeID.imce_0_1: 1>, None)
    """
    if not line.startswith('NodeID.'):
        return None

    node_match = re.search(r'\(<NodeID\.(\w+):\s*\d+>,\s*<NodeID\.(\w+):\s*\d+>', line)
    if node_match:
        return node_match.group(1), node_match.group(2)
    return None


def parse_noc_paths_file(filepath: str) -> Dict[str, FunctionAnalysis]:
    """Parse noc_paths.txt and return analysis for each function."""
    functions: Dict[str, FunctionAnalysis] = {}
    current_func: Optional[FunctionAnalysis] = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is a function name line (no special prefix)
            if not line.startswith('TensorEdge') and not line.startswith('NodeID.'):
                # New function
                current_func = FunctionAnalysis(func_name=line)
                functions[line] = current_func
                continue

            if current_func is None:
                continue

            # Try parsing as TensorEdge
            tensor_edge = parse_tensor_edge_line(line)
            if tensor_edge:
                current_func.tensor_edges.append(tensor_edge)
                continue

            # Try parsing as instruction edge
            inst_edge = parse_inst_edge_line(line)
            if inst_edge:
                current_func.inst_edges.append(inst_edge)

    return functions


def analyze_function(func: FunctionAnalysis) -> None:
    """Perform analysis on a function's edges."""

    # 1. Find nodes receiving multiple DATA edges
    # Group by (dst_node, dst_type) for data types
    incoming_data_edges: Dict[str, List[TensorEdgeInfo]] = defaultdict(list)

    for edge in func.tensor_edges:
        # Only consider data tensor types at destination
        if edge.dst_type in DATA_TENSOR_TYPES:
            incoming_data_edges[edge.dst_node].append(edge)

    # Filter to nodes with multiple data edges
    func.multi_data_receivers = {
        node: edges for node, edges in incoming_data_edges.items()
        if len(edges) > 1
    }

    # 2. Find multicast sources
    # Group by (src_node, src_id, src_type) to find same source sending to multiple destinations
    source_to_dests: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    for edge in func.tensor_edges:
        key = (edge.src_node, edge.src_id, edge.src_type)
        source_to_dests[key].add(edge.dst_node)

    # Filter to sources with multiple destinations (multicast)
    func.multicast_sources = {
        (src_node, f"{src_id}:{src_type}"): dests
        for (src_node, src_id, src_type), dests in source_to_dests.items()
        if len(dests) > 1
    }


def print_analysis(func: FunctionAnalysis, verbose: bool = False) -> None:
    """Print analysis results for a function."""
    print(f"\n{'=' * 70}")
    print(f"Function: {func.func_name}")
    print(f"{'=' * 70}")
    print(f"Total tensor edges: {len(func.tensor_edges)}")
    print(f"Total instruction edges: {len(func.inst_edges)}")

    # Count by type
    data_edges = [e for e in func.tensor_edges if e.dst_type in DATA_TENSOR_TYPES]
    const_edges = [e for e in func.tensor_edges if e.dst_type in CONST_TENSOR_TYPES]
    print(f"  - Data edges (to data/lhs/rhs/odata/var/func_out*): {len(data_edges)}")
    print(f"  - Const edges (to weight/config/min/max/etc.): {len(const_edges)}")

    # 1. Multi-data receivers
    print(f"\n--- Nodes Receiving Multiple DATA Edges ---")
    if func.multi_data_receivers:
        for node, edges in sorted(func.multi_data_receivers.items()):
            print(f"\n  {node}: {len(edges)} incoming data edges")
            for edge in edges:
                print(f"    <- {edge.src_node} ({edge.src_type}) -> ({edge.dst_type})")
                if verbose:
                    print(f"       {edge.raw_line}")
    else:
        print("  None")

    # 2. Multicast sources
    print(f"\n--- Multicast Sources (source_node, {{dst_nodes}}) ---")
    if func.multicast_sources:
        for (src_node, src_info), dst_nodes in sorted(func.multicast_sources.items()):
            print(f"\n  ({src_node}, {src_info}):")
            print(f"    -> {{{', '.join(sorted(dst_nodes))}}}")
    else:
        print("  None")


def print_summary(functions: Dict[str, FunctionAnalysis]) -> None:
    """Print summary across all functions."""
    print(f"\n{'#' * 70}")
    print("SUMMARY")
    print(f"{'#' * 70}")

    total_multi_receivers = 0
    total_multicast = 0

    for func_name, func in functions.items():
        multi_recv = len(func.multi_data_receivers)
        multicast = len(func.multicast_sources)
        total_multi_receivers += multi_recv
        total_multicast += multicast

        if multi_recv > 0 or multicast > 0:
            print(f"\n{func_name}:")
            if multi_recv > 0:
                print(f"  - Multi-data receivers: {multi_recv} nodes")
            if multicast > 0:
                print(f"  - Multicast sources: {multicast}")

    print(f"\nTotal across all functions:")
    print(f"  - Functions analyzed: {len(functions)}")
    print(f"  - Nodes with multiple data inputs: {total_multi_receivers}")
    print(f"  - Multicast sources: {total_multicast}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NoC paths from noc_paths.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python noc_path_analyzer.py noc_paths.txt
    python noc_path_analyzer.py noc_paths.txt --function func_name
    python noc_path_analyzer.py noc_paths.txt --verbose
    python noc_path_analyzer.py noc_paths.txt --summary-only
        """
    )
    parser.add_argument('filepath', help='Path to noc_paths.txt file')
    parser.add_argument('--function', '-f', help='Analyze only specific function')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show raw edge lines')
    parser.add_argument('--summary-only', '-s', action='store_true', help='Show only summary')

    args = parser.parse_args()

    # Parse file
    print(f"Parsing: {args.filepath}")
    functions = parse_noc_paths_file(args.filepath)
    print(f"Found {len(functions)} function(s)")

    # Analyze each function
    for func in functions.values():
        analyze_function(func)

    # Filter if specific function requested
    if args.function:
        if args.function in functions:
            functions = {args.function: functions[args.function]}
        else:
            # Try partial match
            matches = {k: v for k, v in functions.items() if args.function in k}
            if matches:
                functions = matches
            else:
                print(f"Error: Function '{args.function}' not found")
                print(f"Available functions: {list(functions.keys())}")
                sys.exit(1)

    # Print results
    if not args.summary_only:
        for func in functions.values():
            print_analysis(func, verbose=args.verbose)

    print_summary(functions)


if __name__ == '__main__':
    main()
