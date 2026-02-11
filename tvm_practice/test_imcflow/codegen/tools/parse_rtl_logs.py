#!/usr/bin/env python3
"""
RTL Log Parser for IMCFlow Debug

Parses RTL simulation logs to extract node outputs and compare with reference values.
Focuses on:
- VPU MM_QUANT outputs (quantized convolution results)
- PIMC_OUT (post_imcu BPBS results)
- ERF WRITE operations

Usage:
    python parse_rtl_logs.py <test_dir> [options]

Examples:
    python parse_rtl_logs.py ../resnet8_subset13_pretrained_small_evl --compare-imce "3,2:35"
    python parse_rtl_logs.py ../resnet8_subset13_pretrained_small_evl --deep-compare
    python parse_rtl_logs.py ../resnet8_subset13_pretrained_small_evl --imce 3,2
"""

import os
import re
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class IMCECtrlLogEntry:
    """Represents a single IMCE ctrl_pl log entry"""
    timestamp: int
    row: int
    col: int
    log_type: str  # EXECUTE, STALL_START, etc.
    content: str   # Full log line content after timestamp


@dataclass
class MMQuantOutput:
    """Represents a single MM_QUANT operation output"""
    timestamp: int
    input_values: List[int]
    min_val: int
    max_val: int
    output_values: List[int]


@dataclass
class PIMCOutput:
    """Represents a single PIMC_OUT operation output"""
    timestamp: int
    result: List[int]


@dataclass
class ERFWrite:
    """Represents an ERF WRITE operation"""
    timestamp: int
    addr: int
    word: int


def parse_mm_quant_line(line: str) -> Optional[MMQuantOutput]:
    """Parse a MM_QUANT log line"""
    # Format: [timestamp] MM_QUANT | input: [...] | min: X | max: Y | output: [...]
    pattern = r'\[\s*(\d+)\]\s*MM_QUANT\s*\|\s*input:\s*\[(.*?)\]\s*\|\s*min:\s*(-?\d+)\s*\|\s*max:\s*(-?\d+)\s*\|\s*output:\s*\[(.*?)\]'
    match = re.search(pattern, line)
    if match:
        timestamp = int(match.group(1))
        input_str = match.group(2)
        min_val = int(match.group(3))
        max_val = int(match.group(4))
        output_str = match.group(5)

        input_values = [int(x) for x in input_str.split()]
        output_values = [int(x) for x in output_str.split()]

        return MMQuantOutput(timestamp, input_values, min_val, max_val, output_values)
    return None


def parse_pimc_out_line(line: str) -> Optional[PIMCOutput]:
    """Parse a PIMC_OUT log line"""
    # Format: [timestamp] PIMC_OUT | result: [...]
    pattern = r'\[\s*(\d+)\]\s*PIMC_OUT\s*\|\s*result:\s*\[(.*?)\]'
    match = re.search(pattern, line)
    if match:
        timestamp = int(match.group(1))
        result_str = match.group(2)
        result = [int(x) for x in result_str.split()]
        return PIMCOutput(timestamp, result)
    return None


def parse_erf_write_line(line: str) -> Optional[ERFWrite]:
    """Parse an ERF WRITE log line"""
    # Format: [timestamp] WRITE | addr: X | word: Y
    pattern = r'\[\s*(\d+)\]\s*WRITE\s*\|\s*addr:\s*(\d+)\s*\|\s*word:\s*(\d+)'
    match = re.search(pattern, line)
    if match:
        timestamp = int(match.group(1))
        addr = int(match.group(2))
        word = int(match.group(3))
        return ERFWrite(timestamp, addr, word)
    return None


def parse_imce_ctrl_line(line: str, row: int, col: int) -> Optional[IMCECtrlLogEntry]:
    """Parse an IMCE ctrl_pl log line"""
    # Format: [timestamp] [IMCE_EX] LOG_TYPE | ...
    pattern = r'\[\s*(\d+)\]\s*\[IMCE_EX\]\s*(\w+)'
    match = re.search(pattern, line)
    if match:
        timestamp = int(match.group(1))
        log_type = match.group(2)
        # Get content after the log type
        content_start = line.find(log_type) + len(log_type)
        content = line[content_start:].strip()
        return IMCECtrlLogEntry(timestamp, row, col, log_type, content)
    return None


def parse_imce_ctrl_log(log_path: str, row: int, col: int) -> List[IMCECtrlLogEntry]:
    """Parse IMCE ctrl_pl log file and extract entries"""
    entries = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if '[IMCE_EX]' in line:
                    result = parse_imce_ctrl_line(line, row, col)
                    if result:
                        entries.append(result)
    except FileNotFoundError:
        pass
    return entries


def find_imce_ctrl_logs(base_dir: str) -> Dict[Tuple[int, int], str]:
    """Find all IMCE ctrl_pl log files"""
    fsim_dir = os.path.join(base_dir, 'logs', 'rtl_runner', 'fsim_logs')
    ctrl_logs = {}

    if not os.path.exists(fsim_dir):
        return ctrl_logs

    for fname in os.listdir(fsim_dir):
        if 'u_ctrl_pl.log' in fname and 'imce_node.imce' in fname:
            coord = extract_imce_coord_from_path(fname)
            if coord:
                ctrl_logs[coord] = os.path.join(fsim_dir, fname)

    return ctrl_logs


def print_imce_ctrl_sorted(entries: List[IMCECtrlLogEntry], limit: int = 100,
                           from_end: bool = True, filter_type: Optional[str] = None):
    """Print IMCE ctrl log entries sorted by timestamp"""
    # Filter by type if specified
    if filter_type:
        entries = [e for e in entries if filter_type.upper() in e.log_type.upper()]

    # Sort by timestamp
    sorted_entries = sorted(entries, key=lambda x: x.timestamp)

    # Get last N entries if from_end
    if from_end and len(sorted_entries) > limit:
        sorted_entries = sorted_entries[-limit:]
    elif not from_end and len(sorted_entries) > limit:
        sorted_entries = sorted_entries[:limit]

    print(f"\n{'='*100}")
    print(f"IMCE Ctrl Log Entries (showing {len(sorted_entries)} of {len(entries)} total)")
    print(f"{'='*100}")

    for entry in sorted_entries:
        ts_str = f"{entry.timestamp:>18}"
        print(f"[{ts_str}] IMCE({entry.row},{entry.col}) {entry.log_type:<12} {entry.content}")


def find_final_stalls(entries_by_node: Dict[Tuple[int, int], List[IMCECtrlLogEntry]]) -> List[IMCECtrlLogEntry]:
    """
    Find the last STALL for each node, only if no other operation was executed after.
    Excludes stalls where the node completed its compute cycle and returned to S_IDLE.
    Returns list of final stall entries sorted by timestamp.
    """
    final_stalls = []

    for coord, entries in entries_by_node.items():
        if not entries:
            continue

        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)

        # Find the last entry
        last_entry = sorted_entries[-1]

        # Check if it's a STALL_START (meaning node is still stalled)
        if last_entry.log_type == 'STALL_START':
            # Check if node ever entered S_COMPUTE (actually started doing work)
            ever_computed = any(
                'state=S_COMPUTE' in e.content
                for e in sorted_entries
                if e.log_type == 'EXECUTE'
            )

            # Only filter out if the node completed a compute cycle and returned to S_IDLE.
            # Nodes that never entered S_COMPUTE are genuinely stuck (data never arrived).
            if ever_computed and len(sorted_entries) >= 2:
                prev_entry = sorted_entries[-2]
                if 'state=S_IDLE' in prev_entry.content:
                    continue  # Node finished compute and is waiting for next task

            final_stalls.append(last_entry)

    # Sort final stalls by timestamp
    return sorted(final_stalls, key=lambda x: x.timestamp)


def print_final_stalls(entries_by_node: Dict[Tuple[int, int], List[IMCECtrlLogEntry]]):
    """Print the final stall state for each node that is still stalled"""
    final_stalls = find_final_stalls(entries_by_node)

    print(f"\n{'='*100}")
    print(f"Final STALL State per Node (nodes still stalled at end of simulation)")
    print(f"{'='*100}")

    if not final_stalls:
        print("No nodes are in STALL state at the end of simulation.")
        return

    print(f"Found {len(final_stalls)} nodes still stalled:\n")

    for entry in final_stalls:
        ts_str = f"{entry.timestamp:>18}"
        print(f"[{ts_str}] IMCE({entry.row},{entry.col}) {entry.log_type:<12} {entry.content}")


def extract_imce_coord_from_path(path: str) -> Optional[Tuple[int, int]]:
    """Extract IMCE row,col coordinates from log file path"""
    # Pattern: core_row_X_.core_col_Y_.imce_node
    pattern = r'core_row_(\d+)_\.core_col_(\d+)_\.imce_node'
    match = re.search(pattern, path)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def extract_inode_coord_from_path(path: str) -> Optional[Tuple[int, int]]:
    """Extract INODE row,col coordinates from log file path"""
    # Pattern: core_row_X_.core_col_Y_.inode
    pattern = r'core_row_(\d+)_\.core_col_(\d+)_\.inode'
    match = re.search(pattern, path)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def parse_vpu_log(log_path: str) -> List[MMQuantOutput]:
    """Parse VPU log file and extract MM_QUANT outputs"""
    outputs = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'MM_QUANT' in line and 'input:' in line:
                    result = parse_mm_quant_line(line)
                    if result:
                        outputs.append(result)
    except FileNotFoundError:
        pass
    return outputs


def parse_post_imcu_log(log_path: str) -> List[PIMCOutput]:
    """Parse post_imcu log file and extract PIMC_OUT outputs"""
    outputs = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'PIMC_OUT' in line:
                    result = parse_pimc_out_line(line)
                    if result:
                        outputs.append(result)
    except FileNotFoundError:
        pass
    return outputs


def parse_erf_log(log_path: str) -> List[ERFWrite]:
    """Parse ERF log file and extract WRITE operations"""
    outputs = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'WRITE' in line and 'addr:' in line:
                    result = parse_erf_write_line(line)
                    if result:
                        outputs.append(result)
    except FileNotFoundError:
        pass
    return outputs


def find_fsim_logs(base_dir: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    """Find all FSIM log files organized by IMCE coordinate"""
    fsim_dir = os.path.join(base_dir, 'logs', 'rtl_runner', 'fsim_logs')

    imce_logs = defaultdict(dict)

    if not os.path.exists(fsim_dir):
        print(f"Warning: FSIM log directory not found: {fsim_dir}")
        return dict(imce_logs)

    for fname in os.listdir(fsim_dir):
        fpath = os.path.join(fsim_dir, fname)
        coord = extract_imce_coord_from_path(fname)

        if coord:
            if 'u_vpu.log' in fname:
                imce_logs[coord]['vpu'] = fpath
            elif 'u_erf.log' in fname:
                imce_logs[coord]['erf'] = fpath
            elif 'u_linebuffer.log' in fname and not fname.endswith('ctrl.log'):
                imce_logs[coord]['linebuffer'] = fpath
            elif 'post_imcu' in fname.lower():
                imce_logs[coord]['post_imcu'] = fpath

    return dict(imce_logs)


def load_reference_outputs(pkl_path: str) -> Dict:
    """Load reference outputs from pickle file"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def print_mm_quant_outputs(coord: Tuple[int, int], outputs: List[MMQuantOutput], limit: int = 10):
    """Print MM_QUANT outputs for an IMCE node"""
    print(f"\n{'='*80}")
    print(f"IMCE ({coord[0]}, {coord[1]}) - MM_QUANT Outputs ({len(outputs)} total)")
    print(f"{'='*80}")

    for i, out in enumerate(outputs[:limit]):
        print(f"\n[{i}] Time: {out.timestamp}")
        print(f"    Input:  {out.input_values}")
        print(f"    Min/Max: {out.min_val} / {out.max_val}")
        print(f"    Output: {out.output_values}")

    if len(outputs) > limit:
        print(f"\n... and {len(outputs) - limit} more outputs")


def print_pimc_outputs(coord: Tuple[int, int], outputs: List[PIMCOutput], limit: int = 10):
    """Print PIMC outputs for an IMCE node"""
    print(f"\n{'='*80}")
    print(f"IMCE ({coord[0]}, {coord[1]}) - PIMC_OUT Outputs ({len(outputs)} total)")
    print(f"{'='*80}")

    for i, out in enumerate(outputs[:limit]):
        print(f"\n[{i}] Time: {out.timestamp}")
        print(f"    Result: {out.result[:16]}...")  # First 16 channels

    if len(outputs) > limit:
        print(f"\n... and {len(outputs) - limit} more outputs")


def mm_quant_to_nchw(outputs: List[MMQuantOutput], C: int, H: int, W: int) -> np.ndarray:
    """
    Convert MM_QUANT outputs to NCHW format tensor.

    MM_QUANT processes in the order determined by linebuffer:
    - Each MM_QUANT output has 16 channel values (all channels at once for C=16)
    - For C=16, H=8, W=8: 64 outputs total (one per spatial position)
    - The spatial order is row-major (w varies fastest, then h)
    """
    num_blocks = max(1, C // 16)
    total_outputs = H * W * num_blocks

    if len(outputs) != total_outputs:
        print(f"Warning: Expected {total_outputs} outputs, got {len(outputs)}")

    # Create NCHW tensor
    tensor = np.zeros((1, C, H, W), dtype=np.uint8)

    # Fill tensor based on MM_QUANT output order
    # For C=16: one output per spatial position (h, w), row-major order
    # For C>16: multiple blocks per spatial position, need to handle interleaving
    for out_idx, out in enumerate(outputs):
        if num_blocks == 1:
            # C=16: simple row-major spatial ordering
            h = out_idx // W
            w = out_idx % W
            c_start = 0
        else:
            # C>16: blocks are interleaved per spatial position
            spatial_idx = out_idx // num_blocks
            block_idx = out_idx % num_blocks
            h = spatial_idx // W
            w = spatial_idx % W
            c_start = block_idx * 16

        if h < H and w < W:
            for c_off, val in enumerate(out.output_values):
                if c_start + c_off < C:
                    tensor[0, c_start + c_off, h, w] = val

    return tensor


def compare_with_reference(rtl_tensor: np.ndarray, ref_tensor: np.ndarray, name: str) -> bool:
    """Compare RTL output tensor with reference and report differences"""
    if rtl_tensor.shape != ref_tensor.shape:
        print(f"Shape mismatch: RTL {rtl_tensor.shape} vs Ref {ref_tensor.shape}")
        return False

    diff = rtl_tensor.astype(np.int32) - ref_tensor.astype(np.int32)
    num_diff = np.count_nonzero(diff)

    if num_diff == 0:
        print(f"MATCH: {name} - All {rtl_tensor.size} values match")
        return True
    else:
        print(f"MISMATCH: {name} - {num_diff}/{rtl_tensor.size} values differ")

        # Show first few mismatches
        diff_indices = np.argwhere(diff != 0)
        print(f"  First 10 mismatches:")
        for idx in diff_indices[:10]:
            idx_tuple = tuple(idx)
            rtl_val = rtl_tensor[idx_tuple]
            ref_val = ref_tensor[idx_tuple]
            print(f"    {idx_tuple}: RTL={rtl_val}, Ref={ref_val}")

        # Statistics
        print(f"  Max abs diff: {np.abs(diff).max()}")
        print(f"  Mean abs diff: {np.abs(diff).mean():.2f}")

        return False


def parse_hw_node_map(base_dir: str) -> Dict[int, str]:
    """Parse hw_node_map.txt to get relay node ID -> IMCE coordinate mapping"""
    map_path = os.path.join(base_dir, 'hw_node_map.txt')
    node_to_imce = {}

    try:
        with open(map_path, 'r') as f:
            content = f.read()
            # Parse lines like: 35: <NodeID.imce_3_2: 17>
            pattern = r'(\d+):\s*<NodeID\.imce_(\d+)_(\d+):'
            for match in re.finditer(pattern, content):
                node_id = int(match.group(1))
                row = int(match.group(2))
                col = int(match.group(3))
                node_to_imce[node_id] = (row, col)
    except FileNotFoundError:
        print(f"hw_node_map.txt not found")

    return node_to_imce


def get_ref_tensor_by_topo_index(ref_outputs: Dict, topo_idx: int) -> Optional[np.ndarray]:
    """Find reference tensor by topo-index"""
    search_str = f"topo-index:{topo_idx}"
    for key, value in ref_outputs.items():
        if search_str in key and isinstance(value, np.ndarray):
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description='Parse RTL simulation logs')
    parser.add_argument('test_dir', help='Test output directory (e.g., ../resnet8_subset13_pretrained_small_evl)')
    parser.add_argument('--compare', action='store_true', help='Show reference pickle contents')
    parser.add_argument('--imce', type=str, help='Filter to specific IMCE (e.g., "3,2")')
    parser.add_argument('--limit', type=int, default=10, help='Limit output per IMCE')
    parser.add_argument('--all', action='store_true', help='Show all outputs (no limit)')
    parser.add_argument('--deep-compare', action='store_true', help='Deep comparison with hw_node_map')
    parser.add_argument('--compare-imce', type=str, help='Compare specific IMCE with topo-index (e.g., "3,2:35")')
    parser.add_argument('--ctrl', action='store_true', help='Show IMCE ctrl_pl logs sorted by timestamp')
    parser.add_argument('--ctrl-limit', type=int, default=100, help='Number of ctrl log entries to show (default: 100)')
    parser.add_argument('--ctrl-filter', type=str, help='Filter ctrl logs by type (e.g., "EXECUTE", "STALL")')
    parser.add_argument('--ctrl-from-start', action='store_true', help='Show entries from start instead of end')
    parser.add_argument('--final-stall', action='store_true', help='Show final STALL state per node (only nodes still stalled)')

    args = parser.parse_args()

    base_dir = args.test_dir

    # --final-stall implies --ctrl mode
    if args.final_stall:
        args.ctrl = True

    # Handle --ctrl mode: show IMCE ctrl_pl logs sorted by timestamp
    if args.ctrl:
        ctrl_logs = find_imce_ctrl_logs(base_dir)
        print(f"Found ctrl_pl logs for {len(ctrl_logs)} IMCE nodes")

        # Filter by IMCE if specified
        if args.imce:
            row, col = map(int, args.imce.split(','))
            coord = (row, col)
            if coord in ctrl_logs:
                ctrl_logs = {coord: ctrl_logs[coord]}
            else:
                print(f"IMCE ({row}, {col}) ctrl_pl log not found")
                return

        # Collect all entries from all IMCE nodes
        all_entries = []
        entries_by_node = {}
        for coord, log_path in ctrl_logs.items():
            entries = parse_imce_ctrl_log(log_path, coord[0], coord[1])
            entries_by_node[coord] = entries
            all_entries.extend(entries)

        if not all_entries:
            print("No ctrl_pl log entries found")
            return

        # Handle --final-stall mode
        if args.final_stall:
            print_final_stalls(entries_by_node)
            return

        # Print sorted entries
        print_imce_ctrl_sorted(
            all_entries,
            limit=args.ctrl_limit,
            from_end=not args.ctrl_from_start,
            filter_type=args.ctrl_filter
        )
        return

    # Find all IMCE logs
    imce_logs = find_fsim_logs(base_dir)
    print(f"Found logs for {len(imce_logs)} IMCE nodes")

    # Filter if specified
    if args.imce:
        row, col = map(int, args.imce.split(','))
        coord = (row, col)
        if coord in imce_logs:
            imce_logs = {coord: imce_logs[coord]}
        else:
            print(f"IMCE ({row}, {col}) not found in logs")
            return

    # Parse logs for each IMCE
    imce_mm_quant = {}  # Store parsed outputs for comparison
    for coord in sorted(imce_logs.keys()):
        logs = imce_logs[coord]

        # Parse VPU log for MM_QUANT
        if 'vpu' in logs:
            mm_quant_outputs = parse_vpu_log(logs['vpu'])
            if mm_quant_outputs:
                imce_mm_quant[coord] = mm_quant_outputs
                if not args.deep_compare and not args.compare_imce:
                    limit = None if args.all else args.limit
                    print_mm_quant_outputs(coord, mm_quant_outputs, limit or len(mm_quant_outputs))

        # Parse post_imcu log for PIMC_OUT (if exists)
        if 'post_imcu' in logs:
            pimc_outputs = parse_post_imcu_log(logs['post_imcu'])
            if pimc_outputs and not args.deep_compare and not args.compare_imce:
                limit = None if args.all else args.limit
                print_pimc_outputs(coord, pimc_outputs, limit or len(pimc_outputs))

    # Direct IMCE to topo-index comparison
    if args.compare_imce:
        pkl_path = os.path.join(base_dir, 'debug_executor_output_tensors.pkl')
        if not os.path.exists(pkl_path):
            print(f"Reference pickle not found: {pkl_path}")
            return

        ref_outputs = load_reference_outputs(pkl_path)

        # Parse argument like "3,2:35"
        imce_str, topo_str = args.compare_imce.split(':')
        row, col = map(int, imce_str.split(','))
        topo_idx = int(topo_str)
        coord = (row, col)

        print(f"\n{'='*80}")
        print(f"Comparing IMCE ({row}, {col}) with topo-index {topo_idx}")
        print(f"{'='*80}")

        if coord not in imce_mm_quant:
            print(f"No MM_QUANT outputs found for IMCE ({row}, {col})")
            return

        mm_outputs = imce_mm_quant[coord]
        print(f"MM_QUANT outputs: {len(mm_outputs)}")

        if mm_outputs:
            print(f"Min/Max params: {mm_outputs[0].min_val} / {mm_outputs[0].max_val}")

        ref_tensor = get_ref_tensor_by_topo_index(ref_outputs, topo_idx)
        if ref_tensor is None:
            print(f"No reference found for topo-index {topo_idx}")
            return

        print(f"Reference shape: {ref_tensor.shape}, dtype: {ref_tensor.dtype}")

        if len(ref_tensor.shape) == 4:
            N, C, H, W = ref_tensor.shape
            rtl_tensor = mm_quant_to_nchw(mm_outputs, C, H, W)
            compare_with_reference(rtl_tensor, ref_tensor, f"topo-{topo_idx}")

            # Show side-by-side for first spatial position
            print(f"\nFirst spatial position (h=0, w=0) comparison:")
            print(f"  RTL:  {rtl_tensor[0, :, 0, 0]}")
            print(f"  Ref:  {ref_tensor[0, :, 0, 0]}")
            print(f"\nSecond spatial position (h=0, w=1) comparison:")
            print(f"  RTL:  {rtl_tensor[0, :, 0, 1]}")
            print(f"  Ref:  {ref_tensor[0, :, 0, 1]}")

        return  # Skip other comparisons

    # Deep comparison with reference
    if args.deep_compare:
        pkl_path = os.path.join(base_dir, 'debug_executor_output_tensors.pkl')
        if not os.path.exists(pkl_path):
            print(f"Reference pickle not found: {pkl_path}")
            return

        ref_outputs = load_reference_outputs(pkl_path)
        node_to_imce = parse_hw_node_map(base_dir)

        print(f"\n{'='*80}")
        print("Deep Comparison: RTL vs Reference")
        print(f"{'='*80}")

        # Map IMCE coordinates back to topo indices
        imce_to_nodes = defaultdict(list)
        for node_id, imce_coord in node_to_imce.items():
            imce_to_nodes[imce_coord].append(node_id)

        for coord, mm_outputs in sorted(imce_mm_quant.items()):
            print(f"\n--- IMCE ({coord[0]}, {coord[1]}) ---")
            print(f"MM_QUANT outputs: {len(mm_outputs)}")

            # Find corresponding relay nodes
            nodes = imce_to_nodes.get(coord, [])
            print(f"Mapped relay nodes: {nodes}")

            # Try to find matching reference tensor
            for node_id in nodes:
                ref_tensor = get_ref_tensor_by_topo_index(ref_outputs, node_id)
                if ref_tensor is not None and ref_tensor.dtype == np.uint8:
                    print(f"\n  Found reference for topo-index {node_id}: shape={ref_tensor.shape}")

                    # Infer C, H, W from shape (assume NCHW)
                    if len(ref_tensor.shape) == 4:
                        N, C, H, W = ref_tensor.shape
                        expected_outputs = H * W * (C // 16) if C >= 16 else H * W

                        print(f"  Expected MM_QUANT outputs: {expected_outputs}")
                        print(f"  Actual MM_QUANT outputs: {len(mm_outputs)}")

                        if len(mm_outputs) > 0:
                            # Convert RTL outputs to tensor
                            rtl_tensor = mm_quant_to_nchw(mm_outputs, C, H, W)
                            compare_with_reference(rtl_tensor, ref_tensor, f"topo-{node_id}")

    # Basic comparison
    elif args.compare:
        pkl_path = os.path.join(base_dir, 'debug_executor_output_tensors.pkl')
        if os.path.exists(pkl_path):
            print(f"\n{'='*80}")
            print("Reference Outputs from pickle")
            print(f"{'='*80}")

            ref_outputs = load_reference_outputs(pkl_path)

            for key, value in ref_outputs.items():
                if isinstance(value, np.ndarray):
                    print(f"\n{key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"  min={value.min()}, max={value.max()}")
                    # Show first few values
                    flat = value.flatten()
                    print(f"  first 10: {flat[:10]}")
                else:
                    print(f"\n{key}: {type(value)}")
        else:
            print(f"Reference pickle not found: {pkl_path}")


if __name__ == '__main__':
    main()
