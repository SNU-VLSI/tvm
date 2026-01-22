#!/usr/bin/env python3
"""
Log Parser for IMCFlow Debug - Compares Python Simulator and RTL outputs

Parses both Python simulator and RTL simulation logs to find divergence points.
Focuses on:
- MM_QUANT outputs (VPU quantization)
- IMCU compute results (qconv output)
- VPU operations (MULTL, ADD)

Usage:
    python parse_output_from_log.py <test_dir> [options]

Examples:
    # Compare Python vs RTL MM_QUANT outputs
    python parse_output_from_log.py ../resnet8_subset13_pretrained_small_evl --compare-mm-quant

    # Compare IMCU outputs
    python parse_output_from_log.py ../resnet8_subset13_pretrained_small_evl --compare-imcu

    # Show specific IMCE outputs
    python parse_output_from_log.py ../resnet8_subset13_pretrained_small_evl --imce 3,2
"""

import os
import re
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MMQuantOutput:
    """Represents a single MM_QUANT operation output"""
    imce_coord: Tuple[int, int]
    input_values: List[int]
    min_val: int
    max_val: int
    output_values: List[int]
    source: str = ""  # "python" or "rtl"
    timestamp: Optional[int] = None


@dataclass
class IMCUOutput:
    """Represents IMCU compute result (qconv output)"""
    imce_coord: Tuple[int, int]
    result: List[int]
    source: str = ""  # "python" or "rtl"
    timestamp: Optional[int] = None


@dataclass
class VPUOp:
    """Represents a VPU operation (MULTL, ADD, etc.)"""
    imce_coord: Tuple[int, int]
    op_type: str  # "MULTL", "ADD", etc.
    operand_a: List[int]
    operand_b: List[int]
    result: List[int]
    source: str = ""
    timestamp: Optional[int] = None


# =============================================================================
# Python Log Parsers
# =============================================================================

def parse_python_mm_quant(log_path: str) -> Dict[Tuple[int, int], List[MMQuantOutput]]:
    """Parse Python simulator log for MM_QUANT outputs"""
    outputs = defaultdict(list)

    # Pattern: IMCE.X.Y MM_QUANT | input: [...] | min: X | max: Y | output: [...]
    pattern = r'IMCE\.(\d+)\.(\d+)\s+MM_QUANT\s*\|\s*input:\s*\[(.*?)\]\s*\|\s*min:\s*(-?\d+)\s*\|\s*max:\s*(-?\d+)\s*\|\s*output:\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'MM_QUANT' in line and 'input:' in line:
                    match = re.search(pattern, line)
                    if match:
                        row = int(match.group(1))
                        col = int(match.group(2))
                        input_str = match.group(3)
                        min_val = int(match.group(4))
                        max_val = int(match.group(5))
                        output_str = match.group(6)

                        input_values = [int(x) for x in input_str.split()]
                        output_values = [int(x) for x in output_str.split()]

                        coord = (row, col)
                        outputs[coord].append(MMQuantOutput(
                            imce_coord=coord,
                            input_values=input_values,
                            min_val=min_val,
                            max_val=max_val,
                            output_values=output_values,
                            source="python"
                        ))
    except FileNotFoundError:
        print(f"Python log not found: {log_path}")

    return dict(outputs)


def parse_python_imcu_output(log_path: str) -> Dict[Tuple[int, int], List[IMCUOutput]]:
    """Parse Python simulator log for IMCU compute results"""
    outputs = defaultdict(list)
    current_imce = None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Track current IMCE from lines like [IMCE.3.3.imcu]
                imce_match = re.search(r'\[IMCE\.(\d+)\.(\d+)\.imcu\]', line)
                if imce_match:
                    current_imce = (int(imce_match.group(1)), int(imce_match.group(2)))

                # Parse IMCU Compute post_result
                if '[IMCU Compute] post_result:' in line:
                    # Pattern: [IMCU Compute] post_result: [val1 val2 ...]
                    result_match = re.search(r'\[IMCU Compute\] post_result:\s*\[(.*?)\]', line)
                    if result_match and current_imce:
                        result_str = result_match.group(1)
                        result = [int(x) for x in result_str.split()]

                        outputs[current_imce].append(IMCUOutput(
                            imce_coord=current_imce,
                            result=result,
                            source="python"
                        ))
    except FileNotFoundError:
        print(f"Python log not found: {log_path}")

    return dict(outputs)


def parse_python_vpu_ops(log_path: str) -> Dict[Tuple[int, int], List[VPUOp]]:
    """Parse Python simulator log for VPU operations (MULTL, ADD)"""
    outputs = defaultdict(list)

    # Pattern for MULTL: [RINST] IMCE.X.Y OP_MULTL: (a * b) & mask_low = result
    multl_pattern = r'\[RINST\]\s+IMCE\.(\d+)\.(\d+)\s+OP_MULTL:\s*\(\[(.*?)\]\s*\*\s*\[(.*?)\]\)\s*&\s*mask_low\s*=\s*\[(.*?)\]'

    # Pattern for ADD: [RINST] IMCE.X.Y OP_ADD: a + b = result
    add_pattern = r'\[RINST\]\s+IMCE\.(\d+)\.(\d+)\s+OP_ADD:\s*\[(.*?)\]\s*\+\s*\[(.*?)\]\s*=\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if '[RINST]' in line:
                    # Try MULTL pattern
                    match = re.search(multl_pattern, line)
                    if match:
                        row = int(match.group(1))
                        col = int(match.group(2))
                        coord = (row, col)

                        op_a = [int(x) for x in match.group(3).split()]
                        op_b = [int(x) for x in match.group(4).split()]
                        result = [int(x) for x in match.group(5).split()]

                        outputs[coord].append(VPUOp(
                            imce_coord=coord,
                            op_type="MULTL",
                            operand_a=op_a,
                            operand_b=op_b,
                            result=result,
                            source="python"
                        ))
                        continue

                    # Try ADD pattern
                    match = re.search(add_pattern, line)
                    if match:
                        row = int(match.group(1))
                        col = int(match.group(2))
                        coord = (row, col)

                        op_a = [int(x) for x in match.group(3).split()]
                        op_b = [int(x) for x in match.group(4).split()]
                        result = [int(x) for x in match.group(5).split()]

                        outputs[coord].append(VPUOp(
                            imce_coord=coord,
                            op_type="ADD",
                            operand_a=op_a,
                            operand_b=op_b,
                            result=result,
                            source="python"
                        ))
    except FileNotFoundError:
        print(f"Python log not found: {log_path}")

    return dict(outputs)


# =============================================================================
# RTL Log Parsers
# =============================================================================

def extract_imce_coord_from_path(path: str) -> Optional[Tuple[int, int]]:
    """Extract IMCE row,col coordinates from log file path"""
    pattern = r'core_row_(\d+)_\.core_col_(\d+)_\.imce_node'
    match = re.search(pattern, path)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def parse_rtl_mm_quant(log_path: str, coord: Tuple[int, int]) -> List[MMQuantOutput]:
    """Parse RTL VPU log for MM_QUANT outputs"""
    outputs = []

    # Pattern: [timestamp] MM_QUANT | input: [...] | min: X | max: Y | output: [...]
    pattern = r'\[\s*(\d+)\]\s*MM_QUANT\s*\|\s*input:\s*\[(.*?)\]\s*\|\s*min:\s*(-?\d+)\s*\|\s*max:\s*(-?\d+)\s*\|\s*output:\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'MM_QUANT' in line and 'input:' in line:
                    match = re.search(pattern, line)
                    if match:
                        timestamp = int(match.group(1))
                        input_str = match.group(2)
                        min_val = int(match.group(3))
                        max_val = int(match.group(4))
                        output_str = match.group(5)

                        input_values = [int(x) for x in input_str.split()]
                        output_values = [int(x) for x in output_str.split()]

                        outputs.append(MMQuantOutput(
                            imce_coord=coord,
                            input_values=input_values,
                            min_val=min_val,
                            max_val=max_val,
                            output_values=output_values,
                            source="rtl",
                            timestamp=timestamp
                        ))
    except FileNotFoundError:
        pass

    return outputs


def parse_rtl_pimc_output(log_path: str, coord: Tuple[int, int]) -> List[IMCUOutput]:
    """Parse RTL post_imcu log for PIMC_OUT results"""
    outputs = []

    # Pattern: [timestamp] PIMC_OUT | result: [...]
    pattern = r'\[\s*(\d+)\]\s*PIMC_OUT\s*\|\s*result:\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'PIMC_OUT' in line:
                    match = re.search(pattern, line)
                    if match:
                        timestamp = int(match.group(1))
                        result_str = match.group(2)
                        result = [int(x) for x in result_str.split()]

                        outputs.append(IMCUOutput(
                            imce_coord=coord,
                            result=result,
                            source="rtl",
                            timestamp=timestamp
                        ))
    except FileNotFoundError:
        pass

    return outputs


def parse_rtl_vpu_ops(log_path: str, coord: Tuple[int, int]) -> List[VPUOp]:
    """Parse RTL VPU log for MULTL/ADD operations"""
    outputs = []

    # Pattern for MULTL: [timestamp] MULTL | opA: [...] | opB: [...] | shift_amt: X | result: [...]
    multl_pattern = r'\[\s*(\d+)\]\s*MULTL\s*\|\s*opA:\s*\[(.*?)\]\s*\|\s*opB:\s*\[(.*?)\]\s*\|.*?\|\s*result:\s*\[(.*?)\]'

    # Pattern for ADD: [timestamp] ADD | opA: [...] | opB: [...] | result: [...]
    add_pattern = r'\[\s*(\d+)\]\s*ADD\s*\|\s*opA:\s*\[(.*?)\]\s*\|\s*opB:\s*\[(.*?)\]\s*\|\s*result:\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Try MULTL pattern
                match = re.search(multl_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    op_a = [int(x) for x in match.group(2).split()]
                    op_b = [int(x) for x in match.group(3).split()]
                    result = [int(x) for x in match.group(4).split()]

                    outputs.append(VPUOp(
                        imce_coord=coord,
                        op_type="MULTL",
                        operand_a=op_a,
                        operand_b=op_b,
                        result=result,
                        source="rtl",
                        timestamp=timestamp
                    ))
                    continue

                # Try ADD pattern
                match = re.search(add_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    op_a = [int(x) for x in match.group(2).split()]
                    op_b = [int(x) for x in match.group(3).split()]
                    result = [int(x) for x in match.group(4).split()]

                    outputs.append(VPUOp(
                        imce_coord=coord,
                        op_type="ADD",
                        operand_a=op_a,
                        operand_b=op_b,
                        result=result,
                        source="rtl",
                        timestamp=timestamp
                    ))
    except FileNotFoundError:
        pass

    return outputs


# =============================================================================
# Log Collection
# =============================================================================

def find_rtl_logs(base_dir: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    """Find all RTL FSIM log files organized by IMCE coordinate"""
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
            elif 'post_imcu' in fname.lower():
                imce_logs[coord]['post_imcu'] = fpath

    return dict(imce_logs)


def get_python_log_path(base_dir: str) -> str:
    """Get Python simulator log path"""
    return os.path.join(base_dir, 'logs', 'py_runner', 'now.debug.log')


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_mm_quant_outputs(py_outputs: List[MMQuantOutput], rtl_outputs: List[MMQuantOutput],
                             coord: Tuple[int, int], verbose: bool = False) -> Tuple[int, int, int]:
    """Compare Python and RTL MM_QUANT outputs. Returns (match, mismatch, total)"""
    match_count = 0
    mismatch_count = 0

    min_len = min(len(py_outputs), len(rtl_outputs))

    if len(py_outputs) != len(rtl_outputs):
        print(f"  Warning: Output count mismatch - Python: {len(py_outputs)}, RTL: {len(rtl_outputs)}")

    for i in range(min_len):
        py_out = py_outputs[i]
        rtl_out = rtl_outputs[i]

        # Compare outputs
        if py_out.output_values == rtl_out.output_values:
            match_count += 1
        else:
            mismatch_count += 1
            if verbose or mismatch_count <= 5:
                print(f"  MISMATCH at index {i}:")
                print(f"    Python: {py_out.output_values}")
                print(f"    RTL:    {rtl_out.output_values}")
                if py_out.input_values != rtl_out.input_values:
                    print(f"    Input also differs:")
                    print(f"      Python: {py_out.input_values}")
                    print(f"      RTL:    {rtl_out.input_values}")

    return match_count, mismatch_count, min_len


def compare_imcu_outputs(py_outputs: List[IMCUOutput], rtl_outputs: List[IMCUOutput],
                         coord: Tuple[int, int], verbose: bool = False) -> Tuple[int, int, int]:
    """Compare Python and RTL IMCU outputs. Returns (match, mismatch, total)"""
    match_count = 0
    mismatch_count = 0

    min_len = min(len(py_outputs), len(rtl_outputs))

    if len(py_outputs) != len(rtl_outputs):
        print(f"  Warning: Output count mismatch - Python: {len(py_outputs)}, RTL: {len(rtl_outputs)}")

    for i in range(min_len):
        py_out = py_outputs[i]
        rtl_out = rtl_outputs[i]

        # Compare first 16 channels (or actual used channels)
        py_result = py_out.result[:16]  # First 16 are actual, rest are padding
        rtl_result = rtl_out.result[:16]

        if py_result == rtl_result:
            match_count += 1
        else:
            mismatch_count += 1
            if verbose or mismatch_count <= 5:
                print(f"  MISMATCH at index {i}:")
                print(f"    Python: {py_result}")
                print(f"    RTL:    {rtl_result}")
                # Show difference
                diff = [p - r for p, r in zip(py_result, rtl_result)]
                print(f"    Diff:   {diff}")

    return match_count, mismatch_count, min_len


# =============================================================================
# Display Functions
# =============================================================================

def print_summary(title: str, data: Dict[Tuple[int, int], List], limit: int = 5):
    """Print summary of parsed data"""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    for coord in sorted(data.keys()):
        items = data[coord]
        print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")

        for i, item in enumerate(items[:limit]):
            if hasattr(item, 'output_values'):  # MMQuantOutput
                print(f"  [{i}] output: {item.output_values}")
            elif hasattr(item, 'result'):  # IMCUOutput
                print(f"  [{i}] result: {item.result[:16]}...")
            elif hasattr(item, 'op_type'):  # VPUOp
                print(f"  [{i}] {item.op_type}: result={item.result}")

        if len(items) > limit:
            print(f"  ... and {len(items) - limit} more")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Python simulator and RTL logs')
    parser.add_argument('test_dir', help='Test output directory')
    parser.add_argument('--compare-mm-quant', action='store_true', help='Compare MM_QUANT outputs')
    parser.add_argument('--compare-imcu', action='store_true', help='Compare IMCU outputs')
    parser.add_argument('--compare-vpu', action='store_true', help='Compare VPU operations')
    parser.add_argument('--compare-all', action='store_true', help='Compare all available data')
    parser.add_argument('--imce', type=str, help='Filter to specific IMCE (e.g., "3,2")')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all mismatches')
    parser.add_argument('--show-python', action='store_true', help='Show Python log data')
    parser.add_argument('--show-rtl', action='store_true', help='Show RTL log data')

    args = parser.parse_args()
    base_dir = args.test_dir

    # Get log paths
    py_log_path = get_python_log_path(base_dir)
    rtl_logs = find_rtl_logs(base_dir)

    print(f"Python log: {py_log_path}")
    print(f"RTL logs found for {len(rtl_logs)} IMCE nodes")

    # Filter IMCE if specified
    filter_coord = None
    if args.imce:
        row, col = map(int, args.imce.split(','))
        filter_coord = (row, col)
        print(f"Filtering to IMCE ({row}, {col})")

    # Parse Python logs
    print("\nParsing Python logs...")
    py_mm_quant = parse_python_mm_quant(py_log_path)
    py_imcu = parse_python_imcu_output(py_log_path)
    py_vpu = parse_python_vpu_ops(py_log_path)

    print(f"  MM_QUANT: {sum(len(v) for v in py_mm_quant.values())} entries from {len(py_mm_quant)} IMCEs")
    print(f"  IMCU: {sum(len(v) for v in py_imcu.values())} entries from {len(py_imcu)} IMCEs")
    print(f"  VPU ops: {sum(len(v) for v in py_vpu.values())} entries from {len(py_vpu)} IMCEs")

    # Parse RTL logs
    print("\nParsing RTL logs...")
    rtl_mm_quant = {}
    rtl_imcu = {}
    rtl_vpu = {}

    for coord, logs in rtl_logs.items():
        if filter_coord and coord != filter_coord:
            continue

        if 'vpu' in logs:
            mm_outputs = parse_rtl_mm_quant(logs['vpu'], coord)
            if mm_outputs:
                rtl_mm_quant[coord] = mm_outputs

            vpu_outputs = parse_rtl_vpu_ops(logs['vpu'], coord)
            if vpu_outputs:
                rtl_vpu[coord] = vpu_outputs

        if 'post_imcu' in logs:
            imcu_outputs = parse_rtl_pimc_output(logs['post_imcu'], coord)
            if imcu_outputs:
                rtl_imcu[coord] = imcu_outputs

    print(f"  MM_QUANT: {sum(len(v) for v in rtl_mm_quant.values())} entries from {len(rtl_mm_quant)} IMCEs")
    print(f"  IMCU: {sum(len(v) for v in rtl_imcu.values())} entries from {len(rtl_imcu)} IMCEs")
    print(f"  VPU ops: {sum(len(v) for v in rtl_vpu.values())} entries from {len(rtl_vpu)} IMCEs")

    # Show data if requested
    if args.show_python:
        if py_mm_quant:
            print_summary("Python MM_QUANT", py_mm_quant)
        if py_imcu:
            print_summary("Python IMCU", py_imcu)

    if args.show_rtl:
        if rtl_mm_quant:
            print_summary("RTL MM_QUANT", rtl_mm_quant)
        if rtl_imcu:
            print_summary("RTL IMCU", rtl_imcu)

    # Compare MM_QUANT
    if args.compare_mm_quant or args.compare_all:
        print(f"\n{'='*80}")
        print("Comparing MM_QUANT outputs (Python vs RTL)")
        print(f"{'='*80}")

        all_coords = set(py_mm_quant.keys()) | set(rtl_mm_quant.keys())
        if filter_coord:
            all_coords = {filter_coord} & all_coords

        total_match = 0
        total_mismatch = 0

        for coord in sorted(all_coords):
            py_out = py_mm_quant.get(coord, [])
            rtl_out = rtl_mm_quant.get(coord, [])

            print(f"\nIMCE ({coord[0]}, {coord[1]}): Python={len(py_out)}, RTL={len(rtl_out)}")

            if py_out and rtl_out:
                match, mismatch, total = compare_mm_quant_outputs(py_out, rtl_out, coord, args.verbose)
                total_match += match
                total_mismatch += mismatch

                if mismatch == 0:
                    print(f"  ✓ All {total} outputs MATCH")
                else:
                    print(f"  ✗ {mismatch}/{total} outputs MISMATCH")
            elif not py_out:
                print(f"  No Python data")
            else:
                print(f"  No RTL data")

        print(f"\nTotal: {total_match} match, {total_mismatch} mismatch")

    # Compare IMCU
    if args.compare_imcu or args.compare_all:
        print(f"\n{'='*80}")
        print("Comparing IMCU outputs (Python vs RTL)")
        print(f"{'='*80}")

        all_coords = set(py_imcu.keys()) | set(rtl_imcu.keys())
        if filter_coord:
            all_coords = {filter_coord} & all_coords

        total_match = 0
        total_mismatch = 0

        for coord in sorted(all_coords):
            py_out = py_imcu.get(coord, [])
            rtl_out = rtl_imcu.get(coord, [])

            print(f"\nIMCE ({coord[0]}, {coord[1]}): Python={len(py_out)}, RTL={len(rtl_out)}")

            if py_out and rtl_out:
                match, mismatch, total = compare_imcu_outputs(py_out, rtl_out, coord, args.verbose)
                total_match += match
                total_mismatch += mismatch

                if mismatch == 0:
                    print(f"  ✓ All {total} outputs MATCH")
                else:
                    print(f"  ✗ {mismatch}/{total} outputs MISMATCH")
            elif not py_out:
                print(f"  No Python data")
            else:
                print(f"  No RTL data (need to re-run RTL with PIMC_OUT logging)")

        print(f"\nTotal: {total_match} match, {total_mismatch} mismatch")


if __name__ == '__main__':
    main()
