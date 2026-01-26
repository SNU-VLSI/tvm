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

    # Save parsed data to text file
    python parse_output_from_log.py ../resnet8_subset13_pretrained_small_evl --out parsed_data.txt
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


@dataclass
class LinebufferInput:
    """Represents linebuffer input handshake data"""
    imce_coord: Tuple[int, int]
    count: int
    data: str  # hex string
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class LinebufferOutput:
    """Represents linebuffer output handshake data"""
    imce_coord: Tuple[int, int]
    count: int
    bitpos: int
    adata: str  # hex string
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class LinebufferConfig:
    """Represents linebuffer layer configuration"""
    imce_coord: Tuple[int, int]
    height: int
    width: int
    stride: int
    pad: int
    ksel: int
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class IMCUInput:
    """Represents IMCU core input data"""
    imce_coord: Tuple[int, int]
    count: int
    bitpos: int
    adata: str  # hex string
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class PostIMCUAccStep:
    """Represents post_imcu accumulation step"""
    imce_coord: Tuple[int, int]
    output_idx: int
    acc_step: int
    din: List[int]
    bp_mult: List[int]
    i_cnt: int
    b_cnt: int
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class PostIMCUFifoPush:
    """Represents post_imcu FIFO push (data entering FIFO)"""
    imce_coord: Tuple[int, int]
    output_idx: int
    data: List[int]
    fifo_full: bool
    fifo_empty: bool
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class PostIMCUFifoPop:
    """Represents post_imcu FIFO pop (data leaving FIFO to VPU)"""
    imce_coord: Tuple[int, int]
    data: List[int]
    fifo_full: bool
    fifo_empty: bool
    source: str = ""
    timestamp: Optional[int] = None


@dataclass
class VPUInput:
    """Represents VPU input (MM_QUANT input)"""
    imce_coord: Tuple[int, int]
    input_values: List[int]
    thresholds: List[int]
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
    """Parse RTL post_imcu log for IMCU_OUT/PIMC_OUT results"""
    outputs = []

    # Pattern: [timestamp] [IMCU_OUT] output_idx=X | result: [...]
    # or legacy: [timestamp] PIMC_OUT | result: [...]
    pattern_new = r'\[\s*(\d+)\]\s*\[IMCU_OUT\]\s*output_idx=(\d+)\s*\|\s*result:\s*\[(.*?)\]'
    pattern_legacy = r'\[\s*(\d+)\]\s*PIMC_OUT\s*\|\s*result:\s*\[(.*?)\]'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'IMCU_OUT' in line or 'PIMC_OUT' in line:
                    # Try new format first
                    match = re.search(pattern_new, line)
                    if match:
                        timestamp = int(match.group(1))
                        output_idx = int(match.group(2))
                        result_str = match.group(3)
                        result = [int(x) for x in result_str.split()]

                        outputs.append(IMCUOutput(
                            imce_coord=coord,
                            result=result,
                            source="rtl",
                            timestamp=timestamp
                        ))
                        continue

                    # Try legacy format
                    match = re.search(pattern_legacy, line)
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


def parse_rtl_linebuffer(log_path: str, coord: Tuple[int, int]) -> Tuple[List[LinebufferInput], List[LinebufferOutput], List[LinebufferConfig]]:
    """Parse RTL linebuffer log for input/output handshakes and config"""
    inputs = []
    outputs = []
    configs = []

    # Pattern for INPUT_HS: [timestamp] [LBUF_HS] INPUT_HS | count=X | data=0x...
    input_pattern = r'\[\s*(\d+)\]\s*\[LBUF_HS\]\s*INPUT_HS\s*\|\s*count=(\d+)\s*\|\s*data=(0x[0-9a-fA-F]+)'

    # Pattern for OUTPUT_HS: [timestamp] [LBUF_HS] OUTPUT_HS | count=X | bitpos=Y | adata=0x...
    output_pattern = r'\[\s*(\d+)\]\s*\[LBUF_HS\]\s*OUTPUT_HS\s*\|\s*count=(\d+)\s*\|\s*bitpos=(\d+)\s*\|\s*adata=(0x[0-9a-fA-F]+)'

    # Pattern for LAYER_UPDATE: [timestamp] [LBUF_CFG] LAYER_UPDATE | H=X | W=Y | stride=X | pad=X | ksel=X
    config_pattern = r'\[\s*(\d+)\]\s*\[LBUF_CFG\]\s*LAYER_UPDATE\s*\|\s*H=(\d+)\s*\|\s*W=(\d+)\s*\|\s*stride=(\d+)\s*\|\s*pad=(\d+)\s*\|\s*ksel=(\d+)'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Try INPUT_HS pattern
                match = re.search(input_pattern, line)
                if match:
                    inputs.append(LinebufferInput(
                        imce_coord=coord,
                        count=int(match.group(2)),
                        data=match.group(3),
                        source="rtl",
                        timestamp=int(match.group(1))
                    ))
                    continue

                # Try OUTPUT_HS pattern
                match = re.search(output_pattern, line)
                if match:
                    outputs.append(LinebufferOutput(
                        imce_coord=coord,
                        count=int(match.group(2)),
                        bitpos=int(match.group(3)),
                        adata=match.group(4),
                        source="rtl",
                        timestamp=int(match.group(1))
                    ))
                    continue

                # Try LAYER_UPDATE pattern
                match = re.search(config_pattern, line)
                if match:
                    configs.append(LinebufferConfig(
                        imce_coord=coord,
                        height=int(match.group(2)),
                        width=int(match.group(3)),
                        stride=int(match.group(4)),
                        pad=int(match.group(5)),
                        ksel=int(match.group(6)),
                        source="rtl",
                        timestamp=int(match.group(1))
                    ))
    except FileNotFoundError:
        pass

    return inputs, outputs, configs


def parse_rtl_imcu_input(log_path: str, coord: Tuple[int, int]) -> List[IMCUInput]:
    """Parse RTL IMCU core log for input data"""
    inputs = []

    # Pattern: [timestamp] [IMCU_IN] count=X | bitpos=Y | adata=0x...
    pattern = r'\[\s*(\d+)\]\s*\[IMCU_IN\]\s*count=(\d+)\s*\|\s*bitpos=(\d+)\s*\|\s*adata=(0x[0-9a-fA-F]+)'

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    inputs.append(IMCUInput(
                        imce_coord=coord,
                        count=int(match.group(2)),
                        bitpos=int(match.group(3)),
                        adata=match.group(4),
                        source="rtl",
                        timestamp=int(match.group(1))
                    ))
    except FileNotFoundError:
        pass

    return inputs


def parse_rtl_post_imcu(log_path: str, coord: Tuple[int, int]) -> Tuple[List[PostIMCUAccStep], List[IMCUOutput], List[PostIMCUFifoPush], List[PostIMCUFifoPop]]:
    """Parse RTL post_imcu log for accumulation steps, outputs, and FIFO operations"""
    acc_steps = []
    outputs = []
    fifo_pushes = []
    fifo_pops = []

    # Pattern for ACC_STEP + ACC_DIN + ACC_MULT (we'll combine them)
    # [timestamp] [ACC_STEP] output_idx=X acc_step=Y | i_cnt=Z b_cnt=W ...
    step_pattern = r'\[\s*(\d+)\]\s*\[ACC_STEP\]\s*output_idx=(\d+)\s*acc_step=(\d+)\s*\|\s*i_cnt=(\d+)\s*b_cnt=(\d+)'
    # [timestamp] [ACC_DIN] output_idx=X acc_step=Y | din: [...]
    din_pattern = r'\[\s*(\d+)\]\s*\[ACC_DIN\]\s*output_idx=(\d+)\s*acc_step=(\d+)\s*\|\s*din:\s*\[(.*?)\]'
    # [timestamp] [ACC_MULT] output_idx=X acc_step=Y | bp_mult: [...]
    mult_pattern = r'\[\s*(\d+)\]\s*\[ACC_MULT\]\s*output_idx=(\d+)\s*acc_step=(\d+)\s*\|\s*bp_mult:\s*\[(.*?)\]'
    # [timestamp] [IMCU_OUT] output_idx=X | result: [...] (legacy format)
    out_pattern = r'\[\s*(\d+)\]\s*\[IMCU_OUT\]\s*output_idx=(\d+)\s*\|\s*result:\s*\[(.*?)\]'
    # [timestamp] [OUT_FIFO_PUSH] output_idx=X | data: [...] | full=X empty=Y
    push_pattern = r'\[\s*(\d+)\]\s*\[OUT_FIFO_PUSH\]\s*output_idx=(\d+)\s*\|\s*data:\s*\[(.*?)\]\s*\|\s*full=(\d)\s*empty=(\d)'
    # [timestamp] [OUT_FIFO_POP] data: [...] | full=X empty=Y
    pop_pattern = r'\[\s*(\d+)\]\s*\[OUT_FIFO_POP\]\s*data:\s*\[(.*?)\]\s*\|\s*full=(\d)\s*empty=(\d)'

    # Temporary storage for combining step/din/mult
    pending_steps = {}  # key: (output_idx, acc_step), value: dict with i_cnt, b_cnt, timestamp

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Try OUT_FIFO_PUSH pattern (new format)
                match = re.search(push_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    output_idx = int(match.group(2))
                    data_str = match.group(3)
                    data = [int(x) for x in data_str.split()]
                    fifo_full = match.group(4) == '1'
                    fifo_empty = match.group(5) == '1'
                    fifo_pushes.append(PostIMCUFifoPush(
                        imce_coord=coord,
                        output_idx=output_idx,
                        data=data,
                        fifo_full=fifo_full,
                        fifo_empty=fifo_empty,
                        source="rtl",
                        timestamp=timestamp
                    ))
                    continue

                # Try OUT_FIFO_POP pattern (new format)
                match = re.search(pop_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    data_str = match.group(2)
                    data = [int(x) for x in data_str.split()]
                    fifo_full = match.group(3) == '1'
                    fifo_empty = match.group(4) == '1'
                    fifo_pops.append(PostIMCUFifoPop(
                        imce_coord=coord,
                        data=data,
                        fifo_full=fifo_full,
                        fifo_empty=fifo_empty,
                        source="rtl",
                        timestamp=timestamp
                    ))
                    # Also add as IMCUOutput for comparison purposes
                    outputs.append(IMCUOutput(
                        imce_coord=coord,
                        result=data,
                        source="rtl",
                        timestamp=timestamp
                    ))
                    continue

                # Try ACC_STEP pattern
                match = re.search(step_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    output_idx = int(match.group(2))
                    acc_step = int(match.group(3))
                    i_cnt = int(match.group(4))
                    b_cnt = int(match.group(5))
                    key = (output_idx, acc_step)
                    pending_steps[key] = {
                        'timestamp': timestamp,
                        'i_cnt': i_cnt,
                        'b_cnt': b_cnt,
                        'din': None,
                        'bp_mult': None
                    }
                    continue

                # Try ACC_DIN pattern
                match = re.search(din_pattern, line)
                if match:
                    output_idx = int(match.group(2))
                    acc_step = int(match.group(3))
                    din_str = match.group(4)
                    din = [int(x) for x in din_str.split()]
                    key = (output_idx, acc_step)
                    if key in pending_steps:
                        pending_steps[key]['din'] = din
                    continue

                # Try ACC_MULT pattern
                match = re.search(mult_pattern, line)
                if match:
                    output_idx = int(match.group(2))
                    acc_step = int(match.group(3))
                    mult_str = match.group(4)
                    bp_mult = [int(x) for x in mult_str.split()]
                    key = (output_idx, acc_step)
                    if key in pending_steps:
                        pending_steps[key]['bp_mult'] = bp_mult
                        # Now we have all data, create the AccStep
                        step_data = pending_steps[key]
                        if step_data['din'] is not None:
                            acc_steps.append(PostIMCUAccStep(
                                imce_coord=coord,
                                output_idx=output_idx,
                                acc_step=acc_step,
                                din=step_data['din'],
                                bp_mult=bp_mult,
                                i_cnt=step_data['i_cnt'],
                                b_cnt=step_data['b_cnt'],
                                source="rtl",
                                timestamp=step_data['timestamp']
                            ))
                    continue

                # Try IMCU_OUT pattern (legacy format)
                match = re.search(out_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    output_idx = int(match.group(2))
                    result_str = match.group(3)
                    result = [int(x) for x in result_str.split()]
                    outputs.append(IMCUOutput(
                        imce_coord=coord,
                        result=result,
                        source="rtl",
                        timestamp=timestamp
                    ))
    except FileNotFoundError:
        pass

    return acc_steps, outputs, fifo_pushes, fifo_pops


def parse_rtl_vpu_input(log_path: str, coord: Tuple[int, int]) -> List[VPUInput]:
    """Parse RTL VPU log for MM_QUANT inputs (with thresholds)"""
    inputs = []

    # Pattern for QUANT thresholds: [timestamp] [QUANT] thresholds: [...]
    thresh_pattern = r'\[\s*(\d+)\]\s*\[QUANT\]\s*thresholds:\s*\[(.*?)\]'
    # Pattern for MM_QUANT: [timestamp] MM_QUANT | input: [...] | min: X | max: Y | output: [...]
    quant_pattern = r'\[\s*(\d+)\]\s*MM_QUANT\s*\|\s*input:\s*\[(.*?)\]\s*\|'

    current_thresholds = None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Try threshold pattern
                match = re.search(thresh_pattern, line)
                if match:
                    thresh_str = match.group(2)
                    current_thresholds = [int(x.strip()) for x in thresh_str.split(',')]
                    continue

                # Try MM_QUANT pattern (input part)
                match = re.search(quant_pattern, line)
                if match:
                    timestamp = int(match.group(1))
                    input_str = match.group(2)
                    input_values = [int(x) for x in input_str.split()]
                    inputs.append(VPUInput(
                        imce_coord=coord,
                        input_values=input_values,
                        thresholds=current_thresholds if current_thresholds else [],
                        source="rtl",
                        timestamp=timestamp
                    ))
    except FileNotFoundError:
        pass

    return inputs


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
            elif 'u_post_imcu.log' in fname:
                imce_logs[coord]['post_imcu'] = fpath
            elif 'u_imcu_core.log' in fname and 'u_post_imcu' not in fname:
                imce_logs[coord]['imcu_core'] = fpath
            elif 'u_linebuffer.log' in fname and '.ctrl.log' not in fname:
                imce_logs[coord]['linebuffer'] = fpath

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
    parser.add_argument('--out', '-o', type=str, help='Output file path to save parsed data (text format)')
    # New options for detailed component parsing
    parser.add_argument('--parse-linebuffer', action='store_true', help='Parse linebuffer input/output')
    parser.add_argument('--parse-imcu', action='store_true', help='Parse IMCU core input')
    parser.add_argument('--parse-post-imcu', action='store_true', help='Parse post_imcu accumulation steps')
    parser.add_argument('--parse-vpu', action='store_true', help='Parse VPU input/output')
    parser.add_argument('--parse-all', action='store_true', help='Parse all component data')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of entries to show (default: 10)')

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
    rtl_linebuffer_in = {}
    rtl_linebuffer_out = {}
    rtl_linebuffer_cfg = {}
    rtl_imcu_in = {}
    rtl_post_imcu_acc = {}
    rtl_fifo_push = {}
    rtl_fifo_pop = {}
    rtl_vpu_in = {}

    parse_detailed = args.parse_linebuffer or args.parse_imcu or args.parse_post_imcu or args.parse_vpu or args.parse_all

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

            # Parse VPU inputs if requested
            if args.parse_vpu or args.parse_all:
                vpu_inputs = parse_rtl_vpu_input(logs['vpu'], coord)
                if vpu_inputs:
                    rtl_vpu_in[coord] = vpu_inputs

        if 'post_imcu' in logs:
            # Always parse IMCU outputs for comparison
            acc_steps, imcu_outputs, fifo_pushes, fifo_pops = parse_rtl_post_imcu(logs['post_imcu'], coord)
            if imcu_outputs:
                rtl_imcu[coord] = imcu_outputs
            if (args.parse_post_imcu or args.parse_all) and acc_steps:
                rtl_post_imcu_acc[coord] = acc_steps
            if (args.parse_post_imcu or args.parse_all) and fifo_pushes:
                rtl_fifo_push[coord] = fifo_pushes
            if (args.parse_post_imcu or args.parse_all) and fifo_pops:
                rtl_fifo_pop[coord] = fifo_pops

        # Parse linebuffer if requested
        if 'linebuffer' in logs and (args.parse_linebuffer or args.parse_all):
            lb_in, lb_out, lb_cfg = parse_rtl_linebuffer(logs['linebuffer'], coord)
            if lb_in:
                rtl_linebuffer_in[coord] = lb_in
            if lb_out:
                rtl_linebuffer_out[coord] = lb_out
            if lb_cfg:
                rtl_linebuffer_cfg[coord] = lb_cfg

        # Parse IMCU core input if requested
        if 'imcu_core' in logs and (args.parse_imcu or args.parse_all):
            imcu_inputs = parse_rtl_imcu_input(logs['imcu_core'], coord)
            if imcu_inputs:
                rtl_imcu_in[coord] = imcu_inputs

    print(f"  MM_QUANT: {sum(len(v) for v in rtl_mm_quant.values())} entries from {len(rtl_mm_quant)} IMCEs")
    print(f"  IMCU: {sum(len(v) for v in rtl_imcu.values())} entries from {len(rtl_imcu)} IMCEs")
    print(f"  VPU ops: {sum(len(v) for v in rtl_vpu.values())} entries from {len(rtl_vpu)} IMCEs")
    if parse_detailed:
        print(f"  Linebuffer IN: {sum(len(v) for v in rtl_linebuffer_in.values())} entries from {len(rtl_linebuffer_in)} IMCEs")
        print(f"  Linebuffer OUT: {sum(len(v) for v in rtl_linebuffer_out.values())} entries from {len(rtl_linebuffer_out)} IMCEs")
        print(f"  IMCU IN: {sum(len(v) for v in rtl_imcu_in.values())} entries from {len(rtl_imcu_in)} IMCEs")
        print(f"  Post-IMCU ACC: {sum(len(v) for v in rtl_post_imcu_acc.values())} entries from {len(rtl_post_imcu_acc)} IMCEs")
        print(f"  FIFO PUSH: {sum(len(v) for v in rtl_fifo_push.values())} entries from {len(rtl_fifo_push)} IMCEs")
        print(f"  FIFO POP: {sum(len(v) for v in rtl_fifo_pop.values())} entries from {len(rtl_fifo_pop)} IMCEs")
        print(f"  VPU IN: {sum(len(v) for v in rtl_vpu_in.values())} entries from {len(rtl_vpu_in)} IMCEs")

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
                print(f"  No RTL data (need to re-run RTL with IMCU_OUT logging)")

        print(f"\nTotal: {total_match} match, {total_mismatch} mismatch")

    # Display detailed parsed data if requested
    if args.parse_linebuffer or args.parse_all:
        print(f"\n{'='*80}")
        print("Linebuffer Data (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_linebuffer_cfg.keys()):
            cfgs = rtl_linebuffer_cfg[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}) Config: {len(cfgs)} entries")
            for i, cfg in enumerate(cfgs[:args.limit]):
                print(f"  [{i}] H={cfg.height} W={cfg.width} stride={cfg.stride} pad={cfg.pad} ksel={cfg.ksel}")

        for coord in sorted(rtl_linebuffer_in.keys()):
            items = rtl_linebuffer_in[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}) INPUT: {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                print(f"  [{i}] count={item.count} data={item.data[:34]}...")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

        for coord in sorted(rtl_linebuffer_out.keys()):
            items = rtl_linebuffer_out[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}) OUTPUT: {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                print(f"  [{i}] count={item.count} bitpos={item.bitpos} adata={item.adata[:34]}...")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

    if args.parse_imcu or args.parse_all:
        print(f"\n{'='*80}")
        print("IMCU Core Input Data (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_imcu_in.keys()):
            items = rtl_imcu_in[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                print(f"  [{i}] count={item.count} bitpos={item.bitpos} adata={item.adata[:34]}...")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

    if args.parse_post_imcu or args.parse_all:
        print(f"\n{'='*80}")
        print("Post-IMCU Accumulation Steps (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_post_imcu_acc.keys()):
            items = rtl_post_imcu_acc[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                din_str = str(item.din[:8]) if len(item.din) > 8 else str(item.din)
                print(f"  [{i}] out_idx={item.output_idx} step={item.acc_step} i_cnt={item.i_cnt} b_cnt={item.b_cnt} din={din_str}...")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

        print(f"\n{'='*80}")
        print("Post-IMCU FIFO PUSH (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_fifo_push.keys()):
            items = rtl_fifo_push[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                data_str = str(item.data[:8]) if len(item.data) > 8 else str(item.data)
                print(f"  [{i}] @{item.timestamp} out_idx={item.output_idx} data={data_str}... full={item.fifo_full} empty={item.fifo_empty}")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

        print(f"\n{'='*80}")
        print("Post-IMCU FIFO POP (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_fifo_pop.keys()):
            items = rtl_fifo_pop[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                data_str = str(item.data[:8]) if len(item.data) > 8 else str(item.data)
                print(f"  [{i}] @{item.timestamp} data={data_str}... full={item.fifo_full} empty={item.fifo_empty}")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

    if args.parse_vpu or args.parse_all:
        print(f"\n{'='*80}")
        print("VPU Input Data (RTL)")
        print(f"{'='*80}")
        for coord in sorted(rtl_vpu_in.keys()):
            items = rtl_vpu_in[coord]
            print(f"\nIMCE ({coord[0]}, {coord[1]}): {len(items)} entries")
            for i, item in enumerate(items[:args.limit]):
                input_str = str(item.input_values[:8]) if len(item.input_values) > 8 else str(item.input_values)
                print(f"  [{i}] input={input_str}...")
            if len(items) > args.limit:
                print(f"  ... and {len(items) - args.limit} more")

    # Save parsed data to file if --out is specified
    if args.out:
        out_path = args.out
        with open(out_path, 'w') as f:
            # Write Python IMCU outputs
            for coord in sorted(py_imcu.keys()):
                f.write(f"=== Python IMCE ({coord[0]}, {coord[1]}) IMCU_OUT ===\n")
                for i, item in enumerate(py_imcu[coord]):
                    f.write(f"[{i}] {item.result[:16]}\n")
                f.write("\n")

            # Write RTL IMCU outputs
            for coord in sorted(rtl_imcu.keys()):
                f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) IMCU_OUT ===\n")
                for i, item in enumerate(rtl_imcu[coord]):
                    f.write(f"[{i}] {item.result[:16]}\n")
                f.write("\n")

            # Write Python MM_QUANT outputs
            for coord in sorted(py_mm_quant.keys()):
                f.write(f"=== Python IMCE ({coord[0]}, {coord[1]}) MM_QUANT ===\n")
                for i, item in enumerate(py_mm_quant[coord]):
                    f.write(f"[{i}] output: {item.output_values}\n")
                f.write("\n")

            # Write RTL MM_QUANT outputs
            for coord in sorted(rtl_mm_quant.keys()):
                f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) MM_QUANT ===\n")
                for i, item in enumerate(rtl_mm_quant[coord]):
                    f.write(f"[{i}] output: {item.output_values}\n")
                f.write("\n")

            # Write detailed component data if parsed
            if rtl_linebuffer_cfg:
                for coord in sorted(rtl_linebuffer_cfg.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Linebuffer CONFIG ===\n")
                    for i, cfg in enumerate(rtl_linebuffer_cfg[coord]):
                        f.write(f"[{i}] H={cfg.height} W={cfg.width} stride={cfg.stride} pad={cfg.pad} ksel={cfg.ksel}\n")
                    f.write("\n")

            if rtl_linebuffer_in:
                for coord in sorted(rtl_linebuffer_in.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Linebuffer INPUT ===\n")
                    for i, item in enumerate(rtl_linebuffer_in[coord]):
                        f.write(f"[{i}] count={item.count} data={item.data}\n")
                    f.write("\n")

            if rtl_linebuffer_out:
                for coord in sorted(rtl_linebuffer_out.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Linebuffer OUTPUT ===\n")
                    for i, item in enumerate(rtl_linebuffer_out[coord]):
                        f.write(f"[{i}] count={item.count} bitpos={item.bitpos} adata={item.adata}\n")
                    f.write("\n")

            if rtl_imcu_in:
                for coord in sorted(rtl_imcu_in.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) IMCU_IN ===\n")
                    for i, item in enumerate(rtl_imcu_in[coord]):
                        f.write(f"[{i}] count={item.count} bitpos={item.bitpos} adata={item.adata}\n")
                    f.write("\n")

            if rtl_post_imcu_acc:
                for coord in sorted(rtl_post_imcu_acc.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Post-IMCU ACC_STEP ===\n")
                    for i, item in enumerate(rtl_post_imcu_acc[coord]):
                        f.write(f"[{i}] out_idx={item.output_idx} step={item.acc_step} i_cnt={item.i_cnt} b_cnt={item.b_cnt}\n")
                        f.write(f"    din: {item.din[:16]}\n")
                        f.write(f"    bp_mult: {item.bp_mult[:16]}\n")
                    f.write("\n")

            if rtl_fifo_push:
                for coord in sorted(rtl_fifo_push.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Post-IMCU FIFO_PUSH ===\n")
                    for i, item in enumerate(rtl_fifo_push[coord]):
                        f.write(f"[{i}] @{item.timestamp} out_idx={item.output_idx} full={item.fifo_full} empty={item.fifo_empty}\n")
                        f.write(f"    data: {item.data[:16]}\n")
                    f.write("\n")

            if rtl_fifo_pop:
                for coord in sorted(rtl_fifo_pop.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) Post-IMCU FIFO_POP ===\n")
                    for i, item in enumerate(rtl_fifo_pop[coord]):
                        f.write(f"[{i}] @{item.timestamp} full={item.fifo_full} empty={item.fifo_empty}\n")
                        f.write(f"    data: {item.data[:16]}\n")
                    f.write("\n")

            if rtl_vpu_in:
                for coord in sorted(rtl_vpu_in.keys()):
                    f.write(f"=== RTL IMCE ({coord[0]}, {coord[1]}) VPU_IN (MM_QUANT input) ===\n")
                    for i, item in enumerate(rtl_vpu_in[coord]):
                        f.write(f"[{i}] input: {item.input_values}\n")
                        f.write(f"    thresholds: {item.thresholds}\n")
                    f.write("\n")

        print(f"\nSaved parsed data to {out_path}")


if __name__ == '__main__':
    main()
