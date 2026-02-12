"""Parse RTL ``$display`` logs into OpRecord.

RTL Verilog modules still use ``$display`` format, not the structured
``[time] | EVENT | {payload}`` format.  This module provides regex-based
parsers that produce the same :class:`OpRecord` the structured extractor
produces, so the comparator works identically with either source.

Once the RTL Verilog is migrated to structured format, this file can be
replaced by a second call to :func:`extractor.extract_records`.
"""

import os
import re
from collections import defaultdict

from .models import OpRecord


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_COORD_PATH_RE = re.compile(r"core_row_(\d+)_\.core_col_(\d+)_\.imce_node")


def extract_imce_coord_from_path(path):
    """Extract ``(row, col)`` from an RTL log file path."""
    m = _COORD_PATH_RE.search(path)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def find_rtl_logs(base_dir):
    """Discover RTL fsim log files grouped by IMCE coordinate.

    Returns
    -------
    dict[(int,int), dict[str, str]]
        ``{(row, col): {"vpu": path, "post_imcu": path, ...}}``
    """
    fsim_dir = os.path.join(base_dir, "logs", "rtl_runner", "fsim_logs")
    imce_logs = defaultdict(dict)

    if not os.path.exists(fsim_dir):
        return dict(imce_logs)

    for fname in os.listdir(fsim_dir):
        fpath = os.path.join(fsim_dir, fname)
        coord = extract_imce_coord_from_path(fname)
        if not coord:
            continue
        if "u_vpu.log" in fname:
            imce_logs[coord]["vpu"] = fpath
        elif "u_erf.log" in fname:
            imce_logs[coord]["erf"] = fpath
        elif "u_post_imcu.log" in fname:
            imce_logs[coord]["post_imcu"] = fpath
        elif "u_imcu_core.log" in fname and "u_post_imcu" not in fname:
            imce_logs[coord]["imcu_core"] = fpath
        elif "u_linebuffer.log" in fname and ".ctrl.log" not in fname:
            imce_logs[coord]["linebuffer"] = fpath

    return dict(imce_logs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int_list(s):
    """Parse a space-separated string of integers, returning ``[]`` on failure."""
    try:
        return [int(x) for x in s.split()]
    except ValueError:
        return []


def _read_lines(path):
    try:
        with open(path, "r") as fh:
            return fh.readlines()
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# MM_QUANT
# ---------------------------------------------------------------------------

_MM_QUANT_RE = re.compile(
    r"\[\s*(\d+)\]\s*MM_QUANT\s*\|"
    r"\s*input:\s*\[(.*?)\]"
    r"\s*\|\s*min:\s*(-?\d+)"
    r"\s*\|\s*max:\s*(-?\d+)"
    r"\s*\|\s*output:\s*\[(.*?)\]"
)


def parse_rtl_mm_quant(log_path, coord):
    """Parse ``[time] MM_QUANT | input: [...] | …`` → list[OpRecord]."""
    records = []
    idx = 0
    for line in _read_lines(log_path):
        if "MM_QUANT" not in line or "input:" not in line:
            continue
        m = _MM_QUANT_RE.search(line)
        if not m:
            continue
        records.append(OpRecord(
            imce_coord=coord,
            event="MM_QUANT",
            index=idx,
            timestamp=int(m.group(1)),
            fields={
                "input": _safe_int_list(m.group(2)),
                "min": int(m.group(3)),
                "max": int(m.group(4)),
                "output": _safe_int_list(m.group(5)),
            },
            source="rtl",
        ))
        idx += 1
    return records


# ---------------------------------------------------------------------------
# IMCU_OUTPUT  (from post_imcu log)
# ---------------------------------------------------------------------------

_IMCU_OUT_NEW_RE = re.compile(
    r"\[\s*(\d+)\]\s*\[IMCU_OUT\]\s*output_idx=(\d+)\s*\|\s*result:\s*\[(.*?)\]"
)
_IMCU_OUT_LEGACY_RE = re.compile(
    r"\[\s*(\d+)\]\s*PIMC_OUT\s*\|\s*result:\s*\[(.*?)\]"
)
_FIFO_POP_RE = re.compile(
    r"\[\s*(\d+)\]\s*\[OUT_FIFO_POP\]\s*data:\s*\[(.*?)\]"
)


def parse_rtl_pimc_output(log_path, coord):
    """Parse IMCU_OUT / PIMC_OUT / OUT_FIFO_POP → list[OpRecord]."""
    records = []
    idx = 0
    for line in _read_lines(log_path):
        result = None
        timestamp = None

        if "IMCU_OUT" in line:
            m = _IMCU_OUT_NEW_RE.search(line)
            if m:
                timestamp = int(m.group(1))
                result = _safe_int_list(m.group(3))
        elif "PIMC_OUT" in line:
            m = _IMCU_OUT_LEGACY_RE.search(line)
            if m:
                timestamp = int(m.group(1))
                result = _safe_int_list(m.group(2))
        elif "OUT_FIFO_POP" in line:
            m = _FIFO_POP_RE.search(line)
            if m:
                timestamp = int(m.group(1))
                result = _safe_int_list(m.group(2))

        if result is not None:
            records.append(OpRecord(
                imce_coord=coord,
                event="IMCU_OUTPUT",
                index=idx,
                timestamp=timestamp,
                fields={"result": result},
                source="rtl",
            ))
            idx += 1
    return records


# ---------------------------------------------------------------------------
# VPU ops  (MULTL, ADD, SUB)
# ---------------------------------------------------------------------------

_MULTL_RE = re.compile(
    r"\[\s*(\d+)\]\s*MULTL\s*\|"
    r"\s*opA:\s*\[(.*?)\]"
    r"\s*\|\s*opB:\s*\[(.*?)\]"
    r"\s*\|.*?\|\s*result:\s*\[(.*?)\]"
)

_ADD_RE = re.compile(
    r"\[\s*(\d+)\]\s*ADD\s*\|"
    r"\s*opA:\s*\[(.*?)\]"
    r"\s*\|\s*opB:\s*\[(.*?)\]"
    r"\s*\|\s*result:\s*\[(.*?)\]"
)

_SUB_RE = re.compile(
    r"\[\s*(\d+)\]\s*SUB\s*\|"
    r"\s*opA:\s*\[(.*?)\]"
    r"\s*\|\s*opB:\s*\[(.*?)\]"
    r"\s*\|\s*result:\s*\[(.*?)\]"
)


def parse_rtl_vpu_ops(log_path, coord):
    """Parse MULTL / ADD / SUB lines from VPU log → list[OpRecord]."""
    records = []
    counters = defaultdict(int)

    for line in _read_lines(log_path):
        for pattern, op_name in [
            (_MULTL_RE, "MULTL"),
            (_ADD_RE, "ADD"),
            (_SUB_RE, "SUB"),
        ]:
            m = pattern.search(line)
            if not m:
                continue
            idx = counters[op_name]
            counters[op_name] += 1
            records.append(OpRecord(
                imce_coord=coord,
                event=op_name,
                index=idx,
                timestamp=int(m.group(1)),
                fields={
                    "x": _safe_int_list(m.group(2)),
                    "y": _safe_int_list(m.group(3)),
                    "result": _safe_int_list(m.group(4)),
                },
                source="rtl",
            ))
            break  # Only one pattern matches per line

    return records


# ---------------------------------------------------------------------------
# DWCONV
# ---------------------------------------------------------------------------

_DWCONV_RESULT_RE = re.compile(
    r"\[\s*(\d+)\]\s*DWCONV RESULT\s*\|"
    r"\s*bshr_sel:\s*(\d+)"
    r"\s*\|\s*shift_amt:\s*(\d+)"
    r"\s*\|\s*opA_raw:\s*\[([^\]]*)\]"
    r"\s*\|\s*weight\[0\]:\s*\[([^\]]*)\]"
    r"\s*\|\s*mac:\s*\[([^\]]*)\]"
    r"\s*\|\s*mac_shift:\s*\[([^\]]*)\]"
    r"\s*\|\s*acc:\s*\[([^\]]*)\]"
    r"\s*\|\s*result:\s*\[([^\]]*)\]"
)

_DWCONV_ACC_RE = re.compile(
    r"\[\s*(\d+)\]\s*DWCONV ACC\s+\|"
    r"\s*bshr_sel:\s*(\d+)"
    r"\s*\|\s*shift_amt:\s*(\d+)"
    r"\s*\|\s*opA_raw:\s*\[([^\]]*)\]"
    r"\s*\|\s*weight\[0\]:\s*\[([^\]]*)\]"
    r"\s*\|\s*mac:\s*\[([^\]]*)\]"
    r"\s*\|\s*mac_shift:\s*\[([^\]]*)\]"
    r"\s*\|\s*acc:\s*\[([^\]]*)\]"
)


def parse_rtl_dwconv(log_path, coord):
    """Parse ``DWCONV RESULT`` / ``DWCONV ACC`` lines → list[OpRecord].

    Only RESULT entries are emitted by default (they have ``result`` field).
    ACC entries are included too (with empty ``result``) for detailed debug.
    """
    records = []
    idx = 0
    for line in _read_lines(log_path):
        if "DWCONV" not in line:
            continue

        # Try RESULT first (more specific)
        m = _DWCONV_RESULT_RE.search(line)
        if m:
            records.append(OpRecord(
                imce_coord=coord,
                event="DWCONV",
                index=idx,
                timestamp=int(m.group(1)),
                fields={
                    "bshr_sel": int(m.group(2)),
                    "shift_amt": int(m.group(3)),
                    "inputs": _safe_int_list(m.group(4)),
                    "weights": _safe_int_list(m.group(5)),
                    "mac": _safe_int_list(m.group(6)),
                    "mac_shift": _safe_int_list(m.group(7)),
                    "acc": _safe_int_list(m.group(8)),
                    "result": _safe_int_list(m.group(9)),
                    "stage": "result",
                },
                source="rtl",
            ))
            idx += 1
            continue

        # Try ACC
        m = _DWCONV_ACC_RE.search(line)
        if m:
            records.append(OpRecord(
                imce_coord=coord,
                event="DWCONV",
                index=idx,
                timestamp=int(m.group(1)),
                fields={
                    "bshr_sel": int(m.group(2)),
                    "shift_amt": int(m.group(3)),
                    "inputs": _safe_int_list(m.group(4)),
                    "weights": _safe_int_list(m.group(5)),
                    "mac": _safe_int_list(m.group(6)),
                    "mac_shift": _safe_int_list(m.group(7)),
                    "acc": _safe_int_list(m.group(8)),
                    "result": [],
                    "stage": "acc",
                },
                source="rtl",
            ))
            idx += 1

    return records


# ---------------------------------------------------------------------------
# Linebuffer
# ---------------------------------------------------------------------------

_LBUF_INPUT_RE = re.compile(
    r"\[\s*(\d+)\]\s*\[LBUF_HS\]\s*INPUT_HS\s*\|"
    r"\s*count=(\d+)\s*\|\s*data=(0x[0-9a-fA-F]+)"
)


def parse_rtl_linebuffer(log_path, coord):
    """Parse ``[LBUF_HS] INPUT_HS`` lines → list[OpRecord]."""
    records = []
    idx = 0
    for line in _read_lines(log_path):
        if "INPUT_HS" not in line:
            continue
        m = _LBUF_INPUT_RE.search(line)
        if not m:
            continue
        records.append(OpRecord(
            imce_coord=coord,
            event="LBUF_INPUT",
            index=idx,
            timestamp=int(m.group(1)),
            fields={
                "count": int(m.group(2)),
                "data": m.group(3),
            },
            source="rtl",
        ))
        idx += 1
    return records


# ---------------------------------------------------------------------------
# Aggregate extraction
# ---------------------------------------------------------------------------

def extract_rtl_records(base_dir, coord_filter=None):
    """Extract all RTL records from fsim logs.

    Returns
    -------
    dict[(coord, event), list[OpRecord]]
        Same shape as :func:`extractor.extract_records`.
    """
    rtl_logs = find_rtl_logs(base_dir)
    grouped = defaultdict(list)

    for coord, logs in rtl_logs.items():
        if coord_filter and coord != coord_filter:
            continue

        if "vpu" in logs:
            for rec in parse_rtl_mm_quant(logs["vpu"], coord):
                grouped[(coord, "MM_QUANT")].append(rec)
            for rec in parse_rtl_vpu_ops(logs["vpu"], coord):
                grouped[(coord, rec.event)].append(rec)
            for rec in parse_rtl_dwconv(logs["vpu"], coord):
                grouped[(coord, "DWCONV")].append(rec)

        if "post_imcu" in logs:
            for rec in parse_rtl_pimc_output(logs["post_imcu"], coord):
                grouped[(coord, "IMCU_OUTPUT")].append(rec)

        if "linebuffer" in logs:
            for rec in parse_rtl_linebuffer(logs["linebuffer"], coord):
                grouped[(coord, "LBUF_INPUT")].append(rec)

    return dict(grouped)
