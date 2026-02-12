"""Extract comparison-relevant events from structured logs into OpRecords.

Uses log_analyzer.log_format.parse_file() for pysim logs (and eventually
RTL logs once Verilog $display is migrated to structured format).
"""

import re
from collections import defaultdict

from ..log_format import parse_file
from .models import OpRecord

# Events the pysim emits that are relevant for comparison.
COMPARE_EVENTS = {
    "OP_MM_QUANT",
    "IMCU_OUTPUT",
    "OP_ADD",
    "OP_SUB",
    "OP_MULTL",
    "OP_MULTH",
    "OP_DWCONV",
    "LBUF_INPUT",
    "POST_IMCU",
}

# OP_MM_QUANT -> MM_QUANT, OP_ADD -> ADD, etc.
# RTL logs already use the short name.
_OP_PREFIX = "OP_"


def normalize_event(event: str) -> str:
    """Strip ``OP_`` prefix so pysim and RTL names align."""
    if event.startswith(_OP_PREFIX):
        return event[len(_OP_PREFIX) :]
    return event


_COORD_RE = re.compile(r"IMCE\.(\d+)\.(\d+)")


def parse_imce_coord(name):
    """Parse ``'IMCE.3.2'`` into ``(3, 2)``."""
    m = _COORD_RE.search(str(name))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def extract_records(log_path, source="pysim", events=None):
    """Parse a structured log file and return OpRecords grouped by ``(coord, event)``.

    Parameters
    ----------
    log_path : str
        Path to structured log file (pysim ``now.debug.log``).
    source : str
        Label stored in each OpRecord (``"pysim"`` or ``"rtl"``).
    events : set[str] | None
        Event names to extract.  Defaults to :data:`COMPARE_EVENTS`.

    Returns
    -------
    dict[tuple, list[OpRecord]]
        Mapping ``(coord, normalized_event) -> [OpRecord, ...]``.
    """
    target_events = events if events is not None else COMPARE_EVENTS
    entries = parse_file(log_path, events=target_events)

    grouped = defaultdict(list)
    counters = defaultdict(int)

    for entry in entries:
        payload = entry.payload if isinstance(entry.payload, dict) else {}

        # Pysim q_inst / a_inst use "name", r_inst uses "node"
        name = payload.get("name") or payload.get("node") or ""
        coord = parse_imce_coord(name)
        if coord is None:
            continue

        norm = normalize_event(entry.event)

        # For DWCONV, only keep entries with stage == "result_valid"
        if norm == "DWCONV":
            stage = payload.get("stage", "")
            if str(stage) != "result_valid":
                continue

        key = (coord, norm)
        idx = counters[key]
        counters[key] += 1

        grouped[key].append(
            OpRecord(
                imce_coord=coord,
                event=norm,
                index=idx,
                timestamp=entry.time,
                fields=dict(payload),
                source=source,
            )
        )

    return dict(grouped)
