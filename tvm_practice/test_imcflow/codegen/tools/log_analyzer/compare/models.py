"""Data models for pysim/RTL comparison."""

from dataclasses import dataclass, field


@dataclass
class OpRecord:
    """A single operation record from either pysim or RTL.

    Both extractors (structured-log and RTL-compat) produce these,
    so the comparator works source-agnostically.
    """

    imce_coord: tuple  # (row, col)
    event: str  # Normalized: "MM_QUANT", "IMCU_OUTPUT", "MULTL", "ADD", etc.
    index: int  # Sequence index within (coord, event)
    timestamp: int  # Simulation step / VCS time
    fields: dict  # Event-specific data
    source: str = ""  # "pysim" or "rtl"


@dataclass
class CompareResult:
    """Result of comparing one operation type between pysim and RTL."""

    event: str
    imce_coord: tuple
    total: int  # max(pysim_count, rtl_count)
    matches: int
    mismatches: list = field(default_factory=list)  # [{index, pysim, rtl, diff_fields}]
    pysim_count: int = 0
    rtl_count: int = 0
