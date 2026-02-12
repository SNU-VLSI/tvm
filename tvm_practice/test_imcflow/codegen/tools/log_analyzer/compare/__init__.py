"""Compare pysim and RTL simulation outputs.

Usage::

    python -m log_analyzer.compare <test_dir> --compare-all [-v]
"""

from .models import OpRecord, CompareResult
from .extractor import extract_records
from .rtl_compat import extract_rtl_records, find_rtl_logs
from .comparator import compare_all, compare_records, format_results

__all__ = [
    "OpRecord",
    "CompareResult",
    "extract_records",
    "extract_rtl_records",
    "find_rtl_logs",
    "compare_all",
    "compare_records",
    "format_results",
]
