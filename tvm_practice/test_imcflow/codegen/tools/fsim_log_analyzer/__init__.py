"""FSIM Log Analyzer - A tool for parsing, analyzing, and monitoring FSIM log files."""

from .models import FileStatus, PacketEvent, PacketTrace, SyncEvent, StallInfo
from .keyboard import KeyboardHandler
from .monitor import LogMonitor, DebugTestDirectory
from .packet import PacketAnalyzer
from .sync_trace import SyncTraceAnalyzer
from .recv_analysis import (
    count_recv_before_step,
    expand_row_pattern,
    parse_expected_patterns_from_log,
    compare_recv_patterns,
)
from .log_format import LogEntry, ParseError, parse_payload, parse_line, parse_file
from .fast_search import fast_parse_file, fast_parse_files, grep_file
from .stall_analysis import StallAnalyzer
from .utils import parse_patterns, split_log_by_simulation

__all__ = [
    "FileStatus",
    "PacketEvent",
    "PacketTrace",
    "SyncEvent",
    "KeyboardHandler",
    "LogMonitor",
    "DebugTestDirectory",
    "PacketAnalyzer",
    "SyncTraceAnalyzer",
    "count_recv_before_step",
    "expand_row_pattern",
    "parse_expected_patterns_from_log",
    "compare_recv_patterns",
    "LogEntry",
    "ParseError",
    "parse_payload",
    "parse_line",
    "parse_file",
    "fast_parse_file",
    "fast_parse_files",
    "grep_file",
    "StallInfo",
    "StallAnalyzer",
    "parse_patterns",
    "split_log_by_simulation",
]
