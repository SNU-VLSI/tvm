"""Stall analysis for finding nodes stuck at end of simulation.

Scans all inode/imce hazard log files for STALL_START/STALL_END pairs.
Any STALL_START without a matching STALL_END at EOF is reported as an
active stall — the node was still blocked when the simulation ended.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from log_analyzer.log_format import parse_file, LogEntry
from log_analyzer.models import StallInfo

# Events we need to track stall state
STALL_ANALYSIS_EVENTS = {
    "EX_STALL_START", "EX_STALL_END",
    "WB_STALL_START", "WB_STALL_END",
    "IF_STALL_START", "IF_STALL_END",
}

# Events for IMCE ctrl_pl logs (stall + execute in one file)
_IMCE_CTRL_PL_EVENTS = {
    "STALL_START", "STALL_END", "EXECUTE",
}

# Events for inode ex_stage logs (to get PC info)
_INODE_EX_EVENTS = {
    "EX_START", "EX_END",
}

# Regex for extracting node coordinates from log filenames
# Matches: core_row_0_.core_col_3_.inode  or  core_row_2_.core_col_1_.imce_node
_NODE_RE = re.compile(
    r"core_row_(\d+)_\.core_col_(\d+)_\.(inode|imce_node)"
)

# Hazard log file identifiers per node type
_HAZARD_FILE_KEYWORDS = {
    "inode": "hazard_control",
    "imce": "hazard_detector",
}

# Execution log file identifiers per node type (for PC info)
_EXEC_FILE_KEYWORDS = {
    "inode": "ex_stage",
    "imce": "ctrl_pl",
}


class StallAnalyzer:
    """Finds nodes that are still stalled at end of simulation."""

    def __init__(self, log_dir, verbose: bool = False):
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.stalls: list[StallInfo] = []
        self._node_last_times: dict[str, int] = {}
        self._total_nodes: int = 0

    # ------------------------------------------------------------------
    # Node discovery
    # ------------------------------------------------------------------

    def _discover_nodes(self) -> list[tuple[str, str, int, int, Path, Optional[Path]]]:
        """Auto-discover all inode/imce nodes from log filenames.

        Returns:
            List of (node_id, node_type, row, col, hazard_log_path, exec_log_path).
        """
        # First, find all unique (row, col, node_type) from filenames
        nodes: dict[tuple[int, int, str], None] = {}
        for log_file in self.log_dir.glob("*.log"):
            m = _NODE_RE.search(log_file.name)
            if m:
                row, col = int(m.group(1)), int(m.group(2))
                raw_type = m.group(3)
                node_type = "imce" if raw_type == "imce_node" else "inode"
                nodes[(row, col, node_type)] = None

        # Collect all log files once
        all_logs = list(self.log_dir.glob("*.log"))

        # Now find the hazard log and exec log files for each node
        result = []
        for (row, col, node_type) in sorted(nodes):
            hazard_keyword = _HAZARD_FILE_KEYWORDS[node_type]
            exec_keyword = _EXEC_FILE_KEYWORDS[node_type]
            raw_type = "imce_node" if node_type == "imce" else "inode"
            # Build the expected filename pattern (matches: core_row_0_.core_col_1_.imce_node)
            prefix = f"core_row_{row}_.core_col_{col}_.{raw_type}"
            hazard_path = None
            exec_path = None
            for log_file in all_logs:
                name = log_file.name
                if prefix not in name:
                    continue
                if hazard_keyword in name:
                    hazard_path = log_file
                # For inode: match "ex_stage.log" exactly (not "ex_stage.u_recv_fifo.log")
                # For imce: match "ctrl_pl.log" exactly
                if exec_keyword in name and name.endswith(f"{exec_keyword}.log"):
                    exec_path = log_file

            if hazard_path is None:
                if self.verbose:
                    print(f"  Warning: no {hazard_keyword} log for {node_type}({row},{col})",
                          file=sys.stderr)
                continue

            if exec_path is None and self.verbose:
                print(f"  Warning: no {exec_keyword} log for {node_type}({row},{col})",
                      file=sys.stderr)

            node_id = f"{node_type}_{row}_{col}"
            result.append((node_id, node_type, row, col, hazard_path, exec_path))

        return result

    # ------------------------------------------------------------------
    # Per-node parsing
    # ------------------------------------------------------------------

    def _get_pc_at_time(self, exec_path: Optional[Path], node_type: str,
                        timestamp: int) -> tuple[Optional[int], Optional[str]]:
        """Find the PC and opcode at or just before the given timestamp.

        For IMCE: reads ctrl_pl.log EXECUTE events (has pc= field).
        For inode: reads ex_stage.log EX_START events (has pc= field).

        Returns (pc, opcode) or (None, None) if not found.
        """
        if exec_path is None:
            return None, None

        if node_type == "imce":
            events = _IMCE_CTRL_PL_EVENTS
            exec_event = "EXECUTE"
        else:
            events = _INODE_EX_EVENTS
            exec_event = "EX_START"

        try:
            entries = parse_file(exec_path, events=events)
        except Exception:
            return None, None

        # Find the last EXECUTE/EX_START at or before the stall timestamp
        last_pc = None
        last_opcode = None
        for entry in entries:
            if entry.event != exec_event:
                continue
            if entry.time > timestamp:
                break
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            if "pc" in payload:
                last_pc = payload["pc"]
                last_opcode = str(payload.get("opcode", ""))
            elif "opcode" in payload:
                last_opcode = str(payload["opcode"])

        return last_pc, last_opcode

    def _parse_node(self, path: Path, exec_path: Optional[Path],
                    node_id: str, node_type: str,
                    row: int, col: int) -> list[StallInfo]:
        """Parse a single hazard log and return active stalls at EOF."""
        try:
            entries = parse_file(path, events=STALL_ANALYSIS_EVENTS)
        except Exception as e:
            if self.verbose:
                print(f"  Error parsing {path.name}: {e}", file=sys.stderr)
            return []

        # Track last event time for the node
        if entries:
            self._node_last_times[node_id] = entries[-1].time

        # Track open stalls: (stall_prefix, reason) -> LogEntry
        open_stalls: dict[tuple[str, str], LogEntry] = {}

        for entry in entries:
            event = entry.event
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            reason = str(payload.get("reason", "UNKNOWN"))

            if event.endswith("_STALL_START"):
                # e.g. "EX_STALL_START" -> prefix "EX_STALL"
                prefix = event.rsplit("_", 1)[0]  # "EX_STALL"
                open_stalls[(prefix, reason)] = entry

            elif event.endswith("_STALL_END"):
                prefix = event.rsplit("_", 1)[0]  # "EX_STALL"
                # Close the matching stall
                open_stalls.pop((prefix, reason), None)

        # Filter out propagated stalls — these are derived from another
        # pipeline stage's stall and are not root causes.
        _PROPAGATED_REASONS = {"PROPAGATED_FROM_MEM"}
        open_stalls = {
            k: v for k, v in open_stalls.items()
            if k[1] not in _PROPAGATED_REASONS
        }

        # Convert remaining open stalls to StallInfo, enriching with PC info
        active = []
        for (stall_type, reason), entry in open_stalls.items():
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            pc, opcode = self._get_pc_at_time(exec_path, node_type, entry.time)
            active.append(StallInfo(
                node=node_id,
                node_type=node_type,
                row=row,
                col=col,
                stall_type=stall_type,
                reason=reason,
                start_time=entry.time,
                payload=payload,
                source_file=path.name,
                pc=pc,
                opcode=opcode,
            ))

        return active

    # ------------------------------------------------------------------
    # Parallel / sequential orchestration
    # ------------------------------------------------------------------

    def _parse_all_parallel(
        self, discovered: list[tuple[str, str, int, int, Path, Optional[Path]]], max_workers: int = 4
    ):
        """Parse all nodes in parallel using thread pool."""
        def _parse_one(item):
            node_id, node_type, row, col, path, exec_path = item
            return self._parse_node(path, exec_path, node_id, node_type, row, col)

        workers = min(max_workers, len(discovered))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_parse_one, discovered))

        for stalls in results:
            self.stalls.extend(stalls)

    def parse_all(self):
        """Discover all nodes and find active stalls."""
        discovered = self._discover_nodes()
        self._total_nodes = len(discovered)

        if self.verbose:
            print(f"  Discovered {len(discovered)} nodes", file=sys.stderr)

        if not discovered:
            return

        if len(discovered) > 1 and not getattr(self, '_sequential', False):
            self._parse_all_parallel(discovered)
        else:
            for node_id, node_type, row, col, path, exec_path in discovered:
                stalls = self._parse_node(path, exec_path, node_id, node_type, row, col)
                self.stalls.extend(stalls)

        # Sort by start_time
        self.stalls.sort(key=lambda s: s.start_time)

    # ------------------------------------------------------------------
    # Summary / reporting helpers
    # ------------------------------------------------------------------

    def get_stall_summary(self) -> dict:
        """Return a summary dict with counts by reason and node_type."""
        by_reason: dict[str, int] = {}
        by_node_type: dict[str, int] = {}
        stalled_nodes: set[str] = set()

        for s in self.stalls:
            by_reason[s.reason] = by_reason.get(s.reason, 0) + 1
            by_node_type[s.node_type] = by_node_type.get(s.node_type, 0) + 1
            stalled_nodes.add(s.node)

        return {
            "total_nodes": self._total_nodes,
            "stalled_nodes": len(stalled_nodes),
            "total_stalls": len(self.stalls),
            "by_reason": dict(sorted(by_reason.items(), key=lambda x: -x[1])),
            "by_node_type": by_node_type,
        }
