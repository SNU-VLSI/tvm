"""Synchronization trace analysis between nodes."""

import re
import sys
from pathlib import Path
from typing import Optional

from .models import SyncEvent


class SyncTraceAnalyzer:
    """Analyzes synchronization events (set_flag, standby) between nodes."""

    # Patterns for inode hazard_control log
    INODE_STANDBY_START = re.compile(
        r"\[\s*(\d+)\]\s+\[INODE_STALL\]\s+EX_STALL_START:\s+STANDBY_STALL\s*\|\s*(.+)"
    )
    INODE_STANDBY_END = re.compile(
        r"\[\s*(\d+)\]\s+\[INODE_STALL\]\s+EX_STALL_END:\s+STANDBY_STALL\s*\|\s*(.+)"
    )

    # Patterns for inode ex_stage log
    INODE_EX_OP_START = re.compile(
        r"\[\s*(\d+)\]\s+START\s*\|\s*(OP_\w+)\s*\|\s*(.+)"
    )
    INODE_EX_OP_END = re.compile(
        r"\[\s*(\d+)\]\s+END\s*\|\s*(OP_\w+)\s*\|\s*(.+)"
    )

    # Patterns for imce hazard_detector log
    IMCE_STANDBY_START = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_START:\s+STANDBY_STALL\s*\|\s*(.+)"
    )
    IMCE_STANDBY_END = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_END:\s+STANDBY_STALL\s*\|\s*(.+)"
    )
    IMCE_STEP_STALL_START = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_START:\s+STEP_STALL"
    )
    IMCE_STEP_STALL_END = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_END:\s+STEP_STALL\s*\|\s*(.+)"
    )
    IMCE_RECV_STALL_START = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_START:\s+RECV_FIFO_STALL\s*\|\s*(.+)"
    )
    IMCE_RECV_STALL_END = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_STALL\]\s+EX_STALL_END:\s+RECV_FIFO_STALL\s*\|\s*(.+)"
    )

    # Patterns for wb_stage log (SEND, SETFLAG)
    WB_SEND = re.compile(
        r"\[\s*(\d+)\]\s+SEND\s*\|\s*(.+)"
    )
    WB_SETFLAG = re.compile(
        r"\[\s*(\d+)\]\s+SETFLAG\s*\|\s*(.+)"
    )
    WB_POLICY_UPDATE = re.compile(
        r"\[\s*(\d+)\]\s+POLICY_UPDATE\s*\|\s*(.+)"
    )

    # Patterns for imce_ctrl log
    IMCE_CTRL_SETFLAG = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_HS\]\s+(SETFLAG_SUCCESS|FLAG_SET)\s*\|\s*(.+)"
    )
    IMCE_CTRL_RECV = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_HS\]\s+RECV_SUCCESS\s*\|\s*(.+)"
    )
    IMCE_CTRL_SEND = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_HS\]\s+SEND_SUCCESS\s*\|\s*(.+)"
    )
    IMCE_CTRL_STEP = re.compile(
        r"\[\s*(\d+)\]\s+\[IMCE_HS\]\s+STEP_SUCCESS\s*\|\s*(.+)"
    )

    def __init__(self, log_dir: Path, nodes: list[str], verbose: bool = False):
        """
        Initialize sync trace analyzer.

        Args:
            log_dir: Directory containing fsim log files
            nodes: List of node identifiers (e.g., ["inode_0_0", "imce_3_4"])
            verbose: Print detailed parsing information
        """
        self.log_dir = Path(log_dir)
        self.nodes = [n.lower().replace(".", "_") for n in nodes]
        self.verbose = verbose
        self.events: list[SyncEvent] = []

    def _find_log_files_for_node(self, node: str) -> dict[str, Path]:
        """Find relevant log files for a node."""
        files = {}
        node_lower = node.lower()

        # Determine node type
        if "inode" in node_lower:
            node_type = "inode"
            match = re.search(r'inode[_.]?(\d+)[_.]?(\d+)', node_lower)
        else:
            node_type = "imce"
            match = re.search(r'imce[_.]?(\d+)[_.]?(\d+)', node_lower)

        if not match:
            return files

        row, col = match.groups()

        # Find all matching log files
        for log_file in self.log_dir.glob("*.log"):
            fname = log_file.name.lower()
            if f"core_row[{row}].core_col[{col}]" in fname.lower():
                if node_type == "inode":
                    if "hazard_control" in fname:
                        files["hazard_control"] = log_file
                    elif "wb_stage.log" in fname and "send_fifo" not in fname:
                        files["wb_stage"] = log_file
                    elif "ex_stage.log" in fname and "recv_fifo" not in fname:
                        files["ex_stage"] = log_file
                else:  # imce
                    if "hazard_detector" in fname:
                        files["hazard_detector"] = log_file
                    elif "u_imce_ctrl.log" in fname and "hazard" not in fname:
                        files["imce_ctrl"] = log_file

        return files

    def _parse_inode_hazard_control(self, log_file: Path, node: str):
        """Parse inode hazard_control log for standby events."""
        if not log_file.exists():
            return

        if self.verbose:
            print(f"  Parsing {log_file.name}...", file=sys.stderr)

        with open(log_file, "r") as f:
            for line in f:
                match = self.INODE_STANDBY_START.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="inode",
                        event_type="STANDBY_START",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))
                    continue

                match = self.INODE_STANDBY_END.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="inode",
                        event_type="STANDBY_END",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))

    def _parse_inode_wb_stage(self, log_file: Path, node: str):
        """Parse inode wb_stage log for SEND and SETFLAG events."""
        if not log_file.exists():
            return

        if self.verbose:
            print(f"  Parsing {log_file.name}...", file=sys.stderr)

        with open(log_file, "r") as f:
            for line in f:
                match = self.WB_SEND.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="inode",
                        event_type="SEND",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))
                    continue

                match = self.WB_SETFLAG.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="inode",
                        event_type="SETFLAG",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))

    def _parse_inode_ex_stage(self, log_file: Path, node: str):
        """Parse inode ex_stage log for OP_STANDBY, OP_SET_FLAG, OP_RECV events."""
        if not log_file.exists():
            return

        if self.verbose:
            print(f"  Parsing {log_file.name}...", file=sys.stderr)

        target_ops = {"OP_STANDBY", "OP_SET_FLAG", "OP_RECV", "OP_SEND"}

        with open(log_file, "r") as f:
            for line in f:
                match = self.INODE_EX_OP_START.search(line)
                if match:
                    op_name = match.group(2)
                    if op_name in target_ops:
                        self.events.append(SyncEvent(
                            timestamp=int(match.group(1)),
                            node=node, node_type="inode",
                            event_type=f"{op_name}_START",
                            details=match.group(3).strip(),
                            raw_line=line.strip(),
                        ))
                    continue

                match = self.INODE_EX_OP_END.search(line)
                if match:
                    op_name = match.group(2)
                    if op_name in target_ops:
                        self.events.append(SyncEvent(
                            timestamp=int(match.group(1)),
                            node=node, node_type="inode",
                            event_type=f"{op_name}_END",
                            details=match.group(3).strip(),
                            raw_line=line.strip(),
                        ))

    def _parse_imce_hazard_detector(self, log_file: Path, node: str):
        """Parse imce hazard_detector log for stall events."""
        if not log_file.exists():
            return

        if self.verbose:
            print(f"  Parsing {log_file.name}...", file=sys.stderr)

        patterns = [
            (self.IMCE_STANDBY_START, "STANDBY_START", True),
            (self.IMCE_STANDBY_END, "STANDBY_END", True),
            (self.IMCE_STEP_STALL_START, "STEP_STALL_START", False),
            (self.IMCE_STEP_STALL_END, "STEP_STALL_END", True),
            (self.IMCE_RECV_STALL_START, "RECV_STALL_START", True),
            (self.IMCE_RECV_STALL_END, "RECV_STALL_END", True),
        ]

        with open(log_file, "r") as f:
            for line in f:
                for pattern, event_type, has_details in patterns:
                    match = pattern.search(line)
                    if match:
                        details = match.group(2).strip() if has_details else ""
                        self.events.append(SyncEvent(
                            timestamp=int(match.group(1)),
                            node=node, node_type="imce",
                            event_type=event_type,
                            details=details,
                            raw_line=line.strip(),
                        ))
                        break

    def _parse_imce_ctrl(self, log_file: Path, node: str):
        """Parse imce_ctrl log for SETFLAG and handshake events."""
        if not log_file.exists():
            return

        if self.verbose:
            print(f"  Parsing {log_file.name}...", file=sys.stderr)

        with open(log_file, "r") as f:
            for line in f:
                match = self.IMCE_CTRL_SETFLAG.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="imce",
                        event_type="SETFLAG",
                        details=match.group(3).strip(),
                        raw_line=line.strip(),
                    ))
                    continue

                match = self.IMCE_CTRL_RECV.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="imce",
                        event_type="RECV",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))
                    continue

                match = self.IMCE_CTRL_SEND.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="imce",
                        event_type="SEND",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))
                    continue

                match = self.IMCE_CTRL_STEP.search(line)
                if match:
                    self.events.append(SyncEvent(
                        timestamp=int(match.group(1)),
                        node=node, node_type="imce",
                        event_type="STEP",
                        details=match.group(2).strip(),
                        raw_line=line.strip(),
                    ))

    def parse_all(self):
        """Parse all relevant log files for the specified nodes."""
        for node in self.nodes:
            files = self._find_log_files_for_node(node)

            if self.verbose:
                print(f"Found files for {node}: {list(files.keys())}", file=sys.stderr)

            if "inode" in node.lower():
                if "hazard_control" in files:
                    self._parse_inode_hazard_control(files["hazard_control"], node)
                if "wb_stage" in files:
                    self._parse_inode_wb_stage(files["wb_stage"], node)
                if "ex_stage" in files:
                    self._parse_inode_ex_stage(files["ex_stage"], node)
            else:
                if "hazard_detector" in files:
                    self._parse_imce_hazard_detector(files["hazard_detector"], node)
                if "imce_ctrl" in files:
                    self._parse_imce_ctrl(files["imce_ctrl"], node)

        # Sort events by timestamp
        self.events.sort(key=lambda e: e.timestamp)

    def get_events_in_range(self, start_time: Optional[int] = None, end_time: Optional[int] = None) -> list[SyncEvent]:
        """Get events within a time range."""
        result = self.events
        if start_time is not None:
            result = [e for e in result if e.timestamp >= start_time]
        if end_time is not None:
            result = [e for e in result if e.timestamp <= end_time]
        return result

    def filter_by_event_type(self, event_types: list[str]) -> list[SyncEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type in event_types]
