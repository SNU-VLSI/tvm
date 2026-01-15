#!/usr/bin/env python3
"""
FSIM Log Analyzer Tool

A tool for parsing, analyzing, and monitoring FSIM log files.
Useful for detecting deadlocks and analyzing simulation behavior.
Includes packet tracking capabilities for NoC analysis.
"""

import argparse
import fnmatch
import os
import random
import re
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileStatus:
    """Tracks the status of a log file."""

    path: Path
    size: int
    mtime: float
    last_check: float
    changed_since_last_check: bool = False


@dataclass
class PacketEvent:
    """Represents a single packet event in the log."""

    timestamp: int
    uuid: int
    event_type: str  # "TX" or "RX"
    direction: str  # "LOCAL", "WEST", "EAST", "NORTH", "SOUTH", "NoC"
    node: str  # node identifier from filename
    fifo_id: int
    cmd: str  # command type
    addr: int
    word: int
    raw_line: str


@dataclass
class PacketTrace:
    """Tracks the complete trace of a packet by UUID."""

    uuid: int
    events: list[PacketEvent] = field(default_factory=list)
    issued_time: Optional[int] = None
    issued_node: Optional[str] = None
    delivered_time: Optional[int] = None
    delivered_node: Optional[str] = None

    @property
    def is_delivered(self) -> bool:
        """Check if packet reached its destination (LOCAL RX)."""
        return any(
            e.event_type == "RX" and e.direction == "LOCAL" for e in self.events
        )

    @property
    def latency(self) -> Optional[int]:
        """Calculate packet latency if both issued and delivered."""
        if self.issued_time is not None and self.delivered_time is not None:
            return self.delivered_time - self.issued_time
        return None

    @property
    def hop_count(self) -> int:
        """Count number of hops through routers."""
        return len([e for e in self.events if e.event_type == "TX"])

    def get_path(self) -> list[str]:
        """Get the path taken by the packet."""
        return [e.node for e in sorted(self.events, key=lambda x: x.timestamp)]


class KeyboardHandler:
    """Handles non-blocking keyboard input for interactive selection."""

    debug_file = None
    _original_settings = None

    def __init__(self):
        self.fd = sys.stdin.fileno()

    @staticmethod
    def enable_debug(log_file: str):
        """Enable debug logging to a file."""
        KeyboardHandler.debug_file = open(log_file, 'w')

    @staticmethod
    def _debug(msg: str):
        """Write debug message to log file."""
        if KeyboardHandler.debug_file:
            KeyboardHandler.debug_file.write(f"[{time.time():.3f}] {msg}\n")
            KeyboardHandler.debug_file.flush()

    @staticmethod
    def enable_cbreak_mode():
        """Enable cbreak mode: no echo, no line buffering, but ANSI codes work."""
        fd = sys.stdin.fileno()
        try:
            KeyboardHandler._original_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            KeyboardHandler._debug("Enabled cbreak mode")
        except Exception as e:
            KeyboardHandler._debug(f"ERROR enabling cbreak mode: {e}")

    @staticmethod
    def restore_terminal():
        """Restore original terminal settings."""
        if KeyboardHandler._original_settings:
            fd = sys.stdin.fileno()
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, KeyboardHandler._original_settings)
                KeyboardHandler._debug("Restored terminal settings")
            except Exception as e:
                KeyboardHandler._debug(f"ERROR restoring terminal: {e}")

    @staticmethod
    def get_key(timeout: float = 0.0) -> Optional[str]:
        """
        Get a key press without blocking.
        Assumes terminal is already in cbreak mode.

        Args:
            timeout: Time to wait for input (0 = non-blocking)

        Returns:
            Key string or None if no input available.
            Special keys: 'UP', 'DOWN', 'ENTER', 'SPACE', etc.
        """
        try:
            KeyboardHandler._debug(f"get_key called with timeout={timeout}")

            # Check if input is available
            KeyboardHandler._debug("Calling select.select...")
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            KeyboardHandler._debug(f"select returned: ready={len(ready)}")

            if not ready:
                return None

            KeyboardHandler._debug("Input available, reading...")

            # Read the key (terminal already in cbreak mode)
            ch = sys.stdin.read(1)
            KeyboardHandler._debug(f"Read character: {repr(ch)} (ord={ord(ch) if ch else 'None'})")

            # Handle escape sequences (arrow keys, etc.)
            if ch == '\x1b':  # ESC
                KeyboardHandler._debug("ESC sequence detected")
                # Try to read the rest of the escape sequence
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch2 = sys.stdin.read(1)
                    KeyboardHandler._debug(f"ESC+{repr(ch2)}")
                    if ch2 == '[':
                        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if ready:
                            ch3 = sys.stdin.read(1)
                            KeyboardHandler._debug(f"ESC+[+{repr(ch3)}")
                            if ch3 == 'A':
                                KeyboardHandler._debug("Returning UP")
                                return 'UP'
                            elif ch3 == 'B':
                                KeyboardHandler._debug("Returning DOWN")
                                return 'DOWN'
                            elif ch3 == 'C':
                                KeyboardHandler._debug("Returning RIGHT")
                                return 'RIGHT'
                            elif ch3 == 'D':
                                KeyboardHandler._debug("Returning LEFT")
                                return 'LEFT'
                KeyboardHandler._debug("Returning ESC")
                return 'ESC'
            elif ch == '\r' or ch == '\n':
                KeyboardHandler._debug("Returning ENTER")
                return 'ENTER'
            elif ch == ' ':
                KeyboardHandler._debug("Returning SPACE")
                return 'SPACE'
            elif ch == '\x03':  # Ctrl+C
                KeyboardHandler._debug("Returning CTRL_C")
                return 'CTRL_C'
            elif ch == 'q' or ch == 'Q':
                KeyboardHandler._debug("Returning q")
                return 'q'
            elif ch == 'j':
                KeyboardHandler._debug("Returning j")
                return 'j'
            elif ch == 'k':
                KeyboardHandler._debug("Returning k")
                return 'k'
            else:
                KeyboardHandler._debug(f"Returning character: {repr(ch)}")
                return ch
        except Exception as e:
            KeyboardHandler._debug(f"ERROR in get_key: {type(e).__name__}: {e}")
            return None


class LogMonitor:
    """Monitors log files for changes to detect simulation deadlock."""

    DEFAULT_LOG_DIR = (
        Path(__file__).parent.parent / "rtl_runner" / "logs" / "fsim_logs"
    )

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        check_interval: float = 2.0,
        deadlock_threshold: float = 30.0,
        extensions: tuple = (".log",),
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the log monitor.

        Args:
            log_dir: Directory containing log files
            check_interval: How often to check for changes (seconds)
            deadlock_threshold: Time without changes to consider deadlock (seconds)
            extensions: File extensions to monitor
            include_patterns: Glob patterns to include (if set, only matching files are monitored)
            exclude_patterns: Glob patterns to exclude
            verbose: Print detailed information
            debug: Enable debug logging to /tmp/fsim_log_analyzer_debug.log
        """
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.check_interval = check_interval
        self.deadlock_threshold = deadlock_threshold
        self.extensions = extensions
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.verbose = verbose
        self.debug = debug

        if self.debug:
            debug_log = "/tmp/fsim_log_analyzer_debug.log"
            KeyboardHandler.enable_debug(debug_log)
            KeyboardHandler._debug(f"Debug logging enabled to {debug_log}")

        self.file_statuses: dict[str, FileStatus] = {}
        self.last_any_change: float = time.time()
        self.monitoring_start: float = 0

        # Selection state for keyboard navigation
        self.selection_mode: bool = False
        self.selected_index: int = 0
        self.selectable_files: list[Path] = []

    def _matches_pattern(self, filename: str, patterns: list[str]) -> bool:
        """Check if filename matches any of the given glob patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _get_log_files(self) -> list[Path]:
        """Get all log files in the directory, filtered by include/exclude patterns."""
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_dir}")

        files = []
        for ext in self.extensions:
            files.extend(self.log_dir.glob(f"*{ext}"))

        # Apply include patterns (if specified, only keep matching files)
        if self.include_patterns:
            files = [
                f
                for f in files
                if self._matches_pattern(f.name, self.include_patterns)
            ]

        # Apply exclude patterns
        if self.exclude_patterns:
            files = [
                f
                for f in files
                if not self._matches_pattern(f.name, self.exclude_patterns)
            ]

        return sorted(files)

    def _get_file_stat(self, path: Path) -> tuple[int, float]:
        """Get file size and modification time."""
        try:
            stat = path.stat()
            return stat.st_size, stat.st_mtime
        except OSError:
            return 0, 0

    def _open_file_in_vscode(self, file_path: Path):
        """Open a file in VS Code."""
        try:
            subprocess.Popen(
                ['code', str(file_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            # Silently fail - we don't want to crash the monitor
            pass

    def _handle_keyboard_input(self, key: Optional[str]) -> bool:
        """
        Handle keyboard input for file selection.

        Args:
            key: Key pressed by user

        Returns:
            True if should continue monitoring, False if should exit
        """
        if key is None:
            return True

        KeyboardHandler._debug(f"_handle_keyboard_input: key={repr(key)}, selection_mode={self.selection_mode}, selected_index={self.selected_index}")

        # Handle Ctrl+C
        if key == 'CTRL_C':
            KeyboardHandler._debug("Handling CTRL_C - exiting")
            return False

        # If not in selection mode, any navigation key enters selection mode
        if not self.selection_mode:
            KeyboardHandler._debug(f"Not in selection mode, checking if key enters selection mode")
            if key in ['UP', 'DOWN', 'j', 'k'] and self.selectable_files:
                self.selection_mode = True
                self.selected_index = 0
                KeyboardHandler._debug(f"Entered selection mode! selectable_files count={len(self.selectable_files)}")
            return True

        # In selection mode - handle navigation and selection
        KeyboardHandler._debug("In selection mode, handling navigation")
        if key in ['UP', 'k']:
            if self.selectable_files:
                old_index = self.selected_index
                self.selected_index = (self.selected_index - 1) % len(self.selectable_files)
                KeyboardHandler._debug(f"UP/k: moved from {old_index} to {self.selected_index}")
        elif key in ['DOWN', 'j']:
            if self.selectable_files:
                old_index = self.selected_index
                self.selected_index = (self.selected_index + 1) % len(self.selectable_files)
                KeyboardHandler._debug(f"DOWN/j: moved from {old_index} to {self.selected_index}")
        elif key in ['ENTER', 'SPACE']:
            if self.selectable_files and 0 <= self.selected_index < len(self.selectable_files):
                selected_file = self.selectable_files[self.selected_index]
                KeyboardHandler._debug(f"Opening file: {selected_file}")
                self._open_file_in_vscode(selected_file)
        elif key == 'q':
            # Exit selection mode
            KeyboardHandler._debug("Exiting selection mode")
            self.selection_mode = False
            self.selected_index = 0

        return True

    def _update_file_status(self, path: Path, current_time: float) -> bool:
        """
        Update the status of a file and return whether it changed.

        Returns:
            True if the file changed since last check
        """
        size, mtime = self._get_file_stat(path)
        path_str = str(path)

        if path_str not in self.file_statuses:
            self.file_statuses[path_str] = FileStatus(
                path=path,
                size=size,
                mtime=mtime,
                last_check=current_time,
                changed_since_last_check=True,
            )
            return True

        status = self.file_statuses[path_str]
        changed = (size != status.size) or (mtime != status.mtime)

        status.size = size
        status.mtime = mtime
        status.last_check = current_time
        status.changed_since_last_check = changed

        return changed

    def check_once(self) -> dict:
        """
        Perform a single check of all log files.

        Returns:
            Dictionary with check results
        """
        current_time = time.time()
        files = self._get_log_files()

        changed_files = []
        unchanged_files = []

        for f in files:
            if self._update_file_status(f, current_time):
                changed_files.append(f)
            else:
                unchanged_files.append(f)

        if changed_files:
            self.last_any_change = current_time

        time_since_change = current_time - self.last_any_change

        return {
            "timestamp": current_time,
            "total_files": len(files),
            "changed_files": changed_files,
            "unchanged_files": unchanged_files,
            "time_since_any_change": time_since_change,
            "potential_deadlock": time_since_change >= self.deadlock_threshold,
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _print_status(self, result: dict, clear_screen: bool = True):
        """Print the current monitoring status."""
        if clear_screen:
            # ANSI escape code to clear screen and move cursor to top
            print("\033[2J\033[H", end="")

        elapsed = time.time() - self.monitoring_start
        print("=" * 70)
        print(f"  FSIM Log Monitor - Elapsed: {self._format_time(elapsed)}")
        print("=" * 70)
        print(f"  Log directory: {self.log_dir}")
        print(f"  Monitoring files: {result['total_files']}")
        if self.include_patterns:
            print(f"  Include: {', '.join(self.include_patterns)}")
        if self.exclude_patterns:
            print(f"  Exclude: {', '.join(self.exclude_patterns)}")
        print(
            f"  Check interval: {self.check_interval}s | Deadlock threshold: {self.deadlock_threshold}s"
        )
        print("-" * 70)

        time_since_change = result["time_since_any_change"]

        if result["potential_deadlock"]:
            print(f"\n  [!!! POTENTIAL DEADLOCK !!!]")
            print(
                f"  No log file changes for {self._format_time(time_since_change)}"
            )
            print(f"  Threshold: {self.deadlock_threshold}s")
        else:
            progress_bar_width = 40
            progress = min(time_since_change / self.deadlock_threshold, 1.0)
            filled = int(progress_bar_width * progress)
            bar = "█" * filled + "░" * (progress_bar_width - filled)
            print(f"\n  Status: ACTIVE")
            print(
                f"  Time since last change: {self._format_time(time_since_change)}"
            )
            print(f"  Deadlock timer: [{bar}] {progress*100:.0f}%")

        # Update selectable files list
        self.selectable_files = result["changed_files"][:10]  # Limit to 10 files

        if result["changed_files"]:
            print(
                f"\n  Recently changed files ({len(result['changed_files'])}):"
            )
            # Show up to 10 files with numbers for selection
            display_count = min(len(result["changed_files"]), 10)
            for i in range(display_count):
                f = result["changed_files"][i]
                # Show selection indicator if in selection mode
                if self.selection_mode and i == self.selected_index:
                    indicator = "→"
                else:
                    indicator = " "
                print(f"    {indicator} [{i+1}] {f.name}")

            if len(result["changed_files"]) > 10:
                print(f"    ... and {len(result['changed_files']) - 10} more")

        print("\n" + "-" * 70)
        if self.selection_mode:
            print("  ↑↓/jk: navigate | Enter/Space: open in VS Code | q: exit selection")
        else:
            print("  ↑↓/jk: enter selection mode | Ctrl+C: stop monitoring")
        print("=" * 70)
        sys.stdout.flush()  # Ensure output is displayed immediately

    def monitor(self, duration: Optional[float] = None) -> bool:
        """
        Start monitoring log files continuously.

        Args:
            duration: Maximum monitoring duration in seconds (None for indefinite)

        Returns:
            True if deadlock was detected, False otherwise
        """
        print(f"Starting log monitor...")
        print(f"  Directory: {self.log_dir}")
        print(f"  Check interval: {self.check_interval}s")
        print(f"  Deadlock threshold: {self.deadlock_threshold}s")
        print()

        self.monitoring_start = time.time()
        self.last_any_change = time.time()
        deadlock_detected = False

        # Enable cbreak mode for responsive keyboard input
        KeyboardHandler.enable_cbreak_mode()

        try:
            # Do initial file check so result is never None
            result = self.check_once()
            self._print_status(result)
            last_check_time = time.time()

            while True:
                current_time = time.time()

                # Check files at regular intervals
                if current_time - last_check_time >= self.check_interval:
                    result = self.check_once()
                    self._print_status(result)
                    last_check_time = current_time

                    if result and result["potential_deadlock"]:
                        deadlock_detected = True

                # Check for keyboard input frequently (every 100ms)
                key = KeyboardHandler.get_key(timeout=0.1)
                if not self._handle_keyboard_input(key):
                    # User pressed Ctrl+C
                    break

                # If navigation key was pressed, redraw immediately
                # Always redraw on navigation keys (don't check result)
                if key in ['UP', 'DOWN', 'j', 'k', 'q', 'ENTER', 'SPACE']:
                    KeyboardHandler._debug(f"Navigation key {key} pressed, redrawing...")
                    self._print_status(result)

                if (
                    duration
                    and (time.time() - self.monitoring_start) >= duration
                ):
                    print("\nMonitoring duration reached.")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            # Always restore terminal settings
            KeyboardHandler.restore_terminal()
            print("\n\nMonitoring stopped by user.")
            sys.stdout.flush()

        return deadlock_detected

    def get_active_files(
        self, since_seconds: float = 60.0
    ) -> list[FileStatus]:
        """
        Get files that have been modified within the given time window.

        Args:
            since_seconds: Time window in seconds

        Returns:
            List of FileStatus for recently modified files
        """
        current_time = time.time()
        cutoff = current_time - since_seconds

        active = []
        for status in self.file_statuses.values():
            if status.mtime >= cutoff:
                active.append(status)

        return sorted(active, key=lambda x: x.mtime, reverse=True)

    def summary(self) -> dict:
        """
        Get a summary of all monitored files.

        Returns:
            Dictionary with summary statistics
        """
        files = self._get_log_files()
        current_time = time.time()

        total_size = 0
        non_empty_count = 0

        for f in files:
            size, _ = self._get_file_stat(f)
            total_size += size
            if size > 0:
                non_empty_count += 1

        return {
            "log_dir": str(self.log_dir),
            "total_files": len(files),
            "non_empty_files": non_empty_count,
            "empty_files": len(files) - non_empty_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


class PacketAnalyzer:
    """Analyzes packet traces from FSIM log files."""

    # Regex patterns for parsing log lines
    ROUTER_PATTERN = re.compile(
        r"\[\s*(\d+)\]\s+(RX|TX)\s+TRANSFER\s+(from|to)\s+(\w+)\s+\|\s+UUID:\s*(\d+)\s+\|\s+fifo_id:\s*(\d+)\s+\|\s+cmd:\s*(\S+)\s+\|\s+addr:\s*(\d+)\s+\|\s+word:\s*(\d+)"
    )
    NOC_TX_PATTERN = re.compile(
        r"\[\s*(\d+)\]\s+TX\s+to\s+NoC\s+\|\s+UUID:\s*(\d+)\s+\|\s+fifo_id:\s*(\d+)"
    )

    def __init__(self, log_dir: Path, verbose: bool = False):
        """
        Initialize packet analyzer.

        Args:
            log_dir: Directory containing log files
            verbose: Print detailed parsing information
        """
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.packets: dict[int, PacketTrace] = {}
        self.node_stats: dict[str, dict] = defaultdict(
            lambda: {"rx_count": 0, "tx_count": 0, "packets": set()}
        )

    def _extract_node_from_filename(self, filename: str) -> str:
        """Extract node identifier from log filename."""
        # Example: fsim_testbench_imcflow_gem5.u_imcflow_with_axi.u_imcflow_impl.core_row[0].core_col[0].inode.u_router.log
        # Extract: core_row[0].core_col[0].inode (or imce_node)
        match = re.search(
            r"(core_row\[\d+\]\.core_col\[\d+\]\.(inode|imce_node))", filename
        )
        if match:
            return match.group(1)
        return filename

    def _parse_router_log_line(
        self, line: str, node: str
    ) -> Optional[PacketEvent]:
        """Parse a router log line into a PacketEvent."""
        match = self.ROUTER_PATTERN.match(line)
        if not match:
            return None

        timestamp, event_type, _, direction, uuid, fifo_id, cmd, addr, word = (
            match.groups()
        )

        return PacketEvent(
            timestamp=int(timestamp),
            uuid=int(uuid),
            event_type=event_type,
            direction=direction,
            node=node,
            fifo_id=int(fifo_id),
            cmd=cmd,
            addr=int(addr),
            word=int(word),
            raw_line=line.strip(),
        )

    def _parse_noc_tx_line(self, line: str, node: str) -> Optional[PacketEvent]:
        """Parse a NoC TX log line into a PacketEvent."""
        match = self.NOC_TX_PATTERN.match(line)
        if not match:
            return None

        timestamp, uuid, fifo_id = match.groups()

        return PacketEvent(
            timestamp=int(timestamp),
            uuid=int(uuid),
            event_type="TX",
            direction="NoC",
            node=node,
            fifo_id=int(fifo_id),
            cmd="",
            addr=0,
            word=0,
            raw_line=line.strip(),
        )

    def parse_log_file(self, log_file: Path):
        """Parse a single log file and extract packet events."""
        if not log_file.exists():
            return

        node = self._extract_node_from_filename(log_file.name)

        if self.verbose:
            print(f"Parsing {log_file.name}...", file=sys.stderr)

        try:
            with open(log_file, "r") as f:
                for line in f:
                    if "UUID:" not in line:
                        continue

                    # Try parsing as router log
                    event = self._parse_router_log_line(line, node)

                    # If not router log, try NoC TX pattern
                    if event is None:
                        event = self._parse_noc_tx_line(line, node)

                    if event is None:
                        continue

                    # Add event to packet trace
                    uuid = event.uuid
                    if uuid not in self.packets:
                        self.packets[uuid] = PacketTrace(uuid=uuid)

                    trace = self.packets[uuid]
                    trace.events.append(event)

                    # Track issued time and node
                    if (
                        event.direction == "NoC"
                        or (
                            event.event_type == "TX" and event.direction == "LOCAL"
                        )
                    ) and trace.issued_time is None:
                        trace.issued_time = event.timestamp
                        trace.issued_node = node

                    # Track delivered time and node
                    if (
                        event.event_type == "RX" and event.direction == "LOCAL"
                    ):
                        if (
                            trace.delivered_time is None
                            or event.timestamp > trace.delivered_time
                        ):
                            trace.delivered_time = event.timestamp
                            trace.delivered_node = node

                    # Update node statistics
                    if event.event_type == "RX":
                        self.node_stats[node]["rx_count"] += 1
                    else:
                        self.node_stats[node]["tx_count"] += 1
                    self.node_stats[node]["packets"].add(uuid)

        except Exception as e:
            print(
                f"Error parsing {log_file.name}: {e}", file=sys.stderr
            )

    def parse_all_logs(self):
        """Parse all log files in the log directory."""
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_dir}")

        log_files = list(self.log_dir.glob("*.log"))

        if self.verbose:
            print(f"Found {len(log_files)} log files", file=sys.stderr)

        for log_file in log_files:
            self.parse_log_file(log_file)

        if self.verbose:
            print(
                f"Parsed {len(self.packets)} unique packets",
                file=sys.stderr,
            )

    def get_undelivered_packets(self) -> list[PacketTrace]:
        """Get packets that were issued but not delivered to destination."""
        return [
            trace
            for trace in self.packets.values()
            if trace.issued_time is not None and not trace.is_delivered
        ]

    def get_delivered_packets(self) -> list[PacketTrace]:
        """Get packets that were successfully delivered."""
        return [trace for trace in self.packets.values() if trace.is_delivered]

    def get_packet_by_uuid(self, uuid: int) -> Optional[PacketTrace]:
        """Get packet trace by UUID."""
        return self.packets.get(uuid)

    def get_latency_stats(self) -> dict:
        """Calculate latency statistics for delivered packets."""
        latencies = [
            trace.latency
            for trace in self.packets.values()
            if trace.latency is not None
        ]

        if not latencies:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "median": 0,
            }

        latencies.sort()
        return {
            "count": len(latencies),
            "min": latencies[0],
            "max": latencies[-1],
            "avg": sum(latencies) / len(latencies),
            "median": latencies[len(latencies) // 2],
        }

    def get_node_traffic_stats(self) -> dict[str, dict]:
        """Get traffic statistics by node."""
        stats = {}
        for node, data in self.node_stats.items():
            stats[node] = {
                "rx_count": data["rx_count"],
                "tx_count": data["tx_count"],
                "unique_packets": len(data["packets"]),
                "total_traffic": data["rx_count"] + data["tx_count"],
            }
        return stats

    def get_cmd_type_stats(self) -> dict[str, int]:
        """Get statistics by command type."""
        cmd_counts = defaultdict(int)
        for trace in self.packets.values():
            for event in trace.events:
                if event.cmd:
                    cmd_counts[event.cmd] += 1
        return dict(cmd_counts)

    def get_hotspots(
        self, top_n: int = 5, metric: str = "total_traffic"
    ) -> list[tuple[str, dict]]:
        """
        Get top N nodes with highest traffic.

        Args:
            top_n: Number of top nodes to return
            metric: Metric to sort by ('total_traffic', 'rx_count', 'tx_count', 'unique_packets')
        """
        stats = self.get_node_traffic_stats()
        sorted_nodes = sorted(
            stats.items(), key=lambda x: x[1].get(metric, 0), reverse=True
        )
        return sorted_nodes[:top_n]


def _parse_patterns(pattern_str: Optional[str]) -> list[str]:
    """Parse comma-separated pattern string into list."""
    if not pattern_str:
        return []
    return [p.strip() for p in pattern_str.split(",") if p.strip()]


class DebugTestDirectory:
    """Creates and manages a test directory with simulated file activity for debugging."""

    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="fsim_debug_"))
        self.stop_event = threading.Event()
        self.thread = None
        self.files = []

    def create_test_files(self, count: int = 10):
        """Create test log files."""
        print(f"Creating {count} test files in {self.test_dir}")

        # Create various test files with realistic names
        test_names = [
            "testbench.u_core.u_stage1.log",
            "testbench.u_core.u_stage2.log",
            "testbench.u_core.u_mem.log",
            "testbench.u_noc.u_router[0].log",
            "testbench.u_noc.u_router[1].log",
            "testbench.u_imce.u_ctrl.log",
            "testbench.u_imce.u_datapath.log",
            "testbench.u_bridge.u_axi.log",
            "testbench.u_bridge.u_demux.log",
            "run.log",
        ]

        for i, name in enumerate(test_names[:count]):
            file_path = self.test_dir / name
            # Create file with some initial content
            with open(file_path, 'w') as f:
                f.write(f"[0] Test log file: {name}\n")
                f.write(f"[100] Initialized at {time.time()}\n")
            self.files.append(file_path)

        print(f"Created {len(self.files)} test files")

    def _update_files_loop(self):
        """Background thread that periodically updates random files."""
        cycle = 0
        while not self.stop_event.is_set():
            # Update 1-3 random files each cycle
            num_updates = random.randint(1, min(3, len(self.files)))
            files_to_update = random.sample(self.files, num_updates)

            for file_path in files_to_update:
                try:
                    with open(file_path, 'a') as f:
                        timestamp = int(time.time() * 1000)
                        f.write(f"[{timestamp}] Cycle {cycle}: Random activity\n")
                except Exception:
                    pass

            cycle += 1
            # Update every 0.5-2 seconds
            time.sleep(random.uniform(0.5, 2.0))

    def start_activity(self):
        """Start background thread to simulate file activity."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._update_files_loop, daemon=True)
            self.thread.start()
            print("Started simulated file activity (updates every 0.5-2s)")

    def stop_activity(self):
        """Stop background activity thread."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2.0)

    def cleanup(self):
        """Clean up test directory."""
        self.stop_activity()
        # Note: We don't delete the directory automatically so user can inspect it
        # User can manually delete /tmp/fsim_debug_* directories later

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def cmd_monitor(args):
    """Handle the monitor command."""
    test_dir = None
    log_dir = args.log_dir

    # If debug mode and no log_dir specified, create test directory
    if args.debug and log_dir is None:
        print("=" * 70)
        print("  DEBUG MODE - Creating test environment")
        print("=" * 70)
        test_dir = DebugTestDirectory()
        test_dir.create_test_files(count=10)
        test_dir.start_activity()
        log_dir = test_dir.test_dir
        print(f"Test directory: {log_dir}")
        print("Debug logging: /tmp/fsim_log_analyzer_debug.log")
        print("=" * 70)
        print()

    try:
        monitor = LogMonitor(
            log_dir=log_dir,
            check_interval=args.interval,
            deadlock_threshold=args.threshold,
            extensions=tuple(args.extensions.split(",")),
            include_patterns=_parse_patterns(args.include),
            exclude_patterns=_parse_patterns(args.exclude),
            verbose=args.verbose,
            debug=args.debug,
        )

        if args.debug and test_dir is None:
            print("Debug mode enabled - logging to /tmp/fsim_log_analyzer_debug.log")
            print()

        deadlock = monitor.monitor(duration=args.duration)
        sys.exit(1 if deadlock else 0)
    finally:
        if test_dir:
            print("\n\nCleaning up test environment...")
            test_dir.cleanup()
            print(f"Test directory preserved at: {test_dir.test_dir}")
            print("You can inspect it or delete with: rm -rf /tmp/fsim_debug_*")


def cmd_summary(args):
    """Handle the summary command."""
    monitor = LogMonitor(
        log_dir=args.log_dir,
        extensions=tuple(args.extensions.split(",")),
        include_patterns=_parse_patterns(args.include),
        exclude_patterns=_parse_patterns(args.exclude),
    )

    summary = monitor.summary()

    print("=" * 50)
    print("  FSIM Log Summary")
    print("=" * 50)
    print(f"  Directory: {summary['log_dir']}")
    if args.include:
        print(f"  Include: {args.include}")
    if args.exclude:
        print(f"  Exclude: {args.exclude}")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Non-empty files: {summary['non_empty_files']}")
    print(f"  Empty files: {summary['empty_files']}")
    print(f"  Total size: {summary['total_size_mb']:.2f} MB")
    print("=" * 50)


def cmd_check(args):
    """Handle the single check command."""
    monitor = LogMonitor(
        log_dir=args.log_dir,
        deadlock_threshold=args.threshold,
        extensions=tuple(args.extensions.split(",")),
        include_patterns=_parse_patterns(args.include),
        exclude_patterns=_parse_patterns(args.exclude),
    )

    # Do initial check to populate file statuses
    monitor.check_once()

    # Wait and check again
    print(f"Checking for changes over {args.wait}s...")
    time.sleep(args.wait)

    result = monitor.check_once()

    if result["changed_files"]:
        print(f"\n{len(result['changed_files'])} files changed:")
        for f in result["changed_files"]:
            print(f"  - {f.name}")
        print("\nSimulation appears to be ACTIVE.")
        sys.exit(0)
    else:
        print(f"\nNo files changed in {args.wait}s.")
        print("Simulation may be STALLED or COMPLETE.")
        sys.exit(1)


def cmd_list(args):
    """Handle the list command."""
    monitor = LogMonitor(
        log_dir=args.log_dir,
        extensions=tuple(args.extensions.split(",")),
        include_patterns=_parse_patterns(args.include),
        exclude_patterns=_parse_patterns(args.exclude),
    )

    files = monitor._get_log_files()

    print("=" * 70)
    print("  Matching Log Files")
    print("=" * 70)
    print(f"  Directory: {monitor.log_dir}")
    if args.include:
        print(f"  Include: {args.include}")
    if args.exclude:
        print(f"  Exclude: {args.exclude}")
    print(f"  Total: {len(files)} files")
    print("-" * 70)

    if not files:
        print("  No matching files found.")
    else:
        for f in files:
            size, mtime = monitor._get_file_stat(f)
            size_str = f"{size:,}" if size > 0 else "(empty)"
            print(f"  {f.name}")
            if args.verbose:
                mtime_str = datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"      Size: {size_str} bytes | Modified: {mtime_str}")

    print("=" * 70)


def cmd_packet_stats(args):
    """Handle the packet statistics command."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    total_packets = len(analyzer.packets)
    delivered = analyzer.get_delivered_packets()
    undelivered = analyzer.get_undelivered_packets()
    latency_stats = analyzer.get_latency_stats()

    print("=" * 70)
    print("  Packet Statistics")
    print("=" * 70)
    print(f"  Total unique packets: {total_packets}")
    print(f"  Delivered packets: {len(delivered)}")
    print(f"  Undelivered packets: {len(undelivered)}")
    print(
        f"  Delivery rate: {len(delivered)/total_packets*100:.1f}%"
        if total_packets > 0
        else "  Delivery rate: N/A"
    )
    print()
    print("  Latency Statistics (for delivered packets):")
    print(f"    Count: {latency_stats['count']}")
    if latency_stats["count"] > 0:
        print(f"    Min: {latency_stats['min']:,} cycles")
        print(f"    Max: {latency_stats['max']:,} cycles")
        print(f"    Avg: {latency_stats['avg']:,.1f} cycles")
        print(f"    Median: {latency_stats['median']:,} cycles")
    print("=" * 70)


def cmd_packet_undelivered(args):
    """Handle the undelivered packets command."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    undelivered = analyzer.get_undelivered_packets()

    print("=" * 70)
    print(f"  Undelivered Packets: {len(undelivered)}")
    print("=" * 70)

    if not undelivered:
        print("  No undelivered packets found!")
    else:
        # Sort by UUID
        undelivered.sort(key=lambda x: x.uuid)

        if args.limit:
            print(f"  (Showing first {args.limit} packets)")
            undelivered = undelivered[: args.limit]

        for trace in undelivered:
            print(f"\n  UUID: {trace.uuid}")
            print(f"    Issued at: {trace.issued_time:,} cycles")
            print(f"    Issued from: {trace.issued_node}")
            print(f"    Events: {len(trace.events)}")
            print(f"    Hops: {trace.hop_count}")

            if args.verbose:
                print("    Path:")
                for i, event in enumerate(
                    sorted(trace.events, key=lambda x: x.timestamp)
                ):
                    print(
                        f"      [{i}] {event.timestamp:15,} | {event.event_type} {event.direction:6} @ {event.node}"
                    )

    print("=" * 70)


def cmd_packet_trace(args):
    """Handle the packet trace command."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    trace = analyzer.get_packet_by_uuid(args.uuid)

    if trace is None:
        print(f"Packet UUID {args.uuid} not found!")
        sys.exit(1)

    print("=" * 70)
    print(f"  Packet Trace for UUID: {args.uuid}")
    print("=" * 70)
    print(f"  Issued at: {trace.issued_time:,} cycles" if trace.issued_time else "  Issued at: Unknown")
    print(f"  Issued from: {trace.issued_node}" if trace.issued_node else "  Issued from: Unknown")
    print(f"  Delivered: {'Yes' if trace.is_delivered else 'No'}")

    if trace.is_delivered:
        print(f"  Delivered at: {trace.delivered_time:,} cycles")
        print(f"  Delivered to: {trace.delivered_node}")
        if trace.latency:
            print(f"  Latency: {trace.latency:,} cycles")

    print(f"  Total events: {len(trace.events)}")
    print(f"  Hop count: {trace.hop_count}")
    print()
    print("  Event Timeline:")
    print("  " + "-" * 66)

    for i, event in enumerate(sorted(trace.events, key=lambda x: x.timestamp)):
        print(
            f"  [{i:3}] T={event.timestamp:15,} | {event.event_type} {event.direction:6} @ {event.node}"
        )
        if args.verbose:
            print(f"        cmd={event.cmd}, fifo_id={event.fifo_id}, addr={event.addr}, word={event.word}")

    print("=" * 70)


def cmd_packet_node_stats(args):
    """Handle the node traffic statistics command."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    stats = analyzer.get_node_traffic_stats()

    print("=" * 70)
    print("  Node Traffic Statistics")
    print("=" * 70)

    if not stats:
        print("  No traffic data found!")
    else:
        # Sort by total traffic
        sorted_stats = sorted(
            stats.items(), key=lambda x: x[1]["total_traffic"], reverse=True
        )

        if args.limit:
            print(f"  (Showing top {args.limit} nodes)")
            sorted_stats = sorted_stats[: args.limit]

        print(
            f"  {'Node':<40} {'RX':>8} {'TX':>8} {'Unique':>8} {'Total':>8}"
        )
        print("  " + "-" * 68)

        for node, data in sorted_stats:
            print(
                f"  {node:<40} {data['rx_count']:>8} {data['tx_count']:>8} {data['unique_packets']:>8} {data['total_traffic']:>8}"
            )

    print("=" * 70)


def cmd_packet_hotspots(args):
    """Handle the hotspot detection command."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    hotspots = analyzer.get_hotspots(top_n=args.top_n, metric=args.metric)

    print("=" * 70)
    print(f"  Top {args.top_n} Traffic Hotspots (by {args.metric})")
    print("=" * 70)

    if not hotspots:
        print("  No traffic data found!")
    else:
        for i, (node, data) in enumerate(hotspots, 1):
            print(f"\n  [{i}] {node}")
            print(f"      RX count: {data['rx_count']:,}")
            print(f"      TX count: {data['tx_count']:,}")
            print(f"      Unique packets: {data['unique_packets']:,}")
            print(f"      Total traffic: {data['total_traffic']:,}")

    print("=" * 70)


def cmd_packet_cmd_stats(args):
    """Handle the command type statistics."""
    analyzer = PacketAnalyzer(log_dir=args.log_dir, verbose=args.verbose)
    analyzer.parse_all_logs()

    cmd_stats = analyzer.get_cmd_type_stats()

    print("=" * 70)
    print("  Command Type Statistics")
    print("=" * 70)

    if not cmd_stats:
        print("  No command data found!")
    else:
        total = sum(cmd_stats.values())
        sorted_cmds = sorted(
            cmd_stats.items(), key=lambda x: x[1], reverse=True
        )

        print(f"  {'Command':<30} {'Count':>12} {'Percentage':>12}")
        print("  " + "-" * 56)

        for cmd, count in sorted_cmds:
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {cmd:<30} {count:>12,} {percentage:>11.1f}%")

        print("  " + "-" * 56)
        print(f"  {'TOTAL':<30} {total:>12,} {100.0:>11.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="FSIM Log Analyzer Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor logs for deadlock with default settings
  %(prog)s monitor

  # Monitor with custom threshold
  %(prog)s monitor --threshold 60

  # Monitor only router logs
  %(prog)s monitor --include "*router*"

  # Monitor excluding policy_table logs
  %(prog)s monitor --exclude "*policy_table*"

  # Monitor specific module patterns
  %(prog)s monitor --include "*inode*,*ex_stage*"

  # Quick check if simulation is active
  %(prog)s check --wait 5

  # Get summary of log files
  %(prog)s summary

  # List matching files before monitoring
  %(prog)s --include "*router*" list
  %(prog)s --include "*router*" list -v  # with details

  # Packet analysis commands
  %(prog)s packet-stats -d <log_dir>  # Overall packet statistics
  %(prog)s packet-undelivered -d <log_dir>  # Find undelivered packets
  %(prog)s packet-trace -d <log_dir> --uuid 123  # Trace specific packet
  %(prog)s packet-node-stats -d <log_dir>  # Node traffic statistics
  %(prog)s packet-hotspots -d <log_dir>  # Find traffic hotspots
  %(prog)s packet-cmd-stats -d <log_dir>  # Command type statistics
""",
    )

    parser.add_argument(
        "--log-dir",
        "-d",
        type=Path,
        default=None,
        help="Log directory (default: rtl_runner/logs/fsim_logs)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        default=".log",
        help="Comma-separated file extensions to monitor (default: .log)",
    )
    parser.add_argument(
        "--include",
        "-I",
        default=None,
        help="Comma-separated glob patterns to include (e.g., '*router*,*ex_stage*')",
    )
    parser.add_argument(
        "--exclude",
        "-X",
        default=None,
        help="Comma-separated glob patterns to exclude (e.g., '*policy_table*')",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Continuously monitor log files for changes"
    )
    monitor_parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=2.0,
        help="Check interval in seconds (default: 2.0)",
    )
    monitor_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=30.0,
        help="Deadlock threshold in seconds (default: 30.0)",
    )
    monitor_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum monitoring duration in seconds (default: unlimited)",
    )
    monitor_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )
    monitor_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to /tmp/fsim_log_analyzer_debug.log. "
             "If --log-dir is not specified, creates a test directory with simulated file activity.",
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # Summary command
    summary_parser = subparsers.add_parser(
        "summary", help="Show summary of log files"
    )
    summary_parser.set_defaults(func=cmd_summary)

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Single check if simulation is active"
    )
    check_parser.add_argument(
        "--wait",
        "-w",
        type=float,
        default=5.0,
        help="Wait time before checking (default: 5.0s)",
    )
    check_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=30.0,
        help="Deadlock threshold in seconds (default: 30.0)",
    )
    check_parser.set_defaults(func=cmd_check)

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List files matching the current filter patterns"
    )
    list_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show file size and modification time",
    )
    list_parser.set_defaults(func=cmd_list)

    # Packet stats command
    packet_stats_parser = subparsers.add_parser(
        "packet-stats", help="Show overall packet statistics"
    )
    packet_stats_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )
    packet_stats_parser.set_defaults(func=cmd_packet_stats)

    # Packet undelivered command
    packet_undelivered_parser = subparsers.add_parser(
        "packet-undelivered",
        help="Find packets that were issued but not delivered",
    )
    packet_undelivered_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of packets to display",
    )
    packet_undelivered_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed path information",
    )
    packet_undelivered_parser.set_defaults(func=cmd_packet_undelivered)

    # Packet trace command
    packet_trace_parser = subparsers.add_parser(
        "packet-trace", help="Trace a specific packet by UUID"
    )
    packet_trace_parser.add_argument(
        "--uuid",
        "-u",
        type=int,
        required=True,
        help="UUID of the packet to trace",
    )
    packet_trace_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed event information",
    )
    packet_trace_parser.set_defaults(func=cmd_packet_trace)

    # Packet node stats command
    packet_node_stats_parser = subparsers.add_parser(
        "packet-node-stats", help="Show traffic statistics by node"
    )
    packet_node_stats_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of nodes to display",
    )
    packet_node_stats_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )
    packet_node_stats_parser.set_defaults(func=cmd_packet_node_stats)

    # Packet hotspots command
    packet_hotspots_parser = subparsers.add_parser(
        "packet-hotspots", help="Find traffic hotspots in the NoC"
    )
    packet_hotspots_parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=5,
        help="Number of top hotspots to show (default: 5)",
    )
    packet_hotspots_parser.add_argument(
        "--metric",
        "-m",
        choices=["total_traffic", "rx_count", "tx_count", "unique_packets"],
        default="total_traffic",
        help="Metric to sort by (default: total_traffic)",
    )
    packet_hotspots_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )
    packet_hotspots_parser.set_defaults(func=cmd_packet_hotspots)

    # Packet cmd stats command
    packet_cmd_stats_parser = subparsers.add_parser(
        "packet-cmd-stats", help="Show statistics by command type"
    )
    packet_cmd_stats_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )
    packet_cmd_stats_parser.set_defaults(func=cmd_packet_cmd_stats)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
