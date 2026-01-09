#!/usr/bin/env python3
"""
FSIM Log Analyzer Tool

A tool for parsing, analyzing, and monitoring FSIM log files.
Useful for detecting deadlocks and analyzing simulation behavior.
"""

import argparse
import fnmatch
import os
import sys
import time
from dataclasses import dataclass
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
        """
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.check_interval = check_interval
        self.deadlock_threshold = deadlock_threshold
        self.extensions = extensions
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.verbose = verbose

        self.file_statuses: dict[str, FileStatus] = {}
        self.last_any_change: float = time.time()
        self.monitoring_start: float = 0

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

        if result["changed_files"] and self.verbose:
            print(
                f"\n  Recently changed files ({len(result['changed_files'])}):"
            )
            for f in result["changed_files"][:5]:
                print(f"    - {f.name}")
            if len(result["changed_files"]) > 5:
                print(f"    ... and {len(result['changed_files']) - 5} more")

        print("\n" + "-" * 70)
        print("  Press Ctrl+C to stop monitoring")
        print("=" * 70)

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

        try:
            while True:
                result = self.check_once()
                self._print_status(result)

                if result["potential_deadlock"]:
                    deadlock_detected = True

                if (
                    duration
                    and (time.time() - self.monitoring_start) >= duration
                ):
                    print("\nMonitoring duration reached.")
                    break

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")

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


def _parse_patterns(pattern_str: Optional[str]) -> list[str]:
    """Parse comma-separated pattern string into list."""
    if not pattern_str:
        return []
    return [p.strip() for p in pattern_str.split(",") if p.strip()]


def cmd_monitor(args):
    """Handle the monitor command."""
    monitor = LogMonitor(
        log_dir=args.log_dir,
        check_interval=args.interval,
        deadlock_threshold=args.threshold,
        extensions=tuple(args.extensions.split(",")),
        include_patterns=_parse_patterns(args.include),
        exclude_patterns=_parse_patterns(args.exclude),
        verbose=args.verbose,
    )

    deadlock = monitor.monitor(duration=args.duration)
    sys.exit(1 if deadlock else 0)


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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
