"""Packet trace analysis from FSIM log files."""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .models import PacketEvent, PacketTrace


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
