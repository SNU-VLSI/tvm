"""Data models for FSIM Log Analyzer."""

from dataclasses import dataclass, field
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


@dataclass
class SyncEvent:
    """Represents a synchronization event (flag/standby)."""
    timestamp: int
    node: str
    node_type: str  # "inode" or "imce"
    event_type: str  # "STANDBY_STALL_START", "STANDBY_STALL_END", "SETFLAG", etc.
    details: str
    raw_line: str
