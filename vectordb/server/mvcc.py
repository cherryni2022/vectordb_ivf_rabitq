"""
MVCC snapshot read support: ReadSnapshot, acquire_snapshot, is_visible.

Ref: Technical_Design_VectorDB.md §7.2 Snapshot Read 实现 (L1204-L1230)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectordb.server.segment import SegmentData
    from vectordb.server.lsn import LSNGenerator


@dataclass
class ReadSnapshot:
    """Snapshot context for a single query request.

    Ref: Technical Design §7.2 L1204-L1208
    """
    read_lsn: int       # Maximum LSN visible to this query
    active_cursor: int   # Active Segment write cursor at snapshot time


def acquire_snapshot(
    lsn_gen: LSNGenerator,
    active_seg: SegmentData,
) -> ReadSnapshot:
    """Capture a point-in-time read snapshot.

    Ref: Technical Design §7.2 L1210-L1216
    """
    return ReadSnapshot(
        read_lsn=lsn_gen.current(),
        active_cursor=active_seg._write_cursor,
    )


def is_visible(
    segment: SegmentData,
    offset: int,
    snap: ReadSnapshot,
) -> bool:
    """Determine if row at offset is visible under the given snapshot.

    Visibility requires both:
      1. Row LSN ≤ snapshot read_lsn (not "future" data)
      2. Row is not soft-deleted

    Ref: Technical Design §7.2 L1221-L1229
    """
    # Condition 1: not from the future
    if segment.lsns[offset] > snap.read_lsn:
        return False
    # Condition 2: not deleted
    if segment.is_deleted(offset):
        return False
    return True
