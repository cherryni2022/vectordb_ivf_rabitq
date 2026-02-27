"""
LSN (Log Sequence Number) generator for MVCC versioning.

Ref: Technical_Design_VectorDB.md §7.1 全局 LSN 发号器 (L1186-L1198)
"""

import threading


class LSNGenerator:
    """Global monotonically increasing LSN generator (thread-safe).

    Every write operation is assigned a unique, increasing LSN via next().
    Read snapshots use current() to capture the latest visible version.

    Ref: Technical Design §7.1 L1186-L1198
    """

    def __init__(self, start: int = 0) -> None:
        self._counter = start
        self._lock = threading.Lock()

    def next(self) -> int:
        """Atomically increment and return a new LSN."""
        with self._lock:
            self._counter += 1
            return self._counter

    def current(self) -> int:
        """Return the current LSN value (lock-free read)."""
        return self._counter
