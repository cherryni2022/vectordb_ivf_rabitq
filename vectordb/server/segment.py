"""
Storage engine: Row, SegmentData, SegmentMeta, SegmentState, CollectionEngine.

Ref: Technical_Design_VectorDB.md
  §2.1 核心概念: Row 与 Segment (L87-L125)
  §2.2 物理数据布局 (L127-L207)
  §2.3 Collection 级别的 Segment 管理 (L218-L246)
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from vectordb.server.schema import CollectionSchema, FieldType


class SegmentState(Enum):
    """Lifecycle states for a Segment.

    Ref: Technical Design L121-L124
    """
    GROWING = "growing"    # Actively receiving writes (MemTable)
    SEALED = "sealed"      # Closed for writes, not yet indexed
    INDEXED = "indexed"    # Closed and vector index built


@dataclass
class SegmentMeta:
    """Metadata for a single Segment.

    Ref: Technical Design L111-L119
    """
    segment_id: str
    collection_name: str
    state: SegmentState
    row_count: int = 0
    max_rows: int = 100_000   # threshold, seal when reached
    min_lsn: int = 0          # minimum LSN in this segment
    max_lsn: int = 0          # maximum LSN in this segment


@dataclass
class Row:
    """Internal representation of a single data row.

    Ref: Technical Design L96-L103
    """
    pk: Union[int, str]                   # primary key value
    vector: np.ndarray                    # shape=(dim,), dtype=float32
    scalars: Dict[str, Union[int, str]]   # {field_name: value}
    lsn: int = 0                          # assigned at write time


class SegmentData:
    """In-memory columnar storage for a Segment.

    Uses pre-allocated numpy matrix for vectors to avoid O(N×dim)
    copy on each insert (see Technical Design §2.2.2).

    Ref: Technical Design §2.2.1 L149-L207
    """

    def __init__(self, meta: SegmentMeta, schema: CollectionSchema) -> None:
        self.meta = meta
        self.schema = schema
        dim = schema.vector_field().dim

        # ====== Columnar storage ======
        # Vector column: pre-allocated fixed matrix
        self.vectors: np.ndarray = np.zeros(
            (meta.max_rows, dim), dtype=np.float32
        )
        # Primary key column
        self.pks: List[Union[int, str]] = []
        # Scalar columns: keyed by field name
        self.scalar_columns: Dict[str, List[Union[int, str]]] = {
            f.name: []
            for f in schema.fields
            if f.dtype != FieldType.VECTOR and not f.is_primary
        }
        # Hidden column: per-row write version (LSN)
        self.lsns: np.ndarray = np.zeros(meta.max_rows, dtype=np.int64)
        # Delete bitmap: bit i == 1 means row i is soft-deleted
        self.delete_bitset: bytearray = bytearray(
            (meta.max_rows + 7) // 8
        )
        # Current write cursor
        self._write_cursor: int = 0

    def append_row(self, row: Row) -> int:
        """Append a row, return the offset within this Segment.

        Ref: Technical Design L182-L192
        """
        offset = self._write_cursor
        self.vectors[offset] = row.vector
        self.pks.append(row.pk)
        for fname, val in row.scalars.items():
            if fname in self.scalar_columns:
                self.scalar_columns[fname].append(val)
        self.lsns[offset] = row.lsn

        self._write_cursor += 1
        self.meta.row_count = self._write_cursor

        # Update LSN range
        if self.meta.min_lsn == 0 or row.lsn < self.meta.min_lsn:
            self.meta.min_lsn = row.lsn
        if row.lsn > self.meta.max_lsn:
            self.meta.max_lsn = row.lsn

        return offset

    def mark_deleted(self, offset: int) -> None:
        """Soft-delete row at offset via bitmap.

        Ref: Technical Design L194-L198
        """
        byte_idx = offset >> 3        # offset // 8
        bit_idx = offset & 0x07       # offset % 8
        self.delete_bitset[byte_idx] |= (1 << bit_idx)

    def is_deleted(self, offset: int) -> bool:
        """Check if row at offset is soft-deleted.

        Ref: Technical Design L200-L203
        """
        byte_idx = offset >> 3
        bit_idx = offset & 0x07
        return bool(self.delete_bitset[byte_idx] & (1 << bit_idx))

    def is_full(self) -> bool:
        """Check if the segment has reached max_rows.

        Ref: Technical Design L205-L206
        """
        return self._write_cursor >= self.meta.max_rows


class CollectionEngine:
    """Manages all Segments for a single Collection.

    Ref: Technical Design §2.3 L220-L246
    """

    def __init__(self, schema: CollectionSchema) -> None:
        self.schema = schema
        # Current writable segment (GROWING), exactly one
        self.active_segment: SegmentData = self._new_segment()
        # Closed segments list
        self.sealed_segments: List[SegmentData] = []
        # Global PK → (segment_id, offset) mapping
        self.pk_map: Dict[Union[int, str], Tuple[str, int]] = {}

    def _new_segment(self) -> SegmentData:
        """Create a new GROWING segment with a unique ID."""
        seg_id = f"seg_{uuid4().hex[:8]}"
        meta = SegmentMeta(
            segment_id=seg_id,
            collection_name=self.schema.collection_name,
            state=SegmentState.GROWING,
        )
        return SegmentData(meta, self.schema)

    def seal_active_segment(self) -> None:
        """Seal the current active segment and create a new one.

        Ref: Technical Design L241-L246
        """
        self.active_segment.meta.state = SegmentState.SEALED
        self.sealed_segments.append(self.active_segment)
        self.active_segment = self._new_segment()
