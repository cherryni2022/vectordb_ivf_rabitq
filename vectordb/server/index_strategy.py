"""
Dual index strategy: HNSWLib Per-Segment / IVF+RaBitQ Global.

Sprint 1 implements the HNSW path + brute-force + SearchCoordinator.
IVF+RaBitQ global index is stubbed for Sprint 3.

Ref: Technical_Design_VectorDB.md
  §3.1-3.2 HNSWLib Per-Segment (L250-L366)
  §3.4     暴力搜索 (L543-L583)
  §3.5     搜索协调器 (L585-L706)
"""

from __future__ import annotations

import heapq
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None  # type: ignore[assignment]

from vectordb.server.segment import (
    CollectionEngine,
    SegmentData,
    SegmentState,
)
from vectordb.server.mvcc import ReadSnapshot, is_visible


class IndexStrategy(Enum):
    """Collection-level index strategy.

    Ref: Technical Design L254-L259
    """
    HNSW_PER_SEGMENT = "hnsw"          # Each Segment gets its own HNSWLib index
    IVF_RABITQ_GLOBAL = "ivf_rabitq"   # Global IVF + RaBitQ index


class SegmentIndex:
    """Per-Segment HNSWLib vector index.

    Ref: Technical Design §3.2.1 L303-L341
    """

    def __init__(self, segment: SegmentData) -> None:
        self.segment = segment
        self.index: Optional[Any] = None   # hnswlib.Index
        self.index_type: str = "hnsw"
        self.index_params: Dict[str, Any] = {}

    def build_index(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Build an HNSWLib index on the segment's vectors.

        The segment must be in SEALED state.
        Labels are set to segment offsets (0..N-1).

        Ref: Technical Design §3.2.1 L312-L333
        """
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for HNSW indexing. "
                "Install with: pip install hnswlib"
            )

        if params is None:
            params = {}

        vectors = self.segment.vectors[:self.segment.meta.row_count]
        n = self.segment.meta.row_count
        dim = vectors.shape[1]

        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(
            max_elements=n,
            ef_construction=params.get('ef_construction', 200),
            M=params.get('M', 16),
        )
        # Labels = segment offsets (direct mapping, no extra mapping table)
        self.index.add_items(vectors, ids=np.arange(n))
        self.index.set_ef(params.get('ef_search', 50))

        self.segment.meta.state = SegmentState.INDEXED
        self.index_params = params

    def search(
        self, query: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        """Search the index, returning [(offset_in_segment, distance)].

        Caller is responsible for filtering via delete_bitset and LSN.

        Ref: Technical Design §3.2.1 L334-L341
        """
        if self.index is None:
            raise RuntimeError("Index not built yet")
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)
        return list(zip(labels[0].tolist(), distances[0].tolist()))


def brute_force_search(
    segment: SegmentData,
    query: np.ndarray,      # shape=(dim,)
    k: int,
    read_lsn: int,
) -> List[Tuple[int, float]]:
    """Brute-force L2 search with delete/LSN filtering.

    Used for Active Segments and un-indexed Sealed Segments.
    Returns [(offset, distance)] sorted by distance ascending.

    Ref: Technical Design §3.4 L548-L583
    """
    n = segment._write_cursor
    if n == 0:
        return []

    vectors = segment.vectors[:n]   # shape=(n, dim)
    # L2 distance: ||q - v||^2
    diffs = vectors - query.reshape(1, -1)  # broadcast
    distances = np.sum(diffs ** 2, axis=1)  # shape=(n,)

    # Filter deleted and invisible rows
    valid_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if segment.is_deleted(i) or segment.lsns[i] > read_lsn:
            valid_mask[i] = False

    valid_indices = np.where(valid_mask)[0]
    valid_distances = distances[valid_indices]

    if len(valid_indices) == 0:
        return []

    # Top-k via argpartition + argsort
    top_count = min(k, len(valid_indices))
    top_local = np.argpartition(valid_distances, top_count)[:top_count]
    top_local = top_local[np.argsort(valid_distances[top_local])]

    return [
        (int(valid_indices[i]), float(valid_distances[i]))
        for i in top_local
    ]


class SearchCoordinator:
    """Dispatches search requests across segments and merges results.

    Ref: Technical Design §3.5 L587-L706
    """

    def __init__(self, engine: CollectionEngine) -> None:
        self.engine = engine

    def search(
        self,
        query: np.ndarray,
        k: int,
        snap: ReadSnapshot,
        index_strategy: IndexStrategy,
    ) -> List[dict]:
        """Unified search entry point.

        Routes to HNSW or IVF path based on index_strategy.
        Returns List[{"pk": ..., "distance": ..., "scalars": {...}}]

        Ref: Technical Design §3.5 L596-L618
        """
        all_candidates: List[Tuple[float, str, int]] = []

        if index_strategy == IndexStrategy.HNSW_PER_SEGMENT:
            all_candidates = self._search_hnsw(query, k, snap)
        elif index_strategy == IndexStrategy.IVF_RABITQ_GLOBAL:
            all_candidates = self._search_ivf_rabitq(query, k, snap)

        # Merge top_k (ascending by distance)
        top_k = heapq.nsmallest(k, all_candidates, key=lambda x: x[0])

        # Fill in original data
        return self._assemble_results(top_k)

    def _search_hnsw(
        self,
        query: np.ndarray,
        k: int,
        snap: ReadSnapshot,
    ) -> List[Tuple[float, str, int]]:
        """HNSWLib Per-Segment search path (Scatter-Gather).

        Ref: Technical Design §3.5 L620-L645
        """
        candidates: List[Tuple[float, str, int]] = []

        # 1. Active Segment → brute-force search
        bf_results = brute_force_search(
            self.engine.active_segment, query, k, snap.read_lsn
        )
        seg_id = self.engine.active_segment.meta.segment_id
        for offset, dist in bf_results:
            candidates.append((dist, seg_id, offset))

        # 2. Each Sealed / Indexed Segment
        for seg in self.engine.sealed_segments:
            if (
                seg.meta.state == SegmentState.INDEXED
                and hasattr(seg, '_index')
                and seg._index is not None
            ):
                results = seg._index.search(query, k)
                for offset, dist in results:
                    if is_visible(seg, offset, snap):
                        candidates.append((dist, seg.meta.segment_id, offset))
            else:
                # SEALED but not indexed → brute-force
                bf_results = brute_force_search(
                    seg, query, k, snap.read_lsn
                )
                for offset, dist in bf_results:
                    candidates.append((dist, seg.meta.segment_id, offset))

        return candidates

    def _search_ivf_rabitq(
        self,
        query: np.ndarray,
        k: int,
        snap: ReadSnapshot,
    ) -> List[Tuple[float, str, int]]:
        """IVF+RaBitQ global index search path.

        Sprint 1 stub — full implementation in Sprint 3.
        Falls back to brute-force across all segments.
        """
        # Fallback: brute-force all segments
        candidates: List[Tuple[float, str, int]] = []

        # Active Segment
        bf_results = brute_force_search(
            self.engine.active_segment, query, k, snap.read_lsn
        )
        seg_id = self.engine.active_segment.meta.segment_id
        for offset, dist in bf_results:
            candidates.append((dist, seg_id, offset))

        # All Sealed Segments
        for seg in self.engine.sealed_segments:
            bf_results = brute_force_search(seg, query, k, snap.read_lsn)
            for offset, dist in bf_results:
                candidates.append((dist, seg.meta.segment_id, offset))

        return candidates

    def _assemble_results(
        self, top_k: List[Tuple[float, str, int]]
    ) -> List[dict]:
        """Fill in pk and scalar data from segment storage.

        Ref: Technical Design §3.5 L684-L697
        """
        results = []
        for dist, seg_id, offset in top_k:
            seg = self._find_segment(seg_id)
            results.append({
                "pk": seg.pks[offset],
                "distance": dist,
                "scalars": {
                    fname: col[offset]
                    for fname, col in seg.scalar_columns.items()
                },
            })
        return results

    def _find_segment(self, seg_id: str) -> SegmentData:
        """Locate a segment by ID."""
        if self.engine.active_segment.meta.segment_id == seg_id:
            return self.engine.active_segment
        for seg in self.engine.sealed_segments:
            if seg.meta.segment_id == seg_id:
                return seg
        raise ValueError(f"Segment {seg_id} not found")
