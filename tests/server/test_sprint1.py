"""
Sprint 1 Integration Tests: Core Storage Engine

Tests cover:
  1.5.1 - Schema validation (normal + error cases)
  1.5.2 - SegmentData append → seal → is_full lifecycle
  1.5.3 - mark_deleted → is_deleted bitset operations
  1.5.4 - LSNGenerator thread safety
  1.5.5 - ReadSnapshot + is_visible visibility filtering
  1.5.6 - End-to-end: Insert → Seal → BuildIndex(HNSW) → Search

Ref: Technical_Design_VectorDB.md, task.md Sprint 1
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vectordb.server.schema import (
    CollectionSchema,
    FieldSchema,
    FieldType,
    MetricType,
    validate_schema,
)
from vectordb.server.segment import (
    CollectionEngine,
    Row,
    SegmentData,
    SegmentMeta,
    SegmentState,
)
from vectordb.server.lsn import LSNGenerator
from vectordb.server.mvcc import ReadSnapshot, acquire_snapshot, is_visible
from vectordb.server.index_strategy import (
    IndexStrategy,
    SearchCoordinator,
    SegmentIndex,
    brute_force_search,
)


# ============================================================
# Helper: create a valid test schema
# ============================================================

def _make_schema(
    dim: int = 128,
    collection_name: str = "test_collection",
) -> CollectionSchema:
    return CollectionSchema(
        collection_name=collection_name,
        fields=[
            FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
            FieldSchema(
                name="embedding",
                dtype=FieldType.VECTOR,
                dim=dim,
                metric=MetricType.L2,
            ),
            FieldSchema(name="category", dtype=FieldType.STRING),
        ],
    )


def _make_row(pk: int, dim: int = 128, lsn: int = 0) -> Row:
    return Row(
        pk=pk,
        vector=np.random.randn(dim).astype(np.float32),
        scalars={"category": f"cat_{pk % 5}"},
        lsn=lsn,
    )


# ============================================================
# 1.5.1 Schema Validation Tests
# ============================================================

class TestSchemaValidation:
    """Ref: task.md 1.5.1"""

    def test_valid_schema(self):
        """Normal case: 1 INT PK + 1 VECTOR + 1 STRING."""
        schema = _make_schema()
        validate_schema(schema)  # should not raise

    def test_no_primary_key(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="vec", dtype=FieldType.VECTOR, dim=128, metric=MetricType.L2),
            ],
        )
        with pytest.raises(ValueError, match="primary key"):
            validate_schema(schema)

    def test_multiple_primary_keys(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id1", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="id2", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="vec", dtype=FieldType.VECTOR, dim=128, metric=MetricType.L2),
            ],
        )
        with pytest.raises(ValueError, match="exactly one"):
            validate_schema(schema)

    def test_no_vector_field(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="text", dtype=FieldType.STRING),
            ],
        )
        with pytest.raises(ValueError, match="VECTOR"):
            validate_schema(schema)

    def test_duplicate_field_names(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="id", dtype=FieldType.STRING),
                FieldSchema(name="vec", dtype=FieldType.VECTOR, dim=128, metric=MetricType.L2),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            validate_schema(schema)

    def test_vector_missing_dim(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="vec", dtype=FieldType.VECTOR, metric=MetricType.L2),
            ],
        )
        with pytest.raises(ValueError, match="dim"):
            validate_schema(schema)

    def test_vector_missing_metric(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="vec", dtype=FieldType.VECTOR, dim=128),
            ],
        )
        with pytest.raises(ValueError, match="metric"):
            validate_schema(schema)

    def test_primary_key_vector_type(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(
                    name="vec", dtype=FieldType.VECTOR,
                    is_primary=True, dim=128, metric=MetricType.L2,
                ),
            ],
        )
        with pytest.raises(ValueError, match="INT or STRING"):
            validate_schema(schema)

    def test_reserved_field_name(self):
        schema = CollectionSchema(
            collection_name="test",
            fields=[
                FieldSchema(name="id", dtype=FieldType.INT, is_primary=True),
                FieldSchema(name="_lsn", dtype=FieldType.INT),
                FieldSchema(name="vec", dtype=FieldType.VECTOR, dim=128, metric=MetricType.L2),
            ],
        )
        with pytest.raises(ValueError, match="reserved"):
            validate_schema(schema)


# ============================================================
# 1.5.2 SegmentData Append → Seal → is_full
# ============================================================

class TestSegmentLifecycle:
    """Ref: task.md 1.5.2"""

    def test_append_and_fill(self):
        """Write up to max_rows → verify is_full() == True."""
        max_rows = 100
        schema = _make_schema(dim=8)
        meta = SegmentMeta(
            segment_id="seg_test",
            collection_name="test",
            state=SegmentState.GROWING,
            max_rows=max_rows,
        )
        seg = SegmentData(meta, schema)

        for i in range(max_rows):
            row = Row(
                pk=i,
                vector=np.random.randn(8).astype(np.float32),
                scalars={"category": f"c{i}"},
                lsn=i + 1,
            )
            offset = seg.append_row(row)
            assert offset == i

        assert seg.is_full()
        assert seg.meta.row_count == max_rows
        assert seg._write_cursor == max_rows

    def test_offset_increments(self):
        schema = _make_schema(dim=4)
        meta = SegmentMeta(
            segment_id="seg_off",
            collection_name="test",
            state=SegmentState.GROWING,
            max_rows=10,
        )
        seg = SegmentData(meta, schema)

        offsets = []
        for i in range(5):
            row = Row(pk=i, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=i)
            offsets.append(seg.append_row(row))

        assert offsets == [0, 1, 2, 3, 4]


# ============================================================
# 1.5.3 mark_deleted → is_deleted Bitset
# ============================================================

class TestDeleteBitset:
    """Ref: task.md 1.5.3"""

    def test_mark_and_check(self):
        schema = _make_schema(dim=4)
        meta = SegmentMeta(
            segment_id="seg_del",
            collection_name="test",
            state=SegmentState.GROWING,
            max_rows=100,
        )
        seg = SegmentData(meta, schema)

        # Write 20 rows
        for i in range(20):
            seg.append_row(Row(
                pk=i, vector=np.zeros(4, dtype=np.float32),
                scalars={}, lsn=i,
            ))

        # Delete every 3rd row
        deleted = [0, 3, 6, 9, 12, 15, 18]
        for d in deleted:
            seg.mark_deleted(d)

        for i in range(20):
            if i in deleted:
                assert seg.is_deleted(i), f"Row {i} should be deleted"
            else:
                assert not seg.is_deleted(i), f"Row {i} should NOT be deleted"

    def test_non_contiguous_deletes(self):
        schema = _make_schema(dim=4)
        meta = SegmentMeta(
            segment_id="seg_nc",
            collection_name="test",
            state=SegmentState.GROWING,
            max_rows=50,
        )
        seg = SegmentData(meta, schema)
        for i in range(50):
            seg.append_row(Row(
                pk=i, vector=np.zeros(4, dtype=np.float32),
                scalars={}, lsn=i,
            ))

        # Sparse pattern: 1, 7, 15, 31, 49
        for d in [1, 7, 15, 31, 49]:
            seg.mark_deleted(d)
            assert seg.is_deleted(d)

        # Others remain not deleted
        for i in [0, 2, 8, 14, 16, 30, 32, 48]:
            assert not seg.is_deleted(i)


# ============================================================
# 1.5.4 LSNGenerator Thread Safety
# ============================================================

class TestLSNGenerator:
    """Ref: task.md 1.5.4"""

    def test_thread_safety(self):
        """10 threads × 1000 calls → current() == 10000, no duplicates."""
        gen = LSNGenerator()
        results = []
        lock = threading.Lock()

        def worker():
            local = []
            for _ in range(1000):
                local.append(gen.next())
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert gen.current() == 10000
        assert len(results) == 10000
        # All LSNs must be unique
        assert len(set(results)) == 10000

    def test_monotonic(self):
        gen = LSNGenerator()
        prev = 0
        for _ in range(100):
            val = gen.next()
            assert val > prev
            prev = val


# ============================================================
# 1.5.5 ReadSnapshot + is_visible
# ============================================================

class TestSnapshotVisibility:
    """Ref: task.md 1.5.5"""

    def _make_segment_with_rows(self):
        schema = _make_schema(dim=4)
        meta = SegmentMeta(
            segment_id="seg_vis",
            collection_name="test",
            state=SegmentState.GROWING,
            max_rows=100,
        )
        seg = SegmentData(meta, schema)
        return seg

    def test_visible_row(self):
        seg = self._make_segment_with_rows()
        seg.append_row(Row(pk=1, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=1))
        snap = ReadSnapshot(read_lsn=1, active_cursor=1)
        assert is_visible(seg, 0, snap)

    def test_future_row_invisible(self):
        seg = self._make_segment_with_rows()
        seg.append_row(Row(pk=1, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=1))
        seg.append_row(Row(pk=2, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=2))
        snap = ReadSnapshot(read_lsn=1, active_cursor=2)
        assert is_visible(seg, 0, snap)      # lsn=1 ≤ 1
        assert not is_visible(seg, 1, snap)   # lsn=2 > 1

    def test_deleted_row_invisible(self):
        seg = self._make_segment_with_rows()
        seg.append_row(Row(pk=1, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=1))
        seg.mark_deleted(0)
        snap = ReadSnapshot(read_lsn=1, active_cursor=1)
        assert not is_visible(seg, 0, snap)

    def test_acquire_snapshot(self):
        seg = self._make_segment_with_rows()
        for i in range(5):
            seg.append_row(Row(pk=i, vector=np.zeros(4, dtype=np.float32), scalars={}, lsn=i + 1))
        gen = LSNGenerator(start=5)
        snap = acquire_snapshot(gen, seg)
        assert snap.read_lsn == 5
        assert snap.active_cursor == 5


# ============================================================
# 1.5.6 End-to-End: Insert → Seal → BuildIndex → Search
# ============================================================

class TestEndToEnd:
    """Ref: task.md 1.5.6"""

    @pytest.fixture
    def setup_engine(self):
        """Insert 1000 random 128-dim vectors into an engine."""
        np.random.seed(42)
        dim = 128
        schema = _make_schema(dim=dim)
        engine = CollectionEngine(schema)
        lsn_gen = LSNGenerator()

        all_vectors = []
        for i in range(1000):
            vec = np.random.randn(dim).astype(np.float32)
            all_vectors.append(vec)
            row = Row(
                pk=i,
                vector=vec,
                scalars={"category": f"cat_{i % 5}"},
                lsn=lsn_gen.next(),
            )
            engine.active_segment.append_row(row)
            engine.pk_map[row.pk] = (
                engine.active_segment.meta.segment_id, i
            )

        return engine, lsn_gen, np.array(all_vectors)

    def test_seal_changes_state(self, setup_engine):
        engine, _, _ = setup_engine
        old_seg = engine.active_segment
        engine.seal_active_segment()
        assert old_seg.meta.state == SegmentState.SEALED
        assert len(engine.sealed_segments) == 1
        assert engine.active_segment.meta.state == SegmentState.GROWING

    def test_build_hnsw_index(self, setup_engine):
        engine, _, _ = setup_engine
        engine.seal_active_segment()
        sealed = engine.sealed_segments[0]
        idx = SegmentIndex(sealed)
        idx.build_index({"M": 16, "ef_construction": 200, "ef_search": 50})
        assert sealed.meta.state == SegmentState.INDEXED
        assert idx.index is not None

    def test_search_matches_brute_force(self, setup_engine):
        engine, lsn_gen, all_vectors = setup_engine

        # Seal and build HNSW index
        engine.seal_active_segment()
        sealed = engine.sealed_segments[0]
        idx = SegmentIndex(sealed)
        idx.build_index({"M": 16, "ef_construction": 200, "ef_search": 50})
        sealed._index = idx  # Attach index to segment

        # Create search coordinator
        coord = SearchCoordinator(engine)
        snap = acquire_snapshot(lsn_gen, engine.active_segment)

        # Search with a random query
        query = np.random.randn(128).astype(np.float32)
        k = 10

        # HNSW search
        results = coord.search(query, k, snap, IndexStrategy.HNSW_PER_SEGMENT)
        hnsw_pks = {r["pk"] for r in results}

        # Brute-force reference (numpy)
        diffs = all_vectors - query.reshape(1, -1)
        distances = np.sum(diffs ** 2, axis=1)
        bf_top_k = np.argsort(distances)[:k]
        bf_pks = set(bf_top_k.tolist())

        # HNSW should match brute-force for exact recall
        assert len(results) == k
        # Allow minor discrepancies due to HNSW approximation
        overlap = len(hnsw_pks & bf_pks)
        assert overlap >= 8, (
            f"HNSW recall too low: {overlap}/10 overlap with brute-force"
        )

    def test_brute_force_search_direct(self, setup_engine):
        engine, lsn_gen, all_vectors = setup_engine
        seg = engine.active_segment
        snap = acquire_snapshot(lsn_gen, seg)

        query = np.random.randn(128).astype(np.float32)
        results = brute_force_search(seg, query, 10, snap.read_lsn)

        assert len(results) == 10
        # Verify results are sorted by distance
        dists = [r[1] for r in results]
        assert dists == sorted(dists)

    def test_collection_engine_pk_lookup(self, setup_engine):
        engine, _, _ = setup_engine
        # All PKs 0-999 should be in pk_map
        for pk in range(1000):
            assert pk in engine.pk_map
            seg_id, offset = engine.pk_map[pk]
            assert seg_id == engine.active_segment.meta.segment_id
            assert offset == pk
