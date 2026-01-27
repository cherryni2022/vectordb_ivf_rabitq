"""
Tests for VectorDB
"""
import pytest
import numpy as np
from vectordb import VectorDB, IVFIndexConfig, RaBitQConfig, VectorDBConfig


class TestVectorDB:
    """Test suite for VectorDB"""

    def test_basic_add_and_search(self):
        """Test basic add vectors and search"""
        # Create in-memory database
        db = VectorDB(use_memory_storage=True)

        # Create random vectors
        n_vectors = 100
        dimension = 128
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        # Add vectors
        db.add_vectors(vectors)

        # Build index
        db.build()

        # Search for similar vectors
        query = vectors[0]
        results = db.search(query, k=5)

        assert len(results.ids) <= 5
        assert len(results.distances) <= 5
        assert results.ids[0] == 0  # The query itself should be first
        assert results.distances[0] < 1e-5  # Distance to itself should be ~0

    def test_search_with_metadata(self):
        """Test search with metadata"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(50, 64).astype(np.float32)
        ids = [f"doc_{i}" for i in range(50)]
        metadata = [{"category": "A" if i < 25 else "B"} for i in range(50)]

        db.add_vectors(vectors, ids=ids, metadata=metadata)
        db.build()

        query = vectors[0]
        results = db.search(query, k=3)

        assert len(results.ids) == 3
        assert results.ids[0] == "doc_0"
        assert results.metadata[0]["category"] == "A"

    def test_batch_search(self):
        """Test batch search for multiple queries"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(100, 64).astype(np.float32)
        db.add_vectors(vectors)
        db.build()

        queries = vectors[:5]
        results_list = db.search_batch(queries, k=3)

        assert len(results_list) == 5
        for i, results in enumerate(results_list):
            assert len(results.ids) <= 3
            assert results.ids[0] == i

    def test_custom_config(self):
        """Test database with custom configuration"""
        config = VectorDBConfig(
            ivf=IVFIndexConfig(nlist=10, nprobe=3, metric="ip"),
            raq=RaBitQConfig(nsubq=4, nbits=8),
            dimension=64,
            use_quantization=False
        )

        db = VectorDB(config=config, use_memory_storage=True)

        vectors = np.random.randn(50, 64).astype(np.float32)
        db.add_vectors(vectors)
        db.build(use_quantization=False)

        query = vectors[0]
        results = db.search(query, k=5)

        assert len(results.ids) <= 5

    def test_save_and_load(self):
        """Test saving and loading the database"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            # Create database
            db = VectorDB(storage_path=temp_dir)
            vectors = np.random.randn(100, 64).astype(np.float32)
            ids = [i for i in range(100)]
            db.add_vectors(vectors, ids=ids)
            db.build()

            # Save
            db.save()

            # Create new database and load
            db2 = VectorDB(storage_path=temp_dir)
            db2.load()

            # Check if loaded correctly
            assert db2.is_built
            assert db2.dimension == 64

            # Search should work
            results = db2.search(vectors[0], k=5)
            assert len(results.ids) <= 5

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_vector(self):
        """Test retrieving individual vectors"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(50, 64).astype(np.float32)
        db.add_vectors(vectors, ids=[i for i in range(50)])
        db.build()

        # Get a vector by ID
        retrieved = db.get_vector(0)
        assert np.allclose(retrieved, vectors[0])

    def test_get_stats(self):
        """Test getting database statistics"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(100, 64).astype(np.float32)
        db.add_vectors(vectors, ids=[i for i in range(100)])
        db.build()

        stats = db.get_stats()

        assert stats["total_vectors"] == 100
        assert stats["dimension"] == 64
        assert stats["is_built"] == True
        assert stats["ivf_index"]["nlist"] == 100

    def test_nprobe_override(self):
        """Test overriding nprobe at search time"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(200, 64).astype(np.float32)
        db.add_vectors(vectors)
        db.build()

        query = vectors[0]

        # Search with different nprobe values
        results_1 = db.search(query, k=10, nprobe=1)
        results_10 = db.search(query, k=10, nprobe=10)
        results_50 = db.search(query, k=10, nprobe=50)

        # More probes should generally find better (smaller) distances
        assert results_10.distances[0] <= results_1.distances[0]
        assert results_50.distances[0] <= results_10.distances[0]

    def test_clear_database(self):
        """Test clearing the database"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(50, 64).astype(np.float32)
        db.add_vectors(vectors)
        db.build()

        assert db.is_built
        assert db.dimension == 64

        # Clear
        db.clear()

        assert not db.is_built
        assert db.dimension is None

    def test_dimension_validation(self):
        """Test that vector dimension is validated"""
        db = VectorDB(use_memory_storage=True)

        vectors1 = np.random.randn(10, 64).astype(np.float32)
        db.add_vectors(vectors1)

        # This should fail - different dimension
        vectors2 = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db.add_vectors(vectors2)

    def test_search_without_build(self):
        """Test that search fails if index is not built"""
        db = VectorDB(use_memory_storage=True)

        vectors = np.random.randn(10, 64).astype(np.float32)
        db.add_vectors(vectors)

        with pytest.raises(RuntimeError, match="Index not built"):
            db.search(vectors[0], k=5)


def test_create_vector_db_convenience():
    """Test the convenience function for creating VectorDB"""
    from vectordb.core.vector_db import create_vector_db

    db = create_vector_db(
        dimension=128,
        nlist=50,
        nprobe=5,
        use_quantization=True
    )

    vectors = np.random.randn(100, 128).astype(np.float32)
    db.add_vectors(vectors)
    db.build()

    results = db.search(vectors[0], k=5)
    assert len(results.ids) <= 5
