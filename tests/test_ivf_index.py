"""
Tests for IVF Index
"""
import pytest
import numpy as np
from vectordb.index.ivf_index import IVFIndex


class TestIVFIndex:
    """Test suite for IVFIndex"""

    def test_build_and_search(self):
        """Test basic build and search"""
        n_vectors = 1000
        dimension = 128
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        index = IVFIndex(nlist=50, nprobe=10, metric="l2")
        index.build(vectors)

        assert index.is_built
        assert index.dimension == dimension
        assert index.stats.total_vectors == n_vectors

        # Search
        query = vectors[0]
        results = index.search(query, k=5)

        assert len(results) <= 5
        assert results[0][0] == 0  # Index of the query vector

    def test_inner_product_metric(self):
        """Test inner product metric"""
        vectors = np.random.randn(100, 64).astype(np.float32)

        index = IVFIndex(nlist=10, nprobe=5, metric="ip")
        index.build(vectors)

        query = vectors[0]
        results = index.search(query, k=3)

        assert len(results) <= 3

    def test_nprobe_affects_search_quality(self):
        """Test that increasing nprobe improves search quality"""
        vectors = np.random.randn(500, 64).astype(np.float32)

        index = IVFIndex(nlist=50, nprobe=5)
        index.build(vectors)

        query = vectors[0]

        # Search with different nprobe values
        index.nprobe = 1
        results_1 = index.search(query, k=10)

        index.nprobe = 10
        results_10 = index.search(query, k=10)

        index.nprobe = 50
        results_50 = index.search(query, k=10)

        # More probes should find the exact match (distance ~ 0)
        assert results_1[0][1] >= results_10[0][1]
        assert results_10[0][1] >= results_50[0][1]

    def test_batch_search(self):
        """Test batch search"""
        vectors = np.random.randn(200, 64).astype(np.float32)

        index = IVFIndex(nlist=20, nprobe=5)
        index.build(vectors)

        queries = vectors[:10]
        results_list = index.search_batch(queries, k=3)

        assert len(results_list) == 10
        for i, results in enumerate(results_list):
            assert len(results) <= 3
            assert results[0][0] == i

    def test_add_vectors_to_existing_index(self):
        """Test adding vectors after index is built"""
        initial_vectors = np.random.randn(100, 64).astype(np.float32)

        index = IVFIndex(nlist=20, nprobe=5)
        index.build(initial_vectors)

        assert index.stats.total_vectors == 100

        # Add more vectors
        new_vectors = np.random.randn(50, 64).astype(np.float32)
        index.add_vectors(new_vectors)

        assert index.stats.total_vectors == 150

        # Search should still work
        query = initial_vectors[0]
        results = index.search(query, k=5)
        assert len(results) <= 5

    def test_save_and_load(self):
        """Test saving and loading index"""
        import tempfile
        import os

        vectors = np.random.randn(100, 64).astype(np.float32)

        index = IVFIndex(nlist=10, nprobe=5)
        index.build(vectors)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        temp_path = temp_file.name
        temp_file.close()

        try:
            index.save(temp_path)

            # Load
            loaded_index = IVFIndex.load(temp_path)

            assert loaded_index.is_built
            assert loaded_index.dimension == 64
            assert loaded_index.stats.total_vectors == 100

            # Search should work
            query = vectors[0]
            results = loaded_index.search(query, k=5)
            assert len(results) <= 5

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_more_vectors_than_clusters(self):
        """Test when nlist is larger than number of vectors"""
        vectors = np.random.randn(10, 64).astype(np.float32)

        index = IVFIndex(nlist=100, nprobe=5)
        index.build(vectors)

        # nlist should be adjusted
        assert index.nlist <= 10

    def test_empty_search(self):
        """Test search behavior with no results"""
        vectors = np.random.randn(50, 64).astype(np.float32)

        index = IVFIndex(nlist=10, nprobe=0)
        index.build(vectors)

        query = vectors[0]
        results = index.search(query, k=5)

        # With nprobe=0, should return empty results
        assert len(results) == 0

    def test_distance_computation(self):
        """Test internal distance computation"""
        index = IVFIndex(nlist=10, nprobe=5)

        # L2 distance
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        dist = index._compute_distance(a, b)
        assert dist < 1e-10

        c = np.array([2.0, 3.0, 4.0])
        dist = index._compute_distance(a, c)
        assert dist > 0

    def test_stats(self):
        """Test index statistics"""
        vectors = np.random.randn(200, 64).astype(np.float32)

        index = IVFIndex(nlist=20, nprobe=5)
        index.build(vectors)

        stats = index.get_stats()
        assert stats.total_vectors == 200
        assert stats.num_clusters == 20
        assert stats.avg_vectors_per_cluster == 10.0
