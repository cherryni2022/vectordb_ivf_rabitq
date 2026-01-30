"""
Tests for True RaBitQ Quantization
"""
import pytest
import numpy as np
from vectordb.quantization.true_rabitq import TrueRaBitQ, RaBitQWithRerank


class TestTrueRaBitQ:
    """Test suite for True RaBitQ quantizer"""

    def test_train_and_encode(self):
        """Test training and encoding with True RaBitQ"""
        n_vectors = 500
        dimension = 128

        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)

        assert quantizer.is_trained
        assert quantizer.rotation_matrix is not None
        assert quantizer.rotation_matrix.shape == (dimension, dimension)

        # Encode
        binary_codes, norms = quantizer.encode(vectors)
        
        # Check shapes
        assert binary_codes.shape == (n_vectors, (dimension + 7) // 8)  # packed bits
        assert norms.shape == (n_vectors,)
        assert binary_codes.dtype == np.uint8

    def test_compression_ratio(self):
        """Test that True RaBitQ achieves ~32x compression"""
        dimension = 128
        n_vectors = 100
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)
        quantizer.encode(vectors)

        stats = quantizer.get_stats()
        
        # Original: dimension * 4 bytes (float32)
        # Compressed: ceil(dimension/8) + 4 bytes (binary codes + norm)
        # For 128-dim: 512 bytes -> 16 + 4 = 20 bytes ≈ 25.6x compression
        # With just binary codes: 32x
        assert stats.bits_per_vector == dimension
        assert stats.compression_ratio < 0.1  # Should be around 3-4%

    def test_hamming_distance(self):
        """Test Hamming distance computation"""
        quantizer = TrueRaBitQ(dimension=64)
        
        # Test with known binary codes
        code_a = np.array([0b11110000, 0b00001111], dtype=np.uint8)
        codes_b = np.array([
            [0b11110000, 0b00001111],  # Same as a, distance = 0
            [0b00001111, 0b11110000],  # All bits different, distance = 16
            [0b11111111, 0b00001111],  # 4 bits different, distance = 4
        ], dtype=np.uint8)
        
        distances = quantizer.compute_hamming_distance(code_a, codes_b)
        
        assert distances[0] == 0
        assert distances[1] == 16
        assert distances[2] == 4

    def test_approximate_distance(self):
        """Test approximate L2 distance computation"""
        n_vectors = 200
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)
        quantizer.encode(vectors)

        query = vectors[0]
        approx_distances = quantizer.compute_approximate_distances(query)

        assert approx_distances.shape == (n_vectors,)
        
        # Self-distance should be close to 0
        assert approx_distances[0] < 1.0

        # All distances should be non-negative
        assert np.all(approx_distances >= 0)

    def test_approximate_distance_with_indices(self):
        """Test approximate distance computation with index subset"""
        n_vectors = 200
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)
        quantizer.encode(vectors)

        query = vectors[0]
        indices = np.array([0, 1, 5, 10, 50])
        
        approx_distances = quantizer.compute_approximate_distances(query, indices=indices)

        assert approx_distances.shape == (len(indices),)

    def test_approximate_inner_product(self):
        """Test approximate inner product computation"""
        n_vectors = 100
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)
        quantizer.encode(vectors)

        query = vectors[0]
        approx_ip = quantizer.compute_approximate_inner_product(query)

        assert approx_ip.shape == (n_vectors,)
        
        # Self inner product should be close to ||query||^2
        expected_ip = np.dot(query, query)
        # Allow for approximation error
        assert abs(approx_ip[0] - expected_ip) < expected_ip * 0.5

    def test_save_and_load(self):
        """Test saving and loading quantizer"""
        import tempfile
        import os

        n_vectors = 100
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension, random_seed=42)
        quantizer.train(vectors)
        quantizer.encode(vectors)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_path = temp_file.name
        temp_file.close()

        try:
            quantizer.save(temp_path)

            # Load
            loaded = TrueRaBitQ.load(temp_path)

            assert loaded.is_trained
            assert loaded.dimension == dimension
            assert loaded.binary_codes is not None
            assert loaded.norms is not None

            # Rotation matrix should match
            assert np.allclose(loaded.rotation_matrix, quantizer.rotation_matrix)

            # Binary codes should match
            assert np.array_equal(loaded.binary_codes, quantizer.binary_codes)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_encode_without_training(self):
        """Test that encoding fails without training"""
        quantizer = TrueRaBitQ(dimension=64)
        vectors = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(RuntimeError, match="must be trained"):
            quantizer.encode(vectors)

    def test_orthogonality_of_rotation_matrix(self):
        """Test that rotation matrix is orthogonal"""
        dimension = 64
        vectors = np.random.randn(100, dimension).astype(np.float32)

        quantizer = TrueRaBitQ(dimension=dimension)
        quantizer.train(vectors)

        R = quantizer.rotation_matrix
        # R @ R.T should be close to identity
        RRT = np.dot(R, R.T)
        identity = np.eye(dimension)
        
        assert np.allclose(RRT, identity, atol=1e-5)


class TestRaBitQWithRerank:
    """Test suite for RaBitQ with Reranking"""

    def test_search_basic(self):
        """Test basic search functionality"""
        n_vectors = 500
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        searcher = RaBitQWithRerank(dimension=dimension, rerank_factor=10)
        searcher.train(vectors)
        searcher.encode(vectors)

        query = vectors[0]
        indices, distances = searcher.search(query, k=10)

        assert len(indices) == 10
        assert len(distances) == 10
        
        # First result should be the query itself
        assert indices[0] == 0
        assert distances[0] < 1e-5

    def test_search_with_candidates(self):
        """Test search within a subset of candidates"""
        n_vectors = 500
        dimension = 64
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        searcher = RaBitQWithRerank(dimension=dimension, rerank_factor=10)
        searcher.train(vectors)
        searcher.encode(vectors)

        query = vectors[0]
        candidate_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        indices, distances = searcher.search(query, k=5, candidate_indices=candidate_indices)

        assert len(indices) == 5
        # All results should be from the candidate set
        assert all(idx in candidate_indices for idx in indices)

    def test_recall_quality(self):
        """Test that reranking improves recall quality"""
        n_vectors = 1000
        dimension = 64
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        searcher = RaBitQWithRerank(dimension=dimension, rerank_factor=10)
        searcher.train(vectors)
        searcher.encode(vectors)

        # Compute ground truth
        query = vectors[0]
        exact_distances = np.sum((vectors - query) ** 2, axis=1)
        ground_truth = np.argsort(exact_distances)[:10]

        # Search
        indices, _ = searcher.search(query, k=10)

        # Calculate recall
        recall = len(set(indices) & set(ground_truth)) / 10
        print(f"Recall: {recall}")
        
        # With reranking, recall should be very high
        assert recall >= 0.8, f"Recall too low: {recall}"


class TestIVFRaBitQIntegration:
    """Test IVF + RaBitQ integration via VectorDB"""

    def test_vectordb_accelerated_search(self):
        """Test accelerated search through VectorDB"""
        from vectordb import VectorDB

        n_vectors = 500
        dimension = 64
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        db = VectorDB(use_memory_storage=True)
        db.add_vectors(vectors)
        db.build(use_true_rabitq=True)

        # Verify True RaBitQ is enabled
        stats = db.get_stats()
        assert stats["accelerated_search_enabled"] is True
        assert stats["true_rabitq"] is not None

        # Search with acceleration
        query = vectors[0]
        results_accel = db.search(query, k=10, accelerated=True)

        assert len(results_accel.ids) == 10
        assert results_accel.ids[0] == 0  # First result should be the query

    def test_accelerated_vs_exact_search(self):
        """Compare accelerated search with exact search"""
        from vectordb import VectorDB

        n_vectors = 500
        dimension = 64
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        db = VectorDB(use_memory_storage=True)
        db.add_vectors(vectors)
        db.build(use_true_rabitq=True)

        query = vectors[0]

        # Accelerated search
        results_accel = db.search(query, k=10, accelerated=True)

        # Exact search (disable acceleration)
        results_exact = db.search(query, k=10, accelerated=False)

        # Both should find the same first result
        assert results_accel.ids[0] == results_exact.ids[0] == 0

        # Calculate overlap
        accel_set = set(results_accel.ids)
        exact_set = set(results_exact.ids)
        overlap = len(accel_set & exact_set) / 10

        # Should have good overlap
        assert overlap >= 0.7, f"Low overlap between accelerated and exact: {overlap}"

    def test_compression_stats(self):
        """Test compression statistics"""
        from vectordb import VectorDB

        dimension = 128
        n_vectors = 100
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        db = VectorDB(use_memory_storage=True)
        db.add_vectors(vectors)
        db.build(use_true_rabitq=True)

        stats = db.get_stats()
        
        # Check True RaBitQ stats
        assert stats["true_rabitq"]["bits_per_vector"] == dimension
        assert stats["true_rabitq"]["n_vectors_encoded"] == n_vectors
        
        # Compression ratio should be around 3-4% (including norm storage)
        assert stats["true_rabitq"]["compression_ratio"] < 0.1
