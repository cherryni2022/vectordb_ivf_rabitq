"""
Tests for RaBitQ Quantization
"""
import pytest
import numpy as np
from vectordb.quantization.rabitq import RaBitQ


class TestRaBitQ:
    """Test suite for RaBitQ quantizer"""

    def test_train_and_encode_pq(self):
        """Test training and encoding with Product Quantization"""
        n_vectors = 500
        dimension = 128

        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = RaBitQ(
            dimension=dimension,
            nsubq=8,
            nbits=8,
            use_binary=False
        )
        quantizer.train(vectors)

        assert quantizer.is_trained
        assert quantizer.codebook.shape == (8, 256, 16)  # nsubq, 2^nbits, sub_dim

        # Encode
        codes = quantizer.encode(vectors)
        assert codes.shape == (n_vectors, 8)

    def test_train_and_encode_binary(self):
        """Test training and encoding with Binary Quantization"""
        n_vectors = 500
        dimension = 64

        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        quantizer = RaBitQ(
            dimension=dimension,
            nsubq=4,
            nbits=8,
            use_binary=True
        )
        quantizer.train(vectors)

        assert quantizer.is_trained

        # Encode
        codes = quantizer.encode(vectors)
        # Binary codes should have shape (n_vectors, nsubq * sub_dim)
        assert codes.shape[0] == n_vectors
        assert codes.dtype == np.uint8

    def test_distance_table(self):
        """Test computing distance table for a query"""
        vectors = np.random.randn(200, 64).astype(np.float32)

        quantizer = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)
        quantizer.train(vectors)

        query = vectors[0]
        distance_table = quantizer.compute_distance_table(query)

        # Shape should be (nsubq, 2^nbits)
        assert distance_table.shape == (8, 256)

    def test_asymmetric_distance(self):
        """Test computing asymmetric distances"""
        vectors = np.random.randn(200, 64).astype(np.float32)

        quantizer = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)
        quantizer.train(vectors)

        codes = quantizer.encode(vectors)
        query = vectors[0]
        distance_table = quantizer.compute_distance_table(query)

        distances = quantizer.compute_asymmetric_distance(codes[:100], distance_table)

        assert distances.shape == (100,)

    def test_decode_pq(self):
        """Test decoding PQ codes back to vectors"""
        vectors = np.random.randn(100, 64).astype(np.float32)

        quantizer = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)
        quantizer.train(vectors)

        codes = quantizer.encode(vectors)
        reconstructed = quantizer.decode(codes)

        # Shape should match
        assert reconstructed.shape == vectors.shape

        # Reconstruction should be approximate (not exact)
        mse = np.mean((vectors - reconstructed) ** 2)
        # MSE should be relatively small for well-trained quantizer
        assert mse < 100  # Reasonable threshold for random data

    def test_save_and_load(self):
        """Test saving and loading quantizer"""
        import tempfile
        import os

        vectors = np.random.randn(100, 64).astype(np.float32)

        quantizer = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)
        quantizer.train(vectors)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        temp_path = temp_file.name
        temp_file.close()

        try:
            quantizer.save(temp_path)

            # Load
            loaded = RaBitQ.load(temp_path)

            assert loaded.is_trained
            assert loaded.dimension == 64
            assert loaded.nsubq == 8

            # Codebooks should match
            assert np.allclose(loaded.codebook, quantizer.codebook)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stats(self):
        """Test quantizer statistics"""
        vectors = np.random.randn(200, 128).astype(np.float32)

        quantizer = RaBitQ(dimension=128, nsubq=8, nbits=8, use_binary=False)
        quantizer.train(vectors)

        stats = quantizer.get_stats()
        assert stats.dimension == 128
        assert stats.nsubq == 8
        assert stats.nbits == 8
        assert stats.is_trained == True
        assert stats.compression_ratio > 0
        # Compression should be significant
        assert stats.compression_ratio < 0.5

    def test_encode_without_training(self):
        """Test that encoding fails without training"""
        quantizer = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)

        vectors = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(RuntimeError, match="must be trained"):
            quantizer.encode(vectors)

    def test_hamming_distance_binary(self):
        """Test Hamming distance computation for binary codes"""
        # Create test binary codes
        codes_a = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
        codes_b = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)

        quantizer = RaBitQ(dimension=64, nsubq=2, nbits=2, use_binary=True)
        distances = quantizer.compute_hamming_distance(codes_a, codes_b)

        # Expected distances:
        # codes_a[0] vs codes_b[0] = 0 (same)
        # codes_a[0] vs codes_b[1] = 2 (bits differ)
        # codes_a[1] vs codes_b[0] = 4 (all bits differ)
        # codes_a[1] vs codes_b[1] = 2 (2 bits differ)
        assert distances[0, 0] == 0
        assert distances[0, 1] == 2
        assert distances[1, 0] == 4
        assert distances[1, 1] == 2

    def test_nbits_affects_codebook_size(self):
        """Test that nbits affects codebook size"""
        vectors = np.random.randn(100, 64).astype(np.float32)

        # nbits=4 => codebook size = 2^4 = 16
        q1 = RaBitQ(dimension=64, nsubq=8, nbits=4, use_binary=False)
        q1.train(vectors)
        assert q1.codebook.shape == (8, 16, 8)

        # nbits=8 => codebook size = 2^8 = 256
        q2 = RaBitQ(dimension=64, nsubq=8, nbits=8, use_binary=False)
        q2.train(vectors)
        assert q2.codebook.shape == (8, 256, 8)
