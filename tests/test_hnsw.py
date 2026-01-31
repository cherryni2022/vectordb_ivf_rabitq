"""
Tests for HNSW Index Implementation
"""
import numpy as np
import pytest
import tempfile
import os

from vectordb.core.config import HNSWIndexConfig, VectorDBConfig
from vectordb.core.vector_db import VectorDB
from vectordb.index.hnsw_index import HNSWIndex, HNSWIndexStats


class TestHNSWIndex:
    """Test HNSW index functionality"""
    
    @pytest.fixture
    def hnsw_config(self):
        return HNSWIndexConfig(M=16, ef_construction=100, ef_search=50)
    
    @pytest.fixture
    def sample_vectors(self):
        np.random.seed(42)
        return np.random.random((100, 128)).astype(np.float32)
    
    def test_initialization(self, hnsw_config):
        """Test HNSW index initialization"""
        index = HNSWIndex(hnsw_config)
        assert index.M == 16
        assert index.M0 == 32  # M0 = M * 2
        assert index.ef_construction == 100
        assert index.ef_search == 50
        assert index.is_built is False
        assert index.stats.total_vectors == 0
    
    def test_build_and_search(self, hnsw_config, sample_vectors):
        """Test building index and searching"""
        index = HNSWIndex(hnsw_config)
        index.build(sample_vectors)
        
        assert index.is_built is True
        assert index.stats.total_vectors == 100
        assert index.max_level >= 0
        
        # Test searching for an existing vector
        query = sample_vectors[0]
        results = index.search(query, k=5)
        
        assert len(results) == 5
        # The first result should be the vector itself (dist ~ 0)
        assert results[0][0] == 0
        assert results[0][1] < 1e-5  # Distance should be approximately 0
    
    def test_search_quality(self, hnsw_config, sample_vectors):
        """Test search quality against brute force"""
        index = HNSWIndex(hnsw_config)
        index.build(sample_vectors)
        
        # Random query
        np.random.seed(123)
        query = np.random.random(128).astype(np.float32)
        
        # HNSW search
        hnsw_results = index.search(query, k=10)
        hnsw_ids = set(r[0] for r in hnsw_results)
        
        # Brute force search
        distances = np.sum((sample_vectors - query) ** 2, axis=1)
        brute_force_ids = set(np.argsort(distances)[:10])
        
        # Calculate recall
        recall = len(hnsw_ids & brute_force_ids) / 10
        assert recall >= 0.8, f"Recall {recall} is too low"
    
    def test_persistence(self, hnsw_config, sample_vectors):
        """Test save and load functionality"""
        index = HNSWIndex(hnsw_config)
        index.build(sample_vectors)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name
        
        try:
            # Save
            index.save(save_path)
            
            # Load
            loaded_index = HNSWIndex.load(save_path)
            
            assert loaded_index.stats.total_vectors == 100
            assert loaded_index.max_level == index.max_level
            assert loaded_index.is_built is True
            
            # Verify search on loaded index
            query = sample_vectors[0]
            results = loaded_index.search(query, k=5)
            assert results[0][0] == 0
        finally:
            os.unlink(save_path)
    
    def test_incremental_insertion(self, hnsw_config):
        """Test adding vectors incrementally"""
        index = HNSWIndex(hnsw_config)
        vectors = np.random.random((20, 32)).astype(np.float32)
        
        # Add first batch
        index.add_vectors(vectors[:10])
        assert index.stats.total_vectors == 10
        
        # Add second batch
        index.add_vectors(vectors[10:])
        assert index.stats.total_vectors == 20
        
        # Search for vector from second batch
        query = vectors[15]
        res = index.search(query, k=1)
        assert res[0][0] == 15
        assert res[0][1] < 1e-5


class TestVectorDBWithHNSW:
    """Test VectorDB integration with HNSW index"""
    
    def test_vectordb_hnsw_integration(self):
        """Test VectorDB with HNSW index type"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create DB with HNSW index
            config = VectorDBConfig(
                index_type="hnsw",
                hnsw=HNSWIndexConfig(M=8, ef_construction=50, ef_search=30)
            )
            db = VectorDB(config=config, storage_path=tmp_dir)
            
            # Add vectors
            np.random.seed(42)
            vectors = np.random.random((50, 64)).astype(np.float32)
            db.add_vectors(vectors)
            db.build(use_quantization=False, use_true_rabitq=False)
            
            assert db.is_built is True
            stats = db.get_stats()
            assert stats["hnsw_index"]["total_vectors"] == 50
            
            # Search
            query = vectors[0]
            results = db.search(query, k=5)
            
            assert len(results.ids) == 5
            assert results.distances[0] < 1e-5  # First match should be exact
    
    def test_hnsw_vs_ivf_consistency(self):
        """Test that both index types return similar results"""
        np.random.seed(42)
        vectors = np.random.random((100, 64)).astype(np.float32)
        query = vectors[0]
        
        with tempfile.TemporaryDirectory() as tmp_dir1, \
             tempfile.TemporaryDirectory() as tmp_dir2:
            
            # IVF index
            ivf_config = VectorDBConfig(index_type="ivf")
            ivf_db = VectorDB(config=ivf_config, storage_path=tmp_dir1)
            ivf_db.add_vectors(vectors)
            ivf_db.build(use_quantization=False, use_true_rabitq=False)
            ivf_results = ivf_db.search(query, k=10)
            
            # HNSW index
            hnsw_config = VectorDBConfig(
                index_type="hnsw",
                hnsw=HNSWIndexConfig(M=16, ef_construction=100, ef_search=50)
            )
            hnsw_db = VectorDB(config=hnsw_config, storage_path=tmp_dir2)
            hnsw_db.add_vectors(vectors)
            hnsw_db.build(use_quantization=False, use_true_rabitq=False)
            hnsw_results = hnsw_db.search(query, k=10)
            
            # Both should find the exact match as first result
            assert ivf_results.distances[0] < 1e-5
            assert hnsw_results.distances[0] < 1e-5
            
            # Calculate overlap in top-10
            ivf_ids = set(ivf_results.ids)
            hnsw_ids = set(hnsw_results.ids)
            overlap = len(ivf_ids & hnsw_ids)
            assert overlap >= 8, f"Too little overlap: {overlap}"


class TestHNSWMetrics:
    """Test HNSW with different distance metrics"""
    
    def test_l2_metric(self):
        """Test with L2 (Euclidean) distance"""
        config = HNSWIndexConfig(M=16, ef_construction=50, metric="l2")
        index = HNSWIndex(config)
        
        vectors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        index.build(vectors)
        
        query = np.array([0.1, 0.1], dtype=np.float32)
        results = index.search(query, k=2)
        
        # Closest should be [0, 0]
        assert results[0][0] == 0
    
    def test_ip_metric(self):
        """Test with inner product similarity"""
        config = HNSWIndexConfig(M=16, ef_construction=50, metric="ip")
        index = HNSWIndex(config)
        
        # Normalize vectors for meaningful IP comparison
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        index.build(vectors)
        
        query = np.array([1, 0], dtype=np.float32)
        results = index.search(query, k=2)
        
        # Most similar should be [1, 0] (index 0)
        assert results[0][0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
