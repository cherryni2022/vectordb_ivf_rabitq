"""
Tests for HNSWLib Index Implementation
"""
import numpy as np
import pytest
import tempfile
import os

# Skip all tests if hnswlib not available
hnswlib = pytest.importorskip("hnswlib")

from vectordb.core.config import HNSWIndexConfig
from vectordb.index.hnswlib_index import HNSWLibIndex


class TestHNSWLibIndex:
    """Test HNSWLib index functionality"""
    
    @pytest.fixture
    def config(self):
        return HNSWIndexConfig(M=16, ef_construction=100, ef_search=50)
    
    @pytest.fixture
    def sample_vectors(self):
        np.random.seed(42)
        return np.random.random((100, 128)).astype(np.float32)
    
    def test_initialization(self, config):
        """Test index initialization"""
        index = HNSWLibIndex(config)
        assert index.M == 16
        assert index.ef_construction == 100
        assert index.ef_search == 50
        assert index.is_built is False
    
    def test_build_and_search(self, config, sample_vectors):
        """Test building index and searching"""
        index = HNSWLibIndex(config)
        index.build(sample_vectors)
        
        assert index.is_built is True
        assert index.stats.total_vectors == 100
        
        # Search for existing vector
        query = sample_vectors[0]
        results = index.search(query, k=5)
        
        assert len(results) == 5
        assert results[0][0] == 0  # First result is the query itself
        assert results[0][1] < 1e-5  # Distance ~ 0
    
    def test_search_quality(self, config, sample_vectors):
        """Test search recall against brute force"""
        index = HNSWLibIndex(config)
        index.build(sample_vectors)
        
        np.random.seed(123)
        query = np.random.random(128).astype(np.float32)
        
        # HNSWLib search
        results = index.search(query, k=10)
        hnsw_ids = set(r[0] for r in results)
        
        # Brute force
        distances = np.sum((sample_vectors - query) ** 2, axis=1)
        bf_ids = set(np.argsort(distances)[:10])
        
        recall = len(hnsw_ids & bf_ids) / 10
        assert recall >= 0.9, f"Recall {recall} too low"
    
    def test_batch_search(self, config, sample_vectors):
        """Test batch search"""
        index = HNSWLibIndex(config)
        index.build(sample_vectors)
        
        queries = sample_vectors[:5]
        results = index.search_batch(queries, k=3)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert len(result) == 3
            assert result[0][0] == i  # First match is the query
    
    def test_incremental_insertion(self, config):
        """Test adding vectors incrementally"""
        index = HNSWLibIndex(config, max_elements=50)
        vectors = np.random.random((30, 64)).astype(np.float32)
        
        # First batch
        index.add_vectors(vectors[:15])
        assert index.stats.total_vectors == 15
        
        # Second batch
        index.add_vectors(vectors[15:])
        assert index.stats.total_vectors == 30
        
        # Search
        results = index.search(vectors[20], k=1)
        assert results[0][0] == 20
    
    def test_persistence(self, config, sample_vectors):
        """Test save and load"""
        index = HNSWLibIndex(config)
        index.build(sample_vectors)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "test_index")
            
            # Save
            index.save(path)
            
            # Load
            loaded = HNSWLibIndex.load(path)
            
            assert loaded.stats.total_vectors == 100
            assert loaded.is_built is True
            
            # Verify search works
            results = loaded.search(sample_vectors[0], k=3)
            assert results[0][0] == 0
    
    def test_set_ef(self, config, sample_vectors):
        """Test setting ef parameter"""
        index = HNSWLibIndex(config)
        index.build(sample_vectors)
        
        # Set higher ef
        index.set_ef(100)
        assert index.ef_search == 100
        
        # Search should still work
        results = index.search(sample_vectors[0], k=5)
        assert len(results) == 5


class TestHNSWLibVsHNSW:
    """Compare HNSWLib with pure Python HNSW"""
    
    def test_similar_results(self):
        """Both implementations should return similar results"""
        from vectordb.index.hnsw_index import HNSWIndex
        
        np.random.seed(42)
        vectors = np.random.random((200, 64)).astype(np.float32)
        query = vectors[0]
        
        config = HNSWIndexConfig(M=16, ef_construction=100, ef_search=50)
        
        # Pure Python HNSW
        hnsw = HNSWIndex(config)
        hnsw.build(vectors)
        hnsw_results = hnsw.search(query, k=10)
        
        # HNSWLib
        hnswlib_index = HNSWLibIndex(config)
        hnswlib_index.build(vectors)
        lib_results = hnswlib_index.search(query, k=10)
        
        # Both should find the exact match
        assert hnsw_results[0][0] == 0
        assert lib_results[0][0] == 0
        
        # Should have significant overlap
        hnsw_ids = set(r[0] for r in hnsw_results)
        lib_ids = set(r[0] for r in lib_results)
        overlap = len(hnsw_ids & lib_ids)
        assert overlap >= 7, f"Overlap: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
