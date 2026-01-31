"""
HNSWLib Index - High-performance HNSW wrapper using hnswlib

This provides a production-grade HNSW implementation with:
- C++ backend for fast build and search
- SIMD acceleration
- Memory-efficient storage
"""
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    import hnswlib  # type: ignore[import-not-found]
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

from vectordb.core.config import HNSWIndexConfig


@dataclass
class HNSWLibIndexStats:
    """Statistics for the HNSWLib index"""
    total_vectors: int = 0
    max_elements: int = 0
    ef_construction: int = 0
    M: int = 0


class HNSWLibIndex:
    """
    High-performance HNSW Index using hnswlib library
    
    This is a wrapper around the hnswlib C++ implementation,
    providing the same interface as HNSWIndex but with much better performance.
    
    Requires: pip install hnswlib
    """
    
    def __init__(self, config: HNSWIndexConfig, max_elements: int = 100000):
        """
        Initialize HNSWLib index
        
        Args:
            config: HNSW configuration
            max_elements: Maximum number of elements (pre-allocated)
        """
        if not HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is not installed. Install with: pip install hnswlib"
            )
        
        self.config = config
        self.M = config.M
        self.ef_construction = config.ef_construction
        self.ef_search = config.ef_search
        self.max_elements = max_elements
        
        # Map metric to hnswlib space
        self.space = "l2" if config.metric == "l2" else "ip"
        
        self.index: Optional[hnswlib.Index] = None
        self.dimension: Optional[int] = None
        self.vectors: Optional[np.ndarray] = None  # Keep copy for compatibility
        self.n_elements: int = 0
        self.is_built = False
        
        self.stats = HNSWLibIndexStats()
    
    def build(self, vectors: np.ndarray) -> None:
        """
        Build index from scratch
        
        Args:
            vectors: Shape (n_vectors, dimension)
        """
        n_vectors, dimension = vectors.shape
        self.dimension = dimension
        self.vectors = vectors.copy()
        
        # Create index
        self.index = hnswlib.Index(space=self.space, dim=dimension)
        
        # Adjust max_elements if needed
        actual_max = max(self.max_elements, n_vectors)
        
        # Initialize
        self.index.init_index(
            max_elements=actual_max,
            ef_construction=self.ef_construction,
            M=self.M
        )
        
        # Add all vectors (hnswlib uses sequential IDs by default)
        self.index.add_items(vectors, np.arange(n_vectors))
        
        # Set search ef
        self.index.set_ef(self.ef_search)
        
        self.n_elements = n_vectors
        self.is_built = True
        
        # Update stats
        self.stats.total_vectors = n_vectors
        self.stats.max_elements = actual_max
        self.stats.ef_construction = self.ef_construction
        self.stats.M = self.M
    
    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors incrementally
        
        Args:
            vectors: Shape (n_new_vectors, dimension)
        """
        n_new, dim = vectors.shape
        
        if self.index is None:
            # First batch - build from scratch
            self.build(vectors)
            return
        
        # Check dimension
        if dim != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {dim}")
        
        # Check capacity
        if self.n_elements + n_new > self.max_elements:
            # Resize (hnswlib supports resize)
            new_max = max(self.max_elements * 2, self.n_elements + n_new)
            self.index.resize_index(new_max)
            self.max_elements = new_max
        
        # Add new vectors
        start_id = self.n_elements
        ids = np.arange(start_id, start_id + n_new)
        self.index.add_items(vectors, ids)
        
        # Update stored vectors
        if self.vectors is not None:
            self.vectors = np.vstack([self.vectors, vectors])
        else:
            self.vectors = vectors.copy()
        
        self.n_elements += n_new
        self.stats.total_vectors = self.n_elements
        self.is_built = True
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors
        
        Args:
            query: Query vector, shape (dimension,) or (1, dimension)
            k: Number of neighbors to return
            
        Returns:
            List of (vector_idx, distance) tuples
        """
        if self.index is None:
            return []
        
        # Reshape if needed
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        k = min(k, self.n_elements)
        labels, distances = self.index.knn_query(query, k=k)
        
        # Return as list of tuples
        return [(int(labels[0][i]), float(distances[0][i])) for i in range(k)]
    
    def search_batch(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Batch search for multiple queries
        
        Args:
            queries: Shape (n_queries, dimension)
            k: Number of neighbors per query
            
        Returns:
            List of results for each query
        """
        if self.index is None:
            return []
        
        k = min(k, self.n_elements)
        labels, distances = self.index.knn_query(queries, k=k)
        
        results = []
        for i in range(len(queries)):
            results.append([
                (int(labels[i][j]), float(distances[i][j])) 
                for j in range(k)
            ])
        return results
    
    def set_ef(self, ef: int) -> None:
        """Set search ef parameter (higher = better recall, slower)"""
        self.ef_search = ef
        if self.index is not None:
            self.index.set_ef(ef)
    
    def get_stats(self) -> HNSWLibIndexStats:
        """Get index statistics"""
        return self.stats
    
    def save(self, path: str) -> None:
        """
        Save index to disk
        
        Args:
            path: Path to save (without extension)
        """
        if self.index is None:
            raise RuntimeError("Index not built")
        
        # Save hnswlib index
        index_path = path if path.endswith('.bin') else f"{path}.bin"
        self.index.save_index(index_path)
        
        # Save metadata
        meta_path = path.replace('.bin', '') + '_meta.pkl'
        metadata = {
            'config': self.config,
            'dimension': self.dimension,
            'n_elements': self.n_elements,
            'max_elements': self.max_elements,
            'space': self.space,
            'stats': self.stats,
            'vectors': self.vectors,  # Keep for compatibility
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> "HNSWLibIndex":
        """
        Load index from disk
        
        Args:
            path: Path to load from (without extension)
        """
        if not HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib is not installed")
        
        # Load metadata
        meta_path = path.replace('.bin', '') + '_meta.pkl'
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            config=metadata['config'],
            max_elements=metadata['max_elements']
        )
        
        # Restore state
        instance.dimension = metadata['dimension']
        instance.n_elements = metadata['n_elements']
        instance.space = metadata['space']
        instance.stats = metadata['stats']
        instance.vectors = metadata.get('vectors')
        
        # Load hnswlib index
        index_path = path if path.endswith('.bin') else f"{path}.bin"
        instance.index = hnswlib.Index(space=instance.space, dim=instance.dimension)
        instance.index.load_index(index_path, max_elements=instance.max_elements)
        instance.index.set_ef(instance.ef_search)
        
        instance.is_built = True
        
        return instance
