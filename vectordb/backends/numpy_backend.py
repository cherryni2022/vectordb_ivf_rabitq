"""
NumPy backend for VectorDB computations

This is the default backend that works on all systems.
"""
import numpy as np
from typing import Tuple

from .base_backend import BaseBackend, BackendManager


class NumpyBackend(BaseBackend):
    """
    NumPy-based compute backend (CPU)
    
    This is the default fallback backend.
    """
    
    @property
    def name(self) -> str:
        return "numpy"
    
    @property
    def is_gpu(self) -> bool:
        return False
    
    def compute_l2_distances(self, queries: np.ndarray,
                             points: np.ndarray) -> np.ndarray:
        """
        Compute squared L2 distances using efficient formula:
        ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q*p
        """
        queries = np.asarray(queries, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        
        q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
        p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
        dot = np.dot(queries, points.T)
        
        distances = q_norm + p_norm - 2 * dot
        return np.maximum(distances, 0)  # Clip negative values from numerical errors
    
    def compute_ip_distances(self, queries: np.ndarray,
                             points: np.ndarray) -> np.ndarray:
        """Compute negative inner product for ranking"""
        queries = np.asarray(queries, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        return -np.dot(queries, points.T)
    
    def kmeans_iteration(self, vectors: np.ndarray,
                         centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single K-Means iteration"""
        vectors = np.asarray(vectors, dtype=np.float32)
        centroids = np.asarray(centroids, dtype=np.float32)
        
        n_clusters = len(centroids)
        dimension = vectors.shape[1]
        
        # Assignment step
        distances = self.compute_l2_distances(vectors, centroids)
        assignments = np.argmin(distances, axis=1).astype(np.int32)
        
        # Update step - vectorized
        counts = np.bincount(assignments, minlength=n_clusters).astype(np.float32)
        counts = np.maximum(counts, 1)
        
        new_centroids = np.zeros((n_clusters, dimension), dtype=np.float32)
        np.add.at(new_centroids, assignments, vectors)
        new_centroids /= counts[:, np.newaxis]
        
        return new_centroids, assignments
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication"""
        return np.dot(a, b)
    
    def argmin(self, array: np.ndarray, axis: int) -> np.ndarray:
        """Argmin along axis"""
        return np.argmin(array, axis=axis)


# Register numpy backend as default
_numpy_backend = NumpyBackend()
BackendManager.register(_numpy_backend)
