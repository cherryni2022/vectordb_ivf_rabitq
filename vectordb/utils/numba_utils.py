"""
Numba JIT-accelerated utility functions for VectorDB

This module provides high-performance implementations using Numba JIT compilation.
Falls back to NumPy implementations if Numba is not available.
"""
import numpy as np
from typing import Optional

# Check if Numba is available
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def prange(*args, **kwargs):
        return range(*args)


# Pre-computed popcount lookup table
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


# ============================================================================
# Hamming Distance Functions
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _fast_hamming_distance_numba(query_code: np.ndarray, 
                                      database_codes: np.ndarray,
                                      popcount_table: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated Hamming distance computation
        
        Args:
            query_code: Shape (n_bytes,) uint8
            database_codes: Shape (n_vectors, n_bytes) uint8
            popcount_table: Shape (256,) int32, pre-computed popcount table
            
        Returns:
            hamming_distances: Shape (n_vectors,) int32
        """
        n_vectors = database_codes.shape[0]
        n_bytes = query_code.shape[0]
        result = np.zeros(n_vectors, dtype=np.int32)
        
        for i in prange(n_vectors):
            count = 0
            for j in range(n_bytes):
                xor_val = database_codes[i, j] ^ query_code[j]
                count += popcount_table[xor_val]
            result[i] = count
        
        return result


def _fast_hamming_distance_numpy(query_code: np.ndarray,
                                  database_codes: np.ndarray,
                                  popcount_table: np.ndarray) -> np.ndarray:
    """
    NumPy fallback for Hamming distance computation
    """
    xor_result = np.bitwise_xor(database_codes, query_code)
    return np.sum(popcount_table[xor_result], axis=1)


def fast_hamming_distance(query_code: np.ndarray,
                          database_codes: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance between query and database codes
    
    Uses Numba JIT if available, otherwise falls back to NumPy.
    
    Args:
        query_code: Shape (n_bytes,) uint8
        database_codes: Shape (n_vectors, n_bytes) uint8
        
    Returns:
        hamming_distances: Shape (n_vectors,) int32
    """
    if NUMBA_AVAILABLE:
        return _fast_hamming_distance_numba(query_code, database_codes, _POPCOUNT_TABLE)
    else:
        return _fast_hamming_distance_numpy(query_code, database_codes, _POPCOUNT_TABLE)


# ============================================================================
# Distance Computation Functions
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True, fastmath=True)
    def _fast_l2_distances_numba(queries: np.ndarray, 
                                  points: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated L2 squared distance computation
        
        Args:
            queries: Shape (n_queries, dimension) float32
            points: Shape (n_points, dimension) float32
            
        Returns:
            distances: Shape (n_queries, n_points) float32
        """
        n_queries = queries.shape[0]
        n_points = points.shape[0]
        dimension = queries.shape[1]
        
        result = np.zeros((n_queries, n_points), dtype=np.float32)
        
        for i in prange(n_queries):
            for j in range(n_points):
                dist = 0.0
                for k in range(dimension):
                    diff = queries[i, k] - points[j, k]
                    dist += diff * diff
                result[i, j] = dist
        
        return result


def _fast_l2_distances_numpy(queries: np.ndarray,
                              points: np.ndarray) -> np.ndarray:
    """
    NumPy fallback for L2 distance computation
    Uses efficient formula: ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q*p
    """
    q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
    p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
    dot = np.dot(queries, points.T)
    return q_norm + p_norm - 2 * dot


def fast_l2_distances(queries: np.ndarray,
                      points: np.ndarray,
                      use_numba: bool = True) -> np.ndarray:
    """
    Compute squared L2 distances between queries and points
    
    Note: For large matrices, NumPy BLAS may be faster than Numba.
    Use use_numba=False to force NumPy implementation.
    
    Args:
        queries: Shape (n_queries, dimension)
        points: Shape (n_points, dimension)
        use_numba: Whether to use Numba (if available)
        
    Returns:
        distances: Shape (n_queries, n_points)
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    points = np.ascontiguousarray(points, dtype=np.float32)
    
    # For large matrices, NumPy BLAS is often faster
    # Numba is better for small to medium matrices
    n_ops = queries.shape[0] * points.shape[0] * queries.shape[1]
    use_numba_impl = use_numba and NUMBA_AVAILABLE and n_ops < 100000000
    
    if use_numba_impl:
        return _fast_l2_distances_numba(queries, points)
    else:
        return _fast_l2_distances_numpy(queries, points)


# ============================================================================
# K-Means Functions
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True, fastmath=True)
    def fast_kmeans_assignment(vectors: np.ndarray,
                               centroids: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated K-Means assignment step
        
        Assigns each vector to its nearest centroid.
        
        Args:
            vectors: Shape (n_vectors, dimension) float32
            centroids: Shape (n_clusters, dimension) float32
            
        Returns:
            assignments: Shape (n_vectors,) int32
        """
        n_vectors = vectors.shape[0]
        n_clusters = centroids.shape[0]
        dimension = vectors.shape[1]
        
        assignments = np.zeros(n_vectors, dtype=np.int32)
        
        for i in prange(n_vectors):
            min_dist = np.inf
            min_idx = 0
            for j in range(n_clusters):
                dist = 0.0
                for k in range(dimension):
                    diff = vectors[i, k] - centroids[j, k]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            assignments[i] = min_idx
        
        return assignments
    
    
    @njit(cache=True)
    def fast_centroid_update(vectors: np.ndarray,
                             assignments: np.ndarray,
                             n_clusters: int) -> np.ndarray:
        """
        Numba-accelerated centroid update step
        
        Args:
            vectors: Shape (n_vectors, dimension) float32
            assignments: Shape (n_vectors,) int32
            n_clusters: Number of clusters
            
        Returns:
            new_centroids: Shape (n_clusters, dimension) float32
        """
        n_vectors = vectors.shape[0]
        dimension = vectors.shape[1]
        
        new_centroids = np.zeros((n_clusters, dimension), dtype=np.float32)
        counts = np.zeros(n_clusters, dtype=np.int32)
        
        # Sum vectors per cluster
        for i in range(n_vectors):
            cluster = assignments[i]
            counts[cluster] += 1
            for k in range(dimension):
                new_centroids[cluster, k] += vectors[i, k]
        
        # Compute mean
        for j in range(n_clusters):
            if counts[j] > 0:
                for k in range(dimension):
                    new_centroids[j, k] /= counts[j]
        
        return new_centroids


# Fallback implementations
def _fast_kmeans_assignment_numpy(vectors: np.ndarray,
                                   centroids: np.ndarray) -> np.ndarray:
    """NumPy fallback for K-Means assignment"""
    distances = _fast_l2_distances_numpy(vectors, centroids)
    return np.argmin(distances, axis=1).astype(np.int32)


def _fast_centroid_update_numpy(vectors: np.ndarray,
                                 assignments: np.ndarray,
                                 n_clusters: int) -> np.ndarray:
    """NumPy fallback for centroid update"""
    dimension = vectors.shape[1]
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.float32)
    counts = np.maximum(counts, 1)
    
    new_centroids = np.zeros((n_clusters, dimension), dtype=np.float32)
    np.add.at(new_centroids, assignments, vectors)
    new_centroids /= counts[:, np.newaxis]
    
    return new_centroids


# ============================================================================
# Public API
# ============================================================================

def get_backend_info() -> dict:
    """Get information about available backends"""
    return {
        'numba_available': NUMBA_AVAILABLE,
        'numba_version': numba.__version__ if NUMBA_AVAILABLE else None,
    }


def is_numba_available() -> bool:
    """Check if Numba is available"""
    return NUMBA_AVAILABLE
