"""
Parallel and cached search utilities for VectorDB

Provides:
- Multi-threaded batch search
- Search result caching
- Adaptive nprobe selection
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Callable, Dict, Any
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
import time


@dataclass
class SearchStats:
    """Statistics for search operations"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


class SearchCache:
    """
    LRU cache for search results
    
    Caches query results to avoid redundant computation for repeated queries.
    Uses MD5 hash of query vector for efficient key generation.
    
    Args:
        max_size: Maximum number of cached results
        ttl_seconds: Time-to-live for cache entries (0 = no expiry)
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats = SearchStats()
    
    def _hash_query(self, query: np.ndarray, k: int, nprobe: int) -> str:
        """Generate hash key for a query"""
        query_bytes = query.astype(np.float32).tobytes()
        params = f"k={k},nprobe={nprobe}".encode()
        return hashlib.md5(query_bytes + params).hexdigest()
    
    def get(self, query: np.ndarray, k: int, nprobe: int) -> Optional[List]:
        """
        Get cached result for a query
        
        Returns None if not cached or expired.
        """
        key = self._hash_query(query, k, nprobe)
        
        with self._lock:
            self.stats.total_queries += 1
            
            if key not in self._cache:
                self.stats.cache_misses += 1
                return None
            
            # Check TTL
            if self.ttl_seconds > 0:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    self.stats.cache_misses += 1
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            
            return self._cache[key]
    
    def put(self, query: np.ndarray, k: int, nprobe: int, result: List):
        """Cache a search result"""
        key = self._hash_query(query, k, nprobe)
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = result
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached results"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> SearchStats:
        """Get cache statistics"""
        return self.stats


class ParallelSearcher:
    """
    Multi-threaded batch search executor
    
    Parallelizes search across multiple queries for higher throughput.
    
    Args:
        search_fn: Search function with signature (query, k) -> List[Tuple[int, float]]
        n_threads: Number of worker threads
        cache: Optional search cache
    """
    
    def __init__(self, search_fn: Callable, n_threads: int = 4,
                 cache: Optional[SearchCache] = None):
        self.search_fn = search_fn
        self.n_threads = n_threads
        self.cache = cache
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def _ensure_executor(self):
        """Lazily create executor"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.n_threads)
    
    def search_batch(self, queries: np.ndarray, k: int,
                     nprobe: Optional[int] = None) -> List[List[Tuple[int, float]]]:
        """
        Search for multiple queries in parallel
        
        Args:
            queries: Shape (n_queries, dimension)
            k: Number of neighbors per query
            nprobe: Optional nprobe value for caching
            
        Returns:
            List of result lists, one per query
        """
        n_queries = len(queries)
        results = [None] * n_queries
        
        self._ensure_executor()
        
        # Submit all queries
        futures = {}
        for i, query in enumerate(queries):
            # Check cache first
            if self.cache is not None and nprobe is not None:
                cached = self.cache.get(query, k, nprobe)
                if cached is not None:
                    results[i] = cached
                    continue
            
            future = self._executor.submit(self.search_fn, query, k)
            futures[future] = i
        
        # Collect results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                
                # Cache result
                if self.cache is not None and nprobe is not None:
                    self.cache.put(queries[idx], k, nprobe, result)
            except Exception as e:
                results[idx] = []  # Empty result on error
        
        return results
    
    def search_batch_chunked(self, queries: np.ndarray, k: int,
                              chunk_size: int = 100,
                              nprobe: Optional[int] = None) -> List[List[Tuple[int, float]]]:
        """
        Search with chunked parallelism
        
        Processes queries in chunks to reduce thread overhead for many queries.
        
        Args:
            queries: Shape (n_queries, dimension)
            k: Number of neighbors per query
            chunk_size: Queries per chunk
            nprobe: Optional nprobe for caching
            
        Returns:
            List of result lists
        """
        n_queries = len(queries)
        results = [None] * n_queries
        
        self._ensure_executor()
        
        def process_chunk(start: int, end: int):
            chunk_results = []
            for i in range(start, end):
                query = queries[i]
                
                # Check cache
                if self.cache is not None and nprobe is not None:
                    cached = self.cache.get(query, k, nprobe)
                    if cached is not None:
                        chunk_results.append((i, cached))
                        continue
                
                result = self.search_fn(query, k)
                chunk_results.append((i, result))
                
                # Cache result
                if self.cache is not None and nprobe is not None:
                    self.cache.put(query, k, nprobe, result)
            
            return chunk_results
        
        # Submit chunks
        futures = []
        for start in range(0, n_queries, chunk_size):
            end = min(start + chunk_size, n_queries)
            future = self._executor.submit(process_chunk, start, end)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            chunk_results = future.result()
            for idx, result in chunk_results:
                results[idx] = result
        
        return results
    
    def shutdown(self):
        """Shutdown the thread pool"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class AdaptiveNProbe:
    """
    Adaptive nprobe selection based on query characteristics
    
    Automatically adjusts nprobe based on the query's distance distribution
    to balance search quality and performance.
    
    Args:
        base_nprobe: Default nprobe value
        min_nprobe: Minimum nprobe
        max_nprobe: Maximum nprobe
        target_recall: Target recall for adaptation (0.0 to 1.0)
    """
    
    def __init__(self, base_nprobe: int = 10, min_nprobe: int = 1,
                 max_nprobe: int = 100, target_recall: float = 0.95):
        self.base_nprobe = base_nprobe
        self.min_nprobe = min_nprobe
        self.max_nprobe = max_nprobe
        self.target_recall = target_recall
        
        # Learning state
        self._recall_history: List[float] = []
        self._nprobe_history: List[int] = []
    
    def estimate_nprobe(self, centroid_distances: np.ndarray) -> int:
        """
        Estimate optimal nprobe based on centroid distance distribution
        
        If distances to top centroids are similar, we need higher nprobe
        to ensure we don't miss relevant clusters.
        
        Args:
            centroid_distances: Distances to all centroids
            
        Returns:
            Recommended nprobe value
        """
        sorted_dists = np.sort(centroid_distances)
        
        if len(sorted_dists) < 2:
            return self.base_nprobe
        
        # Compute ratio between consecutive distances
        # Higher ratio = more distinct separation = can use lower nprobe
        top_k = min(self.base_nprobe * 2, len(sorted_dists))
        dist_ratios = sorted_dists[1:top_k] / (sorted_dists[:top_k-1] + 1e-8)
        
        # Find where distances start to diverge significantly
        diverge_idx = np.argmax(dist_ratios > 1.5)
        if dist_ratios[diverge_idx] <= 1.5:
            diverge_idx = len(dist_ratios)
        
        # Recommended nprobe is at least until divergence point
        recommended = max(diverge_idx + 1, self.min_nprobe)
        recommended = min(recommended, self.max_nprobe)
        
        return recommended
    
    def update(self, nprobe_used: int, recall: float):
        """
        Update learning state with observed recall
        
        Args:
            nprobe_used: nprobe value that was used
            recall: Observed recall for that search
        """
        self._nprobe_history.append(nprobe_used)
        self._recall_history.append(recall)
        
        # Keep only recent history
        max_history = 1000
        if len(self._nprobe_history) > max_history:
            self._nprobe_history = self._nprobe_history[-max_history:]
            self._recall_history = self._recall_history[-max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        if not self._recall_history:
            return {'n_samples': 0}
        
        return {
            'n_samples': len(self._recall_history),
            'avg_recall': np.mean(self._recall_history),
            'avg_nprobe': np.mean(self._nprobe_history),
            'min_recall': np.min(self._recall_history),
            'max_recall': np.max(self._recall_history),
        }
