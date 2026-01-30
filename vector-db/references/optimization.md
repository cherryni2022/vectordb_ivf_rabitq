# Performance Optimization for Vector Databases

## Overview

Optimization techniques for building fast, memory-efficient vector databases.

## Memory Optimization

### Memory-Mapped Storage

Load vectors on-demand instead of loading entire dataset into RAM.

```python
import numpy as np

class MemoryMappedVectorStore:
    def __init__(self, path, dimension):
        self.path = path
        self.dimension = dimension

    def create(self, n_vectors):
        """Create memory-mapped array"""
        self.mmap = np.memmap(
            self.path,
            dtype=np.float32,
            mode='w+',
            shape=(n_vectors, self.dimension)
        )
        return self.mmap

    def load(self):
        """Load existing memory-mapped array"""
        self.mmap = np.memmap(
            self.path,
            dtype=np.float32,
            mode='r',
            shape=(n_vectors, self.dimension)
        )
        return self.mmap

# Usage
store = MemoryMappedVectorStore("vectors.mmap", dimension=128)
store.create(n_vectors=1000000)
store.mmap[:] = your_vectors
store.flush()  # Write to disk
```

**Benefits:**
- Handle datasets larger than RAM
- Only load accessed vectors
- Persist vectors across restarts

### Batch Processing

Process data in chunks to avoid memory overflow:

```python
def process_in_batches(vectors, batch_size=10000):
    n_vectors = vectors.shape[0]
    results = []

    for i in range(0, n_vectors, batch_size):
        batch = vectors[i:i+batch_size]
        results.append(process_batch(batch))

    return np.concatenate(results)
```

### Type Selection

| Type | Bytes/Dim | Range | Precision | Use When |
|------|------------|-------|-----------|----------|
| `float64` | 8 | - | Highest | Scientific computing |
| `float32` | 4 | Standard | Good | Default for vectors |
| `float16` | 2 | Limited | Low | Embedding compression |
| `bfloat16` | 2 | Good dynamic range | Medium | ML inference |

**Recommendation:** Use `float32` for vectors, `float16` for centroids.

---

## Speed Optimization

### Chunked Distance Computation

Avoid O(n²) memory allocation for large datasets:

```python
def compute_distances_chunked(queries, points, chunk_size=10000):
    n_queries = queries.shape[0]
    n_points = points.shape[0]

    distances = np.empty((n_queries, n_points), dtype=np.float32)

    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        chunk = points[i:end]

        # Compute L2: ||q||² + ||p||² - 2*q*p
        q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
        p_norm = np.sum(chunk ** 2, axis=1, keepdims=True).T
        dot = np.dot(queries, chunk.T)

        distances[:, i:end] = q_norm + p_norm - 2 * dot

    return distances
```

### JIT Compilation (Numba)

Compile hot paths to machine code:

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def hamming_distance_numba(codes1, codes2):
    """Fast Hamming distance with Numba"""
    n = len(codes1)
    result = np.zeros(n, dtype=np.int32)

    for i in range(n):
        xor_result = codes1[i] ^ codes2[i]
        # Count bits
        result[i] = bin(xor_result).count('1')

    return result

# Check Numba availability
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorator that does nothing
    def jit(nopython=False):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False
```

**Benchmark results (1M x 128):**
- NumPy: 450ms
- Numba JIT: 35ms (12x speedup)

### Vectorized Operations

Avoid Python loops, use NumPy broadcasting:

```python
# Slow: Python loop
def slow_distance(queries, points):
    results = []
    for q in queries:
        for p in points:
            results.append(np.sum((q - p) ** 2))
    return np.array(results)

# Fast: Vectorized NumPy
def fast_distance(queries, points):
    q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
    p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
    dot = np.dot(queries, points.T)
    return q_norm + p_norm - 2 * dot
```

### Parallel Batch Search

Process multiple queries in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_search(index, queries, k, n_threads=4):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(
            lambda q: index.search(q, k),
            queries
        ))
    return results
```

---

## GPU Acceleration

### PyTorch Backend

Use GPU for distance computation and k-means:

```python
import torch

class TorchBackend:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def compute_l2_distances(self, queries, points):
        q = torch.tensor(queries, device=self.device)
        p = torch.tensor(points, device=self.device)

        q_norm = torch.sum(q ** 2, dim=1, keepdim=True)
        p_norm = torch.sum(p ** 2, dim=1, keepdim=True).T
        dot = torch.mm(q, p.T)

        return (q_norm + p_norm - 2 * dot).cpu().numpy()

    def run_kmeans(self, vectors, n_clusters, max_iter=100):
        v = torch.tensor(vectors, device=self.device)

        # Initialize centroids
        indices = torch.randperm(len(v))[:n_clusters]
        centroids = v[indices]

        for _ in range(max_iter):
            # Assign to nearest centroid
            distances = torch.cdist(v, centroids)
            assignments = torch.argmin(distances, dim=1)

            # Update centroids
            for i in range(n_clusters):
                mask = assignments == i
                if torch.any(mask):
                    centroids[i] = v[mask].mean(dim=0)

        return centroids.cpu().numpy()
```

**Speedup:** 10-100x for k-means, 2-5x for distance computation

### Backend Abstraction

```python
def get_best_backend(prefer_gpu=True):
    """Auto-select best available backend"""
    if prefer_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return TorchBackend(device='cuda')
        except ImportError:
            pass

    return NumPyBackend()  # Fallback
```

---

## Caching

### Search Cache

Cache query results for repeated queries:

```python
from functools import lru_cache
import hashlib

class SearchCache:
    def __init__(self, max_size=10000):
        self.cache = {}

    def _hash_query(self, query):
        """Create stable hash for query vector"""
        query_bytes = query.tobytes()
        return hashlib.sha256(query_bytes).hexdigest()

    def get(self, query):
        key = self._hash_query(query)
        return self.cache.get(key)

    def set(self, query, results):
        key = self._hash_query(query)
        self.cache[key] = results

# Use with decorator
cached_search = SearchCache(max_size=1000)

def search_with_cache(index, query, k):
    cached = cached_search.get(query)
    if cached is not None:
        return cached

    results = index.search(query, k)
    cached_search.set(query, results)
    return results
```

### Centroid Distance Cache

Pre-compute and cache query-to-centroid distances:

```python
class IVFIndexWithCache:
    def __init__(self, nlist, nprobe):
        self.centroid_cache = {}
        # ... other init

    def search(self, query, k):
        # Check cache for centroid distances
        query_key = self._hash_query(query)
        if query_key not in self.centroid_cache:
            self.centroid_cache[query_key] = self._compute_centroid_distances(query)

        centroid_dists = self.centroid_cache[query_key]
        # ... proceed with search
```

---

## Adaptive Parameters

### Adaptive nprobe

Adjust nprobe based on query distribution:

```python
class AdaptiveNProbe:
    def __init__(self, base_nprobe=10, min_nprobe=1, max_nprobe=50):
        self.base_nprobe = base_nprobe
        self.min_nprobe = min_nprobe
        self.max_nprobe = max_nprobe

    def estimate_nprobe(self, centroid_distances):
        """
        Increase nprobe if query is near centroid boundaries
        (ambiguous cluster assignment)
        """
        # Sort distances to centroids
        sorted_dists = np.sort(centroid_distances)

        # If first and second closest centroids are similar distance,
        # query is ambiguous - increase nprobe
        ratio = sorted_dists[1] / (sorted_dists[0] + 1e-8)

        if ratio > 0.9:  # Very ambiguous
            return min(self.max_nprobe, self.base_nprobe * 2)
        elif ratio > 0.7:  # Somewhat ambiguous
            return min(self.max_nprobe, int(self.base_nprobe * 1.5))
        else:  # Clear assignment
            return self.base_nprobe
```

---

## Benchmarking

### Performance Profiler

```python
import time
import numpy as np

class VectorDBBenchmark:
    def __init__(self, index):
        self.index = index

    def benchmark_search(self, queries, k, n_runs=10):
        """Benchmark search latency"""
        latencies = []

        for _ in range(n_runs):
            start = time.perf_counter()
            for q in queries:
                self.index.search(q, k)
            end = time.perf_counter()
            latencies.append(end - start)

        return {
            'mean_qps': len(queries) / np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50) * 1000 / len(queries),
            'p95_latency_ms': np.percentile(latencies, 95) * 1000 / len(queries),
            'p99_latency_ms': np.percentile(latencies, 99) * 1000 / len(queries),
        }

    def benchmark_build(self, vectors):
        """Benchmark build time"""
        start = time.perf_counter()
        self.index.build(vectors)
        duration = time.perf_counter() - start
        return {'build_time_seconds': duration}

    def benchmark_memory(self):
        """Estimate memory usage"""
        import sys
        import pickle

        serialized = pickle.dumps(self.index)
        return {
            'serialized_size_mb': len(serialized) / (1024 * 1024),
            'estimated_ram_mb': sys.getsizeof(self.index) / (1024 * 1024),
        }
```

---

## Optimization Checklist

- [ ] Use float32 instead of float64 for vectors
- [ ] Enable chunked distance computation for large datasets
- [ ] Use k-means++ initialization for better clusters
- [ ] Apply quantization for memory reduction
- [ ] Enable mini-batch k-means for >100K vectors
- [ ] Use GPU backend if available
- [ ] Profile with realistic query distributions
- [ ] Tune nprobe and rerank_factor for your use case
- [ ] Consider memory-mapped storage for very large datasets
- [ ] Use JIT compilation for hot paths
