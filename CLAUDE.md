# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a lightweight, single-machine vector database MVP implementing IVF (Inverted File) indexing with **True RaBitQ** (Randomized Binary Quantization) for efficient and accelerated similarity search.

**Version**: 0.3.0 (with performance optimizations)

### Architecture

The codebase is organized into several main components:

1. **IVF Index** (`vectordb/index/ivf_index.py`) - Partitions vectors into clusters using k-means clustering. During search, only a subset of closest clusters (nprobe) are examined, providing fast approximate nearest neighbor search.

2. **True RaBitQ** (`vectordb/quantization/true_rabitq.py`) - True RaBitQ implementation based on SIGMOD'24 paper:
   - Random orthogonal transformation for even information distribution
   - 1-bit sign quantization per dimension (32x compression)
   - Hamming distance approximation for L2/IP distances
   - Two-stage search: coarse RaBitQ filtering + exact reranking

3. **Legacy PQ** (`vectordb/quantization/rabitq.py`) - Product Quantization implementation for backward compatibility.

4. **VectorDB** (`vectordb/core/vector_db.py`) - Main entry point that orchestrates IVF indexing, True RaBitQ quantization, and storage.

5. **Optimization Modules** (NEW in v0.3.0):
   - `vectordb/utils/numba_utils.py` - Numba JIT-accelerated utilities
   - `vectordb/backends/` - Compute backend abstraction (NumPy, PyTorch GPU)
   - `vectordb/storage/mmap_store.py` - Memory-mapped vector storage
   - `vectordb/search/parallel_search.py` - Parallel search and caching

**Key Data Flow:**
- `add_vectors()` → Storage → `build()` → IVF k-means + True RaBitQ encoding
- `search(accelerated=True)` → Find nprobe closest centroids → RaBitQ approximate distances → top-k candidates → exact rerank → return results

## Environment Setup

```bash
# Install Python 3.12 (if not already installed)
uv python install 3.12

# Create virtual environment
uv venv .venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Sync dependencies (including dev dependencies)
uv sync --dev
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run optimization tests
python tests/test_phase1_optimizations.py
python tests/test_all_optimizations.py

# Run True RaBitQ tests
pytest tests/test_true_rabitq.py -v

# Run other test files
pytest tests/test_ivf_index.py -v
pytest tests/test_rabitq.py -v
pytest tests/test_vector_db.py -v

# Run performance benchmark
python benchmarks/rabitq_benchmark.py

# Run example usage script
python example_usage.py

# Install dependencies
pip install -r requirements.txt
```

## Configuration Classes

- `IVFIndexConfig` - Controls clustering: `nlist` (number of clusters), `nprobe` (clusters to search), `metric` ("l2" or "ip")
- `RaBitQConfig` - Controls quantization: `nsubq` (sub-quantizers), `nbits` (bits per sub-quantizer), `use_rabitq` (enable quantization)
- `VectorDBConfig` - Combines both configs with `dimension` and `use_quantization` flags

### New IVFIndex Parameters (v0.3.0)

```python
IVFIndex(
    nlist=100,                # Number of clusters
    nprobe=10,                # Clusters to search
    metric="l2",              # Distance metric
    use_minibatch=False,      # Use mini-batch K-Means
    minibatch_size=1024,      # Mini-batch size
    chunk_size=10000,         # Chunk size for distance computation
)
```

## Optimization Features (v0.3.0)

### Phase 1: NumPy Optimizations

1. **Vectorized K-Means Centroid Update**: Uses `np.add.at` and `np.bincount` for efficient centroid updates
2. **Static Popcount Table**: Class-level lookup table for Hamming distance computation
3. **Chunked Distance Computation**: Memory-efficient distance calculation for large datasets
4. **Mini-Batch K-Means**: For datasets > 100K vectors, automatically uses mini-batch K-Means

### Phase 2: Numba JIT Acceleration

Located in `vectordb/utils/numba_utils.py`:

```python
from vectordb.utils import fast_hamming_distance, fast_l2_distances, is_numba_available

# Check if Numba is available
if is_numba_available():
    # Uses JIT-compiled version
    distances = fast_l2_distances(queries, points)
else:
    # Falls back to NumPy
    distances = fast_l2_distances(queries, points)
```

### Phase 3: GPU Acceleration

Backend abstraction in `vectordb/backends/`:

```python
from vectordb.backends import get_best_backend, list_backends

# List available backends
print(list_backends())  # ['numpy', 'pytorch_cuda', 'pytorch_cpu']

# Get best available backend
backend = get_best_backend(prefer_gpu=True)
distances = backend.compute_l2_distances(queries, points)

# Run K-Means on GPU
if backend.is_gpu:
    centroids = backend.run_kmeans(vectors, n_clusters=100)
```

### Phase 4: System-Level Optimizations

#### Memory-Mapped Storage

```python
from vectordb.storage import MemoryMappedVectorStore, ChunkedVectorLoader

# Create memory-mapped storage for large datasets
store = MemoryMappedVectorStore("vectors.mmap", dimension=128)
mmap = store.create(n_vectors=1000000)
mmap[:] = your_vectors
store.flush()

# Load and access efficiently
store = MemoryMappedVectorStore("vectors.mmap", dimension=128)
vectors = store.load()
batch = store.get_batch([0, 1, 2, 100, 200])
```

#### Parallel Search with Caching

```python
from vectordb.search import ParallelSearcher, SearchCache

# Create cached parallel searcher
cache = SearchCache(max_size=10000, ttl_seconds=3600)
searcher = ParallelSearcher(
    search_fn=lambda q, k: index.search(q, k),
    n_threads=4,
    cache=cache
)

# Batch search with automatic parallelization
results = searcher.search_batch(queries, k=10)

# Check cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
```

#### Adaptive nprobe

```python
from vectordb.search import AdaptiveNProbe

adaptive = AdaptiveNProbe(base_nprobe=10, min_nprobe=1, max_nprobe=50)
recommended = adaptive.estimate_nprobe(centroid_distances)
```

## True RaBitQ - Key Concepts

### Compression
- 32x compression: Each dimension compressed from 32-bit float to 1-bit sign
- Storage: `ceil(dimension/8)` bytes for binary codes + 4 bytes for norm
- For 128-dim vectors: 512 bytes → ~20 bytes (including norm)

### Distance Approximation
```
cos(θ) ≈ 1 - 2 * hamming_distance / dimension
||x - y||² ≈ ||x||² + ||y||² - 2 * ||x|| * ||y|| * cos(θ)
```

### Two-Stage Search
1. **Coarse stage**: Use Hamming distance on binary codes for fast candidate selection
2. **Rerank stage**: Compute exact L2 distances for top-k * rerank_factor candidates
3. **Return**: Final top-k results with exact distances

## Important Implementation Details

### Accelerated Search (`search(accelerated=True)`)
- Default: `accelerated=True` uses IVF + RaBitQ
- Falls back to standard IVF if True RaBitQ not available
- `rerank_factor`: Controls candidate pool size (default: 10)

### Distance Computation
- L2 distance: Uses efficient matrix computation `||q||^2 + ||p||^2 - 2*q*p`
- Inner product: Returns `-dot(q, p)` (negative because search minimizes distance)
- All L2 distances are clipped to non-negative values to handle floating point errors

### K-means Clustering
- Uses k-means++ initialization for better cluster quality
- Handles edge cases: duplicate vectors (all distances zero), empty clusters (re-initialize from largest cluster)
- Mini-batch K-Means available for large-scale datasets (> 100K vectors)

### Persistence
- IVF index: `index.npz` (pickle)
- True RaBitQ: `true_rabitq.pkl` (pickle with rotation matrix, binary codes, norms)
- VectorStorage: `vectors.npy` + `metadata.pkl`
- InMemoryVectorStorage skips all filesystem operations for faster testing

### Storage Mappings
- `id_to_index`: Maps vector IDs to array indices
- `index_to_id`: Maps array indices back to vector IDs
- `vector_to_cluster`: Maps each vector index to its assigned cluster ID

## Performance Tuning

### nprobe
- Higher nprobe = better recall, slower search
- Recommended: nprobe = nlist / 10 for 90%+ recall

### rerank_factor
- Higher rerank_factor = better quality, more exact distance computations
- Recommended: 10 for good quality, 20+ for high-precision applications

### Mini-batch K-Means
- Automatically enabled for datasets > 100K vectors
- Use `minibatch_size=1024` for good balance of speed and quality

### GPU Acceleration
- Significant speedup for K-Means (10-100x on large datasets)
- Requires PyTorch with CUDA installed

### Compression Trade-offs
- True RaBitQ: 32x compression with ~5-10% recall loss
- Best for: Large datasets where memory is a concern

