# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a lightweight, single-machine vector database MVP implementing IVF (Inverted File) indexing with **True RaBitQ** (Randomized Binary Quantization) for efficient and accelerated similarity search.

### Architecture

The codebase is organized into four main components:

1. **IVF Index** (`vectordb/index/ivf_index.py`) - Partitions vectors into clusters using k-means clustering. During search, only a subset of closest clusters (nprobe) are examined, providing fast approximate nearest neighbor search.

2. **True RaBitQ** (`vectordb/quantization/true_rabitq.py`) - **NEW!** True RaBitQ implementation based on SIGMOD'24 paper:
   - Random orthogonal transformation for even information distribution
   - 1-bit sign quantization per dimension (32x compression)
   - Hamming distance approximation for L2/IP distances
   - Two-stage search: coarse RaBitQ filtering + exact reranking

3. **Legacy PQ** (`vectordb/quantization/rabitq.py`) - Product Quantization implementation for backward compatibility.

4. **VectorDB** (`vectordb/core/vector_db.py`) - Main entry point that orchestrates IVF indexing, True RaBitQ quantization, and storage.

**Key Data Flow:**
- `add_vectors()` → Storage → `build()` → IVF k-means + True RaBitQ encoding
- `search(accelerated=True)` → Find nprobe closest centroids → RaBitQ approximate distances → top-k candidates → exact rerank → return results

## Common Commands

```bash
# Run all tests
pytest tests/ -v

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
- Handles edge cases: duplicate vectors (all distances zero), empty clusters (re-initialize randomly)

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

### Compression Trade-offs
- True RaBitQ: 32x compression with ~5-10% recall loss
- Best for: Large datasets where memory is a concern
