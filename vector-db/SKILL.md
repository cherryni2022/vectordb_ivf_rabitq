---
name: vector-db
description: Vector database and embedding best practices including IVF indexing, quantization (PQ, RaBitQ, scalar), distance metrics, and performance optimization. Use when working with vector similarity search, embeddings, or building approximate nearest neighbor (ANN) systems. Covers design choices, implementation patterns, performance tuning, and debugging vector database operations.
---

# Vector Database & Embeddings

## Overview

Enable efficient approximate nearest neighbor (ANN) search for high-dimensional vectors using indexing, quantization, and optimization techniques.

## Core Concepts

### Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **L2 (Euclidean)** | `||x - y||²` | General similarity, normalized vectors |
| **Inner Product** | `-<x, y>` | Search by relevance, normalized vectors |
| **Cosine** | `1 - cos(θ)` | Angular similarity |

**Optimized L2 computation:**
```python
# ||q - p||² = ||q||² + ||p||² - 2*q*p
q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
dot = np.dot(queries, points.T)
distances = q_norm + p_norm - 2 * dot
```

### Index Structures

#### IVF (Inverted File)
- Partition vectors into `nlist` clusters via k-means
- At search time, only probe `nprobe` closest clusters
- Trade-off: higher `nprobe` = better recall, slower

**Use IVF when:**
- Dataset > 10K vectors (flat search too slow)
- Need sub-millisecond search
- Accept ~5-15% recall loss

**Configuration:**
```python
nlist = int(4 * sqrt(n_vectors))  # Rule of thumb
nprobe = nlist // 10  # Probes ~10% of clusters
```

#### Flat Index
- Brute-force search over all vectors
- Exact results, O(n) complexity
- Best for small datasets (<10K) or when recall is critical

#### HNSW (Hierarchical Navigable Small World)
- Graph-based index with layered structure
- Near-exact recall at reasonable speed
- Best for applications requiring high precision

### Quantization

**Purpose:** Reduce memory footprint and accelerate distance computations.

#### Product Quantization (PQ)
- Split vectors into `nsubq` sub-vectors
- Each subspace quantized independently to `nbits` centroids
- Compression: `d * 4` bytes → `nsubq * nbits` bytes

#### True RaBitQ (Randomized Binary Quantization)
- 32x compression: each dimension → 1 bit (sign)
- Requires random orthogonal transformation for even info distribution
- Distance approximation via Hamming distance:
  ```
  cos(θ) ≈ 1 - 2 * hamming(sign(x), sign(y)) / d
  ||x-y||² = ||x||² + ||y||² - 2*||x||*||y||*cos(θ)
  ```
- Use for memory-constrained, large-scale scenarios

#### Scalar Quantization (SQ)
- Each dimension quantized to 8-bit (uint8)
- 4x compression with minimal accuracy loss
- Simple and fast

### Two-Stage Search Pattern

For high performance with good recall:
1. **Coarse stage:** Use quantized codes (RaBitQ Hamming distance)
2. **Rerank stage:** Compute exact distances for top candidates

```python
# Example: IVF + RaBitQ + Rerank
indices = ivf.get_cluster_candidates(query, nprobe=10)
approx_dists = rabitq.compute_approximate_distances(query, indices)
rerank_pool = indices[np.argsort(approx_dists)[:k * 10]]
exact_dists = compute_l2_distances(query, vectors[rerank_pool])
results = top_k(exact_dists)[:k]
```

## Performance Tuning

### Parameter Selection

| Parameter | Low Value | High Value | Recommended |
|-----------|-----------|------------|-------------|
| `nlist` | Poor partitioning | Slow building | `4 * sqrt(n)` |
| `nprobe` | Fast, low recall | Slow, high recall | `nlist / 10` |
| `rerank_factor` | Fast, lower quality | Slower, higher quality | 10-20 |
| `minibatch_size` | Noisy clusters | Slow per iteration | 1024-4096 |

### Memory Optimization

1. **Use memory-mapped storage** for datasets >1M vectors
2. **Apply quantization** (32x with RaBitQ, 4x with SQ)
3. **Stream batches** instead of loading all data
4. **Use float32 instead of float64** for vectors

### Speed Optimization

1. **Mini-batch k-means** for >100K vectors
2. **Chunked distance computation** to avoid memory pressure
3. **JIT compilation** (Numba) for hot paths
4. **GPU acceleration** for k-means clustering (10-100x speedup)

### Recall Optimization

1. Increase `nprobe` (search more clusters)
2. Increase `rerank_factor` (rerank more candidates)
3. Use HNSW instead of IVF for high-precision needs
4. Skip quantization or use lighter compression (SQ instead of RaBitQ)

## Implementation Patterns

### K-Means Initialization

Use k-means++ for better cluster quality:
```python
# First centroid: random
centroids = [vectors[random.choice(n)]]

# Subsequent centroids: probability proportional to distance²
for _ in range(n_clusters - 1):
    distances = min_distance_to_centroids(vectors, centroids)
    next_idx = random_choice(weights=distances²)
```

### Empty Cluster Handling

When k-means produces empty clusters:
1. Find the largest cluster
2. Pick the farthest point from its centroid
3. Use that point as the new empty cluster centroid

### Hamming Distance Optimization

Use pre-computed popcount table for binary code comparison:
```python
# Class-level lookup table (computed once)
POPCOUNT_TABLE = [bin(i).count('1') for i in range(256)]

# Compute Hamming distances efficiently
xor_result = np.bitwise_xor(codes, query_code)
hamming = np.sum(POPCOUNT_TABLE[xor_result], axis=1)
```

## Debugging Guide

### Poor Recall

**Symptoms:** Query returns irrelevant vectors

**Check:**
- Is `nprobe` too low? Increase to search more clusters
- Are vectors normalized? Cosine similarity requires unit vectors
- Is quantization too aggressive? Try `rerank_factor` 20+ or skip quantization
- Are clusters unbalanced? Check `ivf.get_stats().avg_vectors_per_cluster`

### Slow Building

**Symptoms:** `index.build()` takes minutes+

**Solutions:**
- Enable mini-batch k-means: `use_minibatch=True, minibatch_size=4096`
- Reduce `nlist` temporarily for testing
- Use GPU backend if available

### Slow Search

**Symptoms:** `index.search()` is slow

**Solutions:**
- Reduce `nprobe` or `rerank_factor`
- Enable quantization
- Use chunked distance computation
- Consider HNSW instead of IVF

### Memory Errors

**Symptoms:** Out of memory errors

**Solutions:**
- Enable quantization
- Use memory-mapped storage
- Reduce `chunk_size` for distance computation
- Process in smaller batches

## Resources

### references/
- [index_structures.md](references/index_structures.md) - Detailed IVF, HNSW, Flat index specifications
- [quantization.md](references/quantization.md) - PQ, RaBitQ, SQ implementation details
- [optimization.md](references/optimization.md) - Performance tuning and system-level optimizations
- [algorithms.md](references/algorithms.md) - K-means, distance computation, clustering patterns

### scripts/
- [benchmark.py](scripts/benchmark.py) - Performance benchmarking utility
