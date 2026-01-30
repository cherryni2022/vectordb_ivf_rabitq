# Vector Quantization Techniques

## Overview

Quantization reduces memory footprint and accelerates distance computations by representing vectors with fewer bits. Trade-off: higher compression = lower accuracy.

## Product Quantization (PQ)

### Concept
Split vectors into `nsubq` sub-vectors, each quantized independently to `nbits` centroids.

**Compression:** `d * 4` bytes → `nsubq * nbits` bytes

**Example:** 128-dim vectors, `nsubq=16`, `nbits=8`
- Original: 128 * 4 = 512 bytes
- Compressed: 16 * 8 = 128 bytes (4x compression)

### Training

1. Split vectors into `nsubq` sub-vectors
2. For each subspace, run k-means with `2^nbits` centroids
3. Store centroid codes

```python
def train_pq(vectors, nsubq, nbits):
    n_vectors, d = vectors.shape
    sub_dim = d // nsubq

    centroids = []

    for i in range(nsubq):
        # Extract subspace
        start, end = i * sub_dim, (i + 1) * sub_dim
        subspace = vectors[:, start:end]

        # K-means on subspace
        sub_centroids = kmeans(subspace, n_clusters=2**nbits)
        centroids.append(sub_centroids)

    return centroids
```

### Encoding

Assign each sub-vector to nearest centroid code:

```python
def encode_pq(vectors, centroids):
    n_vectors, d = vectors.shape
    nsubq = len(centroids)
    sub_dim = d // nsubq

    codes = np.zeros((n_vectors, nsubq), dtype=np.uint8)

    for i in range(nsubq):
        start, end = i * sub_dim, (i + 1) * sub_dim
        subspace = vectors[:, start:end]

        # Assign to nearest centroid
        distances = compute_distance(subspace, centroids[i])
        codes[:, i] = np.argmin(distances, axis=1)

    return codes
```

### Distance Computation

Lookup pre-computed distances between query sub-vectors and centroids:

```python
# Pre-compute distance tables once per query
def compute_distance_tables(query, centroids):
    nsubq = len(centroids)
    sub_dim = query.shape[0] // nsubq

    tables = []
    for i in range(nsubq):
        start, end = i * sub_dim, (i + 1) * sub_dim
        sub_query = query[start:end]

        # Distance to each centroid in this subspace
        distances = compute_distance(sub_query, centroids[i])
        tables.append(distances)

    return tables

# Fast distance lookup during search
def pq_distance(codes, distance_tables):
    # Sum distances from lookup table
    return sum(distance_tables[i][codes[:, i]] for i in range(len(codes)))
```

---

## True RaBitQ (Randomized Binary Quantization)

### Concept
Apply random orthogonal rotation, then quantize each dimension to 1 bit (sign). Leverages high-dimensional measurement concentration.

**Compression:** `d * 4` bytes → `ceil(d/8) + 4` bytes (~32x compression)

**Key insight:** For random-rotated normalized vectors, cosine similarity correlates with Hamming distance of sign bits.

### Distance Approximation

```
For normalized vectors u, v after random rotation:
  cos(θ) ≈ 1 - 2 * H(sign(u), sign(v)) / d

For original vectors x, y with norms ||x||, ||y||:
  ||x - y||² = ||x||² + ||y||² - 2 * ||x|| * ||y|| * cos(θ)
```

Where H is Hamming distance and d is dimension.

### Training

Generate random orthogonal matrix via QR decomposition:

```python
def train_rabitq(vectors, dimension, seed=42):
    np.random.seed(seed)

    # Random orthogonal matrix (uniform rotation)
    random_matrix = np.random.randn(dimension, dimension)
    rotation_matrix, _ = np.linalg.qr(random_matrix)

    # Compute norm statistics
    norms = np.linalg.norm(vectors, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    return {'rotation_matrix': rotation_matrix, 'mean_norm': mean_norm}
```

### Encoding

1. Compute and store norms
2. Normalize vectors
3. Apply random rotation
4. Take sign (1 if positive, 0 if negative)
5. Pack bits into bytes

```python
def encode_rabitq(vectors, rotation_matrix):
    n_vectors, d = vectors.shape

    # Step 1: Compute norms
    norms = np.linalg.norm(vectors, axis=1)

    # Step 2: Normalize
    normalized = vectors / (norms[:, np.newaxis] + 1e-8)

    # Step 3: Apply rotation
    rotated = np.dot(normalized, rotation_matrix)

    # Step 4: Sign quantization
    signs = (rotated > 0).astype(np.uint8)

    # Step 5: Pack bits (8 per byte)
    binary_codes = np.packbits(signs, axis=1)

    return binary_codes, norms
```

### Hamming Distance (Fast)

Use XOR + pre-computed popcount table:

```python
# Class-level lookup table
POPCOUNT_TABLE = [bin(i).count('1') for i in range(256)]

def hamming_distance(code1, codes2):
    """Compute Hamming distance between code1 and all codes in codes2"""
    xor_result = np.bitwise_xor(codes2, code1)
    return np.sum(POPCOUNT_TABLE[xor_result], axis=1)
```

### Approximate L2 Distance

```python
def approximate_l2(query, database_codes, database_norms, rotation_matrix, d):
    # Encode query
    query_code, query_norm = encode_rabitq(query.reshape(1, -1), rotation_matrix)

    # Hamming distances
    hamming_dists = hamming_distance(query_code[0], database_codes)

    # Cosine approximation
    cos_approx = 1.0 - 2.0 * hamming_dists / d

    # L2 approximation
    query_norm_sq = query_norm ** 2
    database_norms_sq = database_norms ** 2

    approx_dist = query_norm_sq + database_norms_sq - 2 * query_norm * database_norms * cos_approx
    return np.maximum(approx_dist, 0.0)  # Clip negative values
```

---

## Scalar Quantization (SQ)

### Concept
Quantize each dimension independently to 8-bit (uint8) by linear scaling.

**Compression:** `d * 4` bytes → `d` bytes (4x compression)

**Accuracy loss:** Minimal (~1-2% recall)

### Training

Compute min/max per dimension for scaling:

```python
def train_sq(vectors):
    # Range per dimension
    d = vectors.shape[1]
    min_vals = np.min(vectors, axis=0)
    max_vals = np.max(vectors, axis=0)

    # Avoid division by zero
    ranges = np.maximum(max_vals - min_vals, 1e-8)

    # Scale factors: (255 - 0) / range
    scales = 255.0 / ranges
    offsets = -min_vals * scales

    return {'scales': scales, 'offsets': offsets}
```

### Encoding

```python
def encode_sq(vectors, scales, offsets):
    # Linear scaling to uint8
    quantized = np.round(vectors * scales + offsets)
    return np.clip(quantized, 0, 255).astype(np.uint8)
```

### Distance Computation

Use Asymmetric Distance Computation (ADC): query in float32, database in uint8:

```python
def sq_distance(query, database_q, scales, offsets):
    # Dequantize database on-the-fly
    database_f = (database_q - offsets) / scales

    # Compute distance
    return np.sum((query - database_f) ** 2, axis=1)
```

---

## Quantization Comparison

| Method | Compression | Speed | Accuracy | Complexity |
|--------|-------------|-------|----------|------------|
| None | 1x | Baseline | 100% | O(1) |
| SQ | 4x | ~0.9x | 98-99% | Low |
| PQ (nbits=8) | 4-8x | ~0.7x | 95-98% | Medium |
| RaBitQ | ~32x | ~0.3x | 90-95% | Low-Medium |

---

## Two-Stage Search with Reranking

Use quantization for fast coarse filtering, then exact reranking:

```python
def search_with_rerank(query, database, quantizer, k, rerank_factor=10):
    n_candidates = k * rerank_factor

    # Stage 1: Coarse search with quantization
    approx_dists = quantizer.compute_distances(query)
    candidate_indices = np.argsort(approx_dists)[:n_candidates]

    # Stage 2: Exact reranking
    candidate_vectors = database[candidate_indices]
    exact_dists = compute_l2_distance(query, candidate_vectors)

    # Return top-k by exact distance
    top_k = np.argpartition(exact_dists, k-1)[:k]
    return candidate_indices[top_k], exact_dists[top_k]
```

**Benefits:**
- Speed of quantized search
- Accuracy of exact search
- Controllable quality via `rerank_factor`

---

## Best Practices

1. **Always normalize vectors** before RaBitQ (relies on unit vectors)
2. **Use reranking** with RaBitQ for better accuracy
3. **Tune `rerank_factor`**: 10 for general use, 20+ for high precision
4. **Store norms separately** for L2 distance reconstruction
5. **Use static popcount table** for Hamming distance
6. **Profile compression ratio** to validate memory savings
7. **Benchmark recall@k** before/after quantization
