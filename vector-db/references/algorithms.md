# Core Algorithms for Vector Databases

## Distance Metrics

### L2 (Euclidean) Distance

```python
def l2_distance(a, b):
    """Squared Euclidean distance"""
    return np.sum((a - b) ** 2)

def l2_distance_optimized(queries, points):
    """
    Vectorized L2 distance: ||q - p||² = ||q||² + ||p||² - 2*q*p
    More efficient than naive computation.
    """
    q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
    p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
    dot = np.dot(queries, points.T)
    return q_norm + p_norm - 2 * dot

# Single query version
def l2_distance_single(query, points):
    q_norm = np.sum(query ** 2)
    p_norms = np.sum(points ** 2, axis=1)
    dot = np.dot(query, points.T)
    return q_norm + p_norms - 2 * dot
```

### Inner Product

```python
def inner_product(queries, points):
    """Negative inner product (for minimization)"""
    return -np.dot(queries, points.T)

def inner_product_single(query, points):
    return -np.dot(query, points.T)
```

### Cosine Similarity

```python
def cosine_distance(queries, points):
    """
    Cosine distance = 1 - cosine_similarity
    Cosine similarity = (q . p) / (||q|| * ||p||)
    """
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    p_norms = np.linalg.norm(points, axis=1, keepdims=True).T
    dot = np.dot(queries, points.T)

    # Handle zero-length vectors
    q_norms = np.maximum(q_norms, 1e-8)
    p_norms = np.maximum(p_norms, 1e-8)

    similarity = dot / (q_norms * p_norms)
    return 1 - similarity

# Single query version
def cosine_distance_single(query, points):
    query = query / (np.linalg.norm(query) + 1e-8)
    points = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-8)
    return 1 - np.dot(query, points.T)
```

---

## K-Means Clustering

### Standard K-Means

```python
def kmeans(vectors, n_clusters, max_iter=100, tol=1e-4, random_state=None):
    """
    Standard k-means clustering with k-means++ initialization
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_vectors, dimension = vectors.shape

    # Initialize centroids with k-means++
    centroids = kmeans_plusplus_init(vectors, n_clusters)

    for iteration in range(max_iter):
        # Assign to nearest centroid
        distances = l2_distance_optimized(vectors, centroids)
        assignments = np.argmin(distances, axis=1)

        # Update centroids (vectorized)
        new_centroids = vectorized_centroid_update(
            vectors, assignments, n_clusters, dimension
        )

        # Handle empty clusters
        empty_clusters = np.where(np.bincount(assignments, minlength=n_clusters) == 0)[0]
        if len(empty_clusters) > 0:
            new_centroids = handle_empty_clusters(
                vectors, assignments, new_centroids, empty_clusters
            )

        # Check convergence
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        if shift < tol:
            break

        centroids = new_centroids

    return centroids
```

### K-Means++ Initialization

```python
def kmeans_plusplus_init(vectors, n_clusters):
    """
    K-means++ initialization: spread centroids more evenly
    """
    n_vectors = vectors.shape[0]

    # First centroid: random
    centroids = [vectors[np.random.choice(n_vectors)]]

    for _ in range(n_clusters - 1):
        # Distance to nearest existing centroid
        current_centroids = np.array(centroids)
        distances = l2_distance_optimized(vectors, current_centroids)
        min_distances = np.min(distances, axis=1)

        # Ensure non-negative
        min_distances = np.maximum(min_distances, 0.0)

        # Choose with probability proportional to distance²
        total = np.sum(min_distances ** 2)
        if total == 0:
            # All distances zero (e.g., duplicates), pick uniformly
            probabilities = np.ones(n_vectors) / n_vectors
        else:
            probabilities = min_distances ** 2 / total

        next_idx = np.random.choice(n_vectors, p=probabilities)
        centroids.append(vectors[next_idx])

    return np.array(centroids)
```

### Vectorized Centroid Update

```python
def vectorized_centroid_update(vectors, assignments, n_clusters, dimension):
    """
    O(n) centroid update using np.add.at and np.bincount
    """
    # Count vectors per cluster
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.float32)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    # Sum vectors per cluster
    new_centroids = np.zeros((n_clusters, dimension), dtype=np.float32)
    np.add.at(new_centroids, assignments, vectors)

    # Compute mean
    new_centroids /= counts[:, np.newaxis]

    return new_centroids
```

### Empty Cluster Handling

```python
def handle_empty_clusters(vectors, assignments, centroids, empty_clusters):
    """
    Reinitialize empty clusters from farthest point in largest cluster
    """
    cluster_sizes = np.bincount(assignments, minlength=len(centroids))

    for cluster_idx in empty_clusters:
        # Find largest cluster
        largest_cluster = np.argmax(cluster_sizes)

        # Find farthest point in largest cluster
        mask = assignments == largest_cluster
        cluster_vectors = vectors[mask]
        cluster_centroid = centroids[largest_cluster]

        distances = np.sum((cluster_vectors - cluster_centroid) ** 2, axis=1)
        farthest_local_idx = np.argmax(distances)

        # Initialize empty cluster with farthest point
        centroids[cluster_idx] = cluster_vectors[farthest_local_idx]

        # Update sizes (will be corrected next iteration)
        cluster_sizes[largest_cluster] -= 1
        cluster_sizes[cluster_idx] = 1

    return centroids
```

### Mini-Batch K-Means

```python
def minibatch_kmeans(vectors, n_clusters, batch_size=1024, max_iter=100, tol=1e-4):
    """
    Faster k-means for large datasets using mini-batches
    """
    n_vectors, dimension = vectors.shape
    batch_size = min(batch_size, n_vectors)

    # Initialize with k-means++ on sample
    sample_size = min(batch_size * 10, n_vectors)
    sample_indices = np.random.choice(n_vectors, sample_size, replace=False)
    centroids = kmeans_plusplus_init(vectors[sample_indices], n_clusters)

    # Track counts for streaming average
    centroid_counts = np.zeros(n_clusters, dtype=np.float32)

    prev_centroids = centroids.copy()

    for iteration in range(max_iter):
        # Sample mini-batch
        batch_indices = np.random.choice(n_vectors, batch_size, replace=False)
        batch = vectors[batch_indices]

        # Assign to nearest centroids
        distances = l2_distance_optimized(batch, centroids)
        assignments = np.argmin(distances, axis=1)

        # Update centroids with streaming average
        for i in range(n_clusters):
            mask = assignments == i
            if np.any(mask):
                cluster_vectors = batch[mask]
                n_new = np.sum(mask)

                old_count = centroid_counts[i]
                new_count = old_count + n_new

                # Weighted update: c_new = (old * c_old + sum) / new_count
                centroids[i] = (
                    old_count * centroids[i] + np.sum(cluster_vectors, axis=0)
                ) / new_count
                centroid_counts[i] = new_count

        # Check convergence every 10 iterations
        if iteration > 0 and iteration % 10 == 0:
            shift = np.max(np.linalg.norm(centroids - prev_centroids, axis=1))
            if shift < tol:
                break
            prev_centroids = centroids.copy()

    return centroids
```

---

## Hamming Distance

### Optimized with Popcount Table

```python
# Class-level lookup table (computed once)
POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def hamming_distance_packed(query_code, database_codes):
    """
    Compute Hamming distance between query and database codes
    using pre-computed popcount table.

    query_code: shape (n_bytes,)
    database_codes: shape (n_vectors, n_bytes)
    """
    xor_result = np.bitwise_xor(database_codes, query_code)
    return np.sum(POPCOUNT_TABLE[xor_result], axis=1)

def hamming_distance_single(code1, code2):
    """Hamming distance between two packed codes"""
    xor_result = code1 ^ code2
    return np.sum(POPCOUNT_TABLE[xor_result])
```

### Numba-Accelerated Version

```python
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(nopython=False):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def hamming_distance_numba(code1, code2):
    """Fast Hamming distance with Numba"""
    n = len(code1)
    result = 0
    for i in range(n):
        xor_val = code1[i] ^ code2[i]
        # Count bits in byte
        xor_val = xor_val - ((xor_val >> 1) & 0x55)
        xor_val = (xor_val & 0x33) + ((xor_val >> 2) & 0x33)
        xor_val = (xor_val + (xor_val >> 4)) & 0x0F
        result += xor_val
    return result
```

---

## Binary Code Operations

### Pack Bits

```python
def pack_bits(bits):
    """
    Pack bit array into bytes.

    bits: shape (n,) or (n_vectors, dimension) with values 0/1
    returns: shape (n_vectors, ceil(dimension/8)) uint8
    """
    if bits.ndim == 1:
        return np.packbits(bits)
    return np.packbits(bits, axis=1)

def unpack_bits(packed, dimension=None):
    """
    Unpack bytes to bit array.

    packed: shape (n_bytes,) or (n_vectors, n_bytes) uint8
    dimension: original bit dimension (inferred if None)
    """
    unpacked = np.unpackbits(packed)
    if dimension is not None:
        return unpacked[..., :dimension]
    return unpacked
```

---

## Top-K Selection

### Using Argpartition

```python
def top_k_argpartition(arr, k):
    """
    Get top-k indices using argpartition (more efficient than full sort)
    """
    k = min(k, len(arr))
    # Partial sort: elements at indices < k are the smallest k
    partitioned = np.argpartition(arr, k-1)
    return partitioned[:k]
```

### Using Nsmallest (for small k)

```python
import heapq

def top_k_nsmallest(arr, k):
    """
    Get top-k values using heapq (efficient for small k relative to array size)
    """
    k = min(k, len(arr))
    # Get indices of k smallest elements
    indexed = list(enumerate(arr))
    smallest = heapq.nsmallest(k, indexed, key=lambda x: x[1])
    return np.array([idx for idx, val in smallest])
```

### Get Top-K With Values

```python
def top_k_with_values(arr, k):
    """Return both indices and values for top-k"""
    k = min(k, len(arr))
    partitioned = np.argpartition(arr, k-1)
    top_indices = partitioned[:k]
    top_values = arr[top_indices]
    # Sort the top-k by value
    sorted_order = np.argsort(top_values)
    return top_indices[sorted_order], top_values[sorted_order]
```

---

## Random Orthogonal Matrix

### Via QR Decomposition

```python
def random_orthogonal_matrix(dimension, random_seed=None):
    """
    Generate random orthogonal matrix using QR decomposition.
    Ensures uniform random rotation.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    random_matrix = np.random.randn(dimension, dimension)
    orthogonal, _ = np.linalg.qr(random_matrix)

    # Ensure determinant is +1 (proper rotation, not reflection)
    if np.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] *= -1

    return orthogonal.astype(np.float32)
```

---

## Normalization

### L2 Normalization

```python
def normalize_l2(vectors):
    """
    L2-normalize vectors to unit length

    vectors: shape (n_vectors, dimension)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    return vectors / norms

def normalize_l2_single(vector):
    """L2-normalize single vector"""
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-8)
```

---

## Algorithms Quick Reference

| Algorithm | Complexity | Use Case |
|-----------|-------------|----------|
| L2 Distance | O(d) | General similarity |
| K-Means | O(n * k * d * iter) | Clustering |
| K-Means++ | O(n * k * d) | Cluster initialization |
| Hamming | O(n_bytes) | Binary code comparison |
| Top-K Argpartition | O(n + k log k) | Finding nearest neighbors |
| QR Decomposition | O(d³) | Random rotation matrix |
