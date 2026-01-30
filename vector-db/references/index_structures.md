# Index Structures for Vector Search

## Flat Index

### Overview
Brute-force linear scan over all vectors. Computes exact distances between query and every database vector.

**Complexity:** O(n) query, O(1) insert

**Use Cases:**
- Small datasets (<10K vectors)
- Applications requiring exact results
- When building other indexes (e.g., as baseline)

**Pros:**
- Exact nearest neighbors
- Simple implementation
- No training phase

**Cons:**
- Query time scales linearly with dataset size
- Not suitable for large datasets

### Implementation

```python
def flat_search(query, database, k, metric='l2'):
    """Linear scan for exact nearest neighbors"""
    if metric == 'l2':
        # Optimized L2: ||q||² + ||p||² - 2*q*p
        q_norm = np.sum(query ** 2)
        p_norms = np.sum(database ** 2, axis=1)
        distances = q_norm + p_norms - 2 * np.dot(query, database.T)
    elif metric == 'ip':
        distances = -np.dot(query, database.T)
    elif metric == 'cosine':
        distances = 1 - np.dot(query, database.T) / (
            np.linalg.norm(query) * np.linalg.norm(database, axis=1)
        )

    top_k = np.argpartition(distances, k-1)[:k]
    return top_k, distances[top_k]
```

---

## IVF (Inverted File Index)

### Overview
Partitions vectors into clusters via k-means. During search, only probes `nprobe` closest clusters to query.

**Complexity:** O(n/nlist * nprobe) query, O(n log n) build

**Use Cases:**
- Datasets >10K vectors
- Sub-millisecond search requirements
- Accept ~5-15% recall loss

**Pros:**
- Significant speedup over flat search
- Tunable accuracy via `nprobe`
- Works well with quantization

**Cons:**
- Requires build phase
- Recall depends on `nprobe`
- Cluster quality affects performance

### Configuration Parameters

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| `nlist` | Number of clusters | `4 * sqrt(n)` to `100 * sqrt(n)` | More clusters = better partitioning, slower build |
| `nprobe` | Clusters to search | `nlist/10` to `nlist/2` | Higher = better recall, slower search |
| `metric` | Distance metric | 'l2', 'ip' | Determines similarity measure |

### Build Process

1. **K-means clustering** to find `nlist` centroids
2. **Assign** each vector to nearest centroid
3. **Build inverted lists** mapping cluster IDs to vector indices

```python
def build_ivf(vectors, nlist, nprobe):
    # Step 1: K-means clustering
    centroids = kmeans(vectors, n_clusters=nlist)

    # Step 2: Assign vectors to clusters
    distances = compute_distances(vectors, centroids)
    assignments = np.argmin(distances, axis=1)

    # Step 3: Build inverted lists
    inverted_lists = [[] for _ in range(nlist)]
    for idx, cluster_id in enumerate(assignments):
        inverted_lists[cluster_id].append(idx)

    return IVFIndex(centroids, inverted_lists, nprobe)
```

### Search Process

1. **Find closest centroids** to query
2. **Get candidates** from those clusters
3. **Compute distances** only to candidates
4. **Return top-k**

```python
def ivf_search(query, ivf_index, k):
    # Step 1: Find closest centroids
    centroid_dists = compute_distance(query, ivf_index.centroids)
    closest_clusters = np.argsort(centroid_dists)[:ivf_index.nprobe]

    # Step 2: Collect candidates
    candidates = []
    for cluster_id in closest_clusters:
        candidates.extend(ivf_index.inverted_lists[cluster_id])

    # Step 3: Compute distances to candidates only
    candidate_vectors = ivf_index.vectors[candidates]
    distances = compute_distance(query, candidate_vectors)

    # Step 4: Return top-k
    top_k = np.argpartition(distances, k-1)[:k]
    return np.array(candidates)[top_k], distances[top_k]
```

### K-Means++ Initialization

Standard k-means randomly picks initial centroids, leading to poor clusters. K-means++ spreads centroids more evenly:

```python
def kmeans_plusplus_init(vectors, n_clusters):
    n = len(vectors)
    centroids = []

    # First centroid: random
    centroids.append(vectors[np.random.choice(n)])

    for _ in range(n_clusters - 1):
        # Distance to nearest existing centroid
        dists = np.min([np.linalg.norm(vectors - c, axis=1) for c in centroids], axis=0)
        dists_sq = dists ** 2

        # Choose next centroid with probability proportional to distance²
        probs = dists_sq / np.sum(dists_sq)
        next_idx = np.random.choice(n, p=probs)
        centroids.append(vectors[next_idx])

    return np.array(centroids)
```

### Empty Cluster Handling

K-means may produce empty clusters. Solution: reinitialize from largest cluster:

```python
def handle_empty_clusters(vectors, centroids, assignments):
    empty_clusters = [i for i in range(len(centroids))
                     if i not in np.unique(assignments)]

    for cluster_id in empty_clusters:
        # Find largest cluster
        cluster_sizes = np.bincount(assignments)
        largest_cluster = np.argmax(cluster_sizes)

        # Find farthest point from largest cluster's centroid
        mask = assignments == largest_cluster
        cluster_vectors = vectors[mask]
        dists = np.linalg.norm(cluster_vectors - centroids[largest_cluster], axis=1)
        farthest_point = cluster_vectors[np.argmax(dists)]

        # Initialize empty cluster with farthest point
        centroids[cluster_id] = farthest_point

    return centroids
```

### Mini-Batch K-Means

For large datasets (>100K vectors), mini-batch k-means is faster:

```python
def minibatch_kmeans(vectors, n_clusters, batch_size=1024, max_iter=100):
    n, d = vectors.shape

    # Initialize with k-means++ on sample
    centroids = kmeans_plusplus_init(vectors[np.random.choice(n, batch_size*10)], n_clusters)
    centroid_counts = np.zeros(n_clusters)

    for iteration in range(max_iter):
        # Sample random mini-batch
        batch_indices = np.random.choice(n, batch_size, replace=False)
        batch = vectors[batch_indices]

        # Assign to nearest centroids
        distances = compute_distance(batch, centroids)
        assignments = np.argmin(distances, axis=1)

        # Update centroids with streaming mean
        for i in range(n_clusters):
            mask = assignments == i
            if np.any(mask):
                cluster_vectors = batch[mask]
                n_new = np.sum(mask)
                old_count = centroid_counts[i]
                new_count = old_count + n_new

                # Weighted update
                centroids[i] = (old_count * centroids[i] + np.sum(cluster_vectors, axis=0)) / new_count
                centroid_counts[i] = new_count

    return centroids
```

---

## HNSW (Hierarchical Navigable Small World)

### Overview
Graph-based index with multiple layers. Coarse layers use fewer edges for fast navigation, fine layers provide accuracy.

**Complexity:** O(log n) query, O(n log n) build

**Use Cases:**
- High-precision requirements
- Mixed read/write workloads
- When IVF recall is insufficient

**Pros:**
- Near-exact recall
- Sub-logarithmic query time
- Supports incremental inserts

**Cons:**
- Higher memory than IVF
- Build time increases with complexity
- More complex implementation

### Parameters

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| `M` | Max connections per node | 5-64 | Higher = better recall, more memory |
| `ef_construction` | Candidate pool during build | 40-400 | Higher = better index quality, slower build |
| `ef_search` | Candidate pool during search | 10-200 | Higher = better recall, slower search |

### Structure

```
Layer 2: o------o       (very sparse, navigation)
Layer 1: o--o--o--o     (sparse)
Layer 0: o-o-o-o-o-o-o   (dense, contains all vectors)
```

### Search Algorithm

1. Start at top layer's entry point
2. Greedy descent through layers
3. In bottom layer, explore `ef_search` candidates
4. Return best `k` found

---

## Index Selection Guide

| Dataset Size | Latency Target | Accuracy Target | Recommended Index |
|--------------|----------------|-----------------|------------------|
| <10K | Any | Exact | Flat |
| 10K-1M | <10ms | 90-95% | IVF |
| 1M-100M | <10ms | 90-95% | IVF + Quantization |
| 100M+ | <10ms | 90-95% | IVF + RaBitQ + Distributed |
| Any | <1ms | 95-99% | HNSW |
| Any | Any | 99%+ | HNSW (high ef_search) or Flat |
