# HNSW Index Implementation

## Overview

This document describes the HNSW (Hierarchical Navigable Small World) index implementation for vectordb.

## Algorithm Description

HNSW is a graph-based approximate nearest neighbor search algorithm that builds a multi-layer navigable small world graph. It offers:

- **High Recall**: Typically 90%+ recall with proper parameter tuning
- **Fast Search**: O(log N) search complexity
- **Incremental Updates**: Supports adding vectors without full rebuild

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Max connections per node in layers > 0 | 16 |
| `M0` | Max connections in layer 0 (= 2*M) | 32 |
| `ef_construction` | Search width during index build | 200 |
| `ef_search` | Search width during query | 50 |

### How It Works

1. **Layered Graph Structure**: Vectors are placed in multiple layers. Higher layers have fewer nodes but longer-range connections.

2. **Greedy Search**: Queries start from the top layer, greedily moving to closer nodes, then descend to lower layers.

3. **Bottom Layer Search**: The final layer 0 search with `ef_search` candidates provides the actual results.

## Usage

### Standalone HNSW Index

```python
from vectordb.index.hnsw_index import HNSWIndex
from vectordb.core.config import HNSWIndexConfig
import numpy as np

# Configure
config = HNSWIndexConfig(
    M=16,
    ef_construction=200,
    ef_search=50,
    metric="l2"
)

# Build index
index = HNSWIndex(config)
vectors = np.random.random((10000, 128)).astype(np.float32)
index.build(vectors)

# Search
query = np.random.random(128).astype(np.float32)
results = index.search(query, k=10)
# Returns: [(vector_idx, distance), ...]
```

### VectorDB with HNSW

```python
from vectordb.core.vector_db import VectorDB
from vectordb.core.config import VectorDBConfig, HNSWIndexConfig

config = VectorDBConfig(
    index_type="hnsw",  # Select HNSW instead of IVF
    hnsw=HNSWIndexConfig(M=16, ef_construction=200, ef_search=50)
)

db = VectorDB(config=config)
db.add_vectors(vectors)
db.build()

results = db.search(query, k=10)
```

## Benchmark Results

See [IVF vs HNSW Benchmark](./ivf_vs_hnsw_benchmark.md) for detailed performance comparison.

### Summary

| Metric | IVF (nprobe=20) | HNSW (ef_search=100) |
|--------|-----------------|----------------------|
| Build Time | ~0.7s | ~55s |
| Search Latency | ~1.5ms | ~6.9ms |
| Recall@10 | 57% | 93% |

**Recommendations:**
- Use **IVF** when build time is critical or memory is limited
- Use **HNSW** when recall is the priority

## Files

- `vectordb/index/hnsw_index.py` - HNSW index implementation
- `vectordb/core/config.py` - `HNSWIndexConfig` configuration class
- `tests/test_hnsw.py` - Unit tests
- `benchmarks/benchmark_ivf_vs_hnsw.py` - Benchmark script

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
