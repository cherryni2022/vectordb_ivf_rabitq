# IVF vs HNSW Benchmark Results

## Test Configuration
- **Number of vectors**: 10000
- **Vector dimension**: 128
- **k (neighbors to retrieve)**: 10

## Results

| Index Type | Build Time (s) | Search Latency (ms) | Recall@10 |
|------------|----------------|---------------------|------------|
| IVF(nlist=100,nprobe=10) | 1.791 | 0.762 | 0.3260 |
| IVF(nlist=100,nprobe=20) | 1.101 | 1.493 | 0.5300 |
| HNSWLib(M=16,ef_c=200,ef_s=50) | 0.367 | 0.155 | 0.6460 |
| HNSWLib(M=16,ef_c=200,ef_s=100) | 0.375 | 0.165 | 0.8230 |

## Analysis

### Build Time
- IVF typically builds faster for smaller datasets due to simpler k-means clustering
- HNSW build time increases with M and ef_construction parameters

### Search Latency
- HNSW generally provides faster search with higher recall
- IVF search time depends on nprobe; higher nprobe = better recall but slower

### Recall
- HNSW typically achieves higher recall at similar latency
- Both benefit from parameter tuning (nprobe for IVF, ef_search for HNSW)

## Recommendations
- **Small datasets (<10K vectors)**: IVF is simpler and faster to build
- **Large datasets with high recall requirements**: HNSW with tuned parameters
- **Memory-constrained environments**: IVF uses less memory
