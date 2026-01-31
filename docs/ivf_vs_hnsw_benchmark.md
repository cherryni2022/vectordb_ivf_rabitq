# IVF vs HNSW Benchmark Results

## Test Configuration
- **Number of vectors**: 5000
- **Vector dimension**: 128
- **k (neighbors to retrieve)**: 10

## Results

| Index Type | Build Time (s) | Search Latency (ms) | Recall@10 |
|------------|----------------|---------------------|------------|
| IVF(nlist=100,nprobe=10) | 0.764 | 0.670 | 0.4000 |
| IVF(nlist=100,nprobe=20) | 0.690 | 1.515 | 0.5720 |
| HNSW(M=16,ef_c=200,ef_s=50) | 54.696 | 4.228 | 0.8440 |
| HNSW(M=16,ef_c=200,ef_s=100) | 55.512 | 6.850 | 0.9320 |
| HNSW(M=32,ef_c=200,ef_s=50) | 118.944 | 6.863 | 0.9500 |

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
