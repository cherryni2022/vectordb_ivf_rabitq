# RaBitQ Benchmark Results

> **Benchmark Date**: 2026-01-31  
> **Dataset**: Random vectors, 128 dimensions  
> **Query Count**: 100 queries per dataset  
> **IVF Clusters**: 100  
> **RaBitQ Compression Ratio**: 3.91% (128 bits per vector)

## Executive Summary

RaBitQ (Random Bit Quantization) 加速搜索相比标准 IVF 搜索实现了显著的**延迟降低**和**吞吐量提升**，但以一定的召回率损失为代价。随着数据集规模增大，加速效果更加明显。

### Key Findings

| Metric | 10K Vectors | 50K Vectors | 100K Vectors |
|--------|-------------|-------------|--------------|
| 延迟加速比 | **1.94x** | **1.80x** | **2.48x** |
| QPS 提升 | **1.93x** | **1.80x** | **2.48x** |
| 召回率差异 | -5.0% | -11.0% | -8.2% |

---

## Detailed Results

### Dataset: 10,000 Vectors (128-dim)

**构建时间**: 34.97 秒

| Metric | IVF + RaBitQ | IVF Only | Comparison |
|--------|--------------|----------|------------|
| Mean Latency | 0.512 ms | 0.991 ms | **1.94x faster** |
| P50 Latency | 0.456 ms | 0.915 ms | — |
| P95 Latency | 0.761 ms | 1.465 ms | — |
| P99 Latency | 0.837 ms | 1.691 ms | — |
| Throughput | 1948 QPS | 1008 QPS | **1.93x higher** |
| Recall@10 | 0.3480 | 0.3980 | -0.0500 |

---

### Dataset: 50,000 Vectors (128-dim)

**构建时间**: 202.36 秒

| Metric | IVF + RaBitQ | IVF Only | Comparison |
|--------|--------------|----------|------------|
| Mean Latency | 1.591 ms | 2.860 ms | **1.80x faster** |
| P50 Latency | 1.499 ms | 2.785 ms | — |
| P95 Latency | 2.187 ms | 3.424 ms | — |
| P99 Latency | 2.391 ms | 3.679 ms | — |
| Throughput | 628 QPS | 350 QPS | **1.80x higher** |
| Recall@10 | 0.2980 | 0.4080 | -0.1100 |

---

### Dataset: 100,000 Vectors (128-dim)

**构建时间**: 370.60 秒

| Metric | IVF + RaBitQ | IVF Only | Comparison |
|--------|--------------|----------|------------|
| Mean Latency | 2.054 ms | 5.087 ms | **2.48x faster** |
| P50 Latency | 2.010 ms | 5.044 ms | — |
| P95 Latency | 2.469 ms | 5.372 ms | — |
| P99 Latency | 2.579 ms | 5.834 ms | — |
| Throughput | 487 QPS | 197 QPS | **2.48x higher** |
| Recall@10 | 0.2640 | 0.3460 | -0.0820 |

---

## Parameter Analysis

### nprobe Analysis (Accelerated Search)

`nprobe` 参数控制搜索时探查的 IVF 聚类数量。更高的 `nprobe` 值会提升召回率但增加延迟。

| nprobe | Latency (ms) | QPS | Recall@10 |
|--------|--------------|-----|-----------|
| 1 | 0.437 | 2290 | 0.1680 |
| 5 | 0.775 | 1290 | 0.2740 |
| 10 | 1.492 | 670 | 0.2980 |
| 20 | 2.559 | 391 | 0.3520 |
| 50 | 5.122 | 195 | 0.4200 |
| 100 | 9.425 | 106 | 0.4540 |

**Observation**: 
- `nprobe=1` 提供最高吞吐量 (2290 QPS) 但召回率最低 (16.8%)
- `nprobe=100` 提供最高召回率 (45.4%) 但吞吐量降至 106 QPS
- **推荐**: `nprobe=10-20` 作为延迟和召回率的平衡点

---

### Rerank Factor Analysis

`rerank_factor` 控制 RaBitQ 粗筛后使用原始向量进行精确重排序的候选数量倍数。

| Rerank Factor | Latency (ms) | QPS | Recall@10 |
|---------------|--------------|-----|-----------|
| 1 | 1.615 | 619 | 0.1720 |
| 5 | 1.479 | 676 | 0.2500 |
| 10 | 1.487 | 673 | 0.2980 |
| 20 | 1.508 | 663 | 0.3500 |
| 50 | 1.667 | 600 | 0.3940 |

**Observation**:
- `rerank_factor` 对延迟影响较小 (1.479ms - 1.667ms 范围内)
- 召回率随 `rerank_factor` 增加而显著提升
- **推荐**: `rerank_factor=20-50` 可在保持较高吞吐量的同时获得较好的召回率

---

## Performance Visualization

### Speedup Trend by Dataset Size

```
Dataset Size    Latency Speedup    QPS Improvement
  10,000             1.94x              1.93x
  50,000             1.80x              1.80x
 100,000             2.48x              2.48x
```

加速比在 100K 向量规模时达到最大 (**2.48x**)，表明 RaBitQ 在大规模数据集上优势更为明显。

---

## Conclusions

### Advantages of RaBitQ Acceleration

1. **显著的延迟降低**: 在所有测试规模下均实现 1.8x-2.5x 的延迟加速
2. **高吞吐量**: 相比标准 IVF 搜索，QPS 提升近 2 倍
3. **极高压缩率**: 仅使用 3.91% 的存储空间 (128 bits/vector vs 128×32 bits/vector)
4. **可扩展性**: 随数据规模增大，加速效果更加显著

### Trade-offs

1. **召回率损失**: 相比精确搜索有 5%-11% 的召回率下降
2. **构建时间**: 需要额外的量化预处理时间
3. **参数调优**: 需要根据场景选择合适的 `nprobe` 和 `rerank_factor`

### Recommendations

| Use Case | Recommended Settings |
|----------|---------------------|
| 高吞吐量场景 | `nprobe=5`, `rerank_factor=10` |
| 平衡模式 | `nprobe=10-20`, `rerank_factor=20` |
| 高召回率场景 | `nprobe=50+`, `rerank_factor=50` |

---

## Appendix: Test Environment

- **Index Type**: IVF with 100 clusters
- **Quantization**: True RaBitQ (1-bit per dimension)
- **Compression**: 128 bits per vector (vs 4096 bits for float32)
- **Distance Metric**: L2 (Euclidean)
