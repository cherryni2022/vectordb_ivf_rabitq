# VectorDB Performance Testing Plan

**Date**: 2026-01-20  
**Target System**: RaBitQ + IVF Vector Database MVP  
**Python Version**: >= 3.8

---

## 📋 Overview

本文档提供完整的性能测试方案,涵盖:
1. 基准测试 (Benchmark Tests)
2. 压力测试 (Stress Tests)
3. 对比测试 (Comparison Tests)
4. 回归测试 (Recall Tests)

---

## 🛠️ 测试环境设置

### 依赖安装

```bash
pip install numpy pytest pytest-benchmark memory-profiler matplotlib pandas tqdm
```

### 目录结构

```
tests/
├── benchmarks/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   ├── test_indexing_perf.py # 索引构建性能
│   ├── test_search_perf.py   # 搜索性能
│   ├── test_memory_perf.py   # 内存使用
│   └── test_recall.py        # 召回率测试
├── datasets/
│   └── generate_datasets.py  # 测试数据生成
└── results/
    └── .gitkeep
```

---

## 📊 测试数据集规范

### 合成数据集

| 数据集 | 向量数 | 维度 | 分布 | 用途 |
|--------|--------|------|------|------|
| small | 10,000 | 128 | 随机高斯 | 快速验证 |
| medium | 100,000 | 128 | 随机高斯 | 标准基准 |
| large | 1,000,000 | 128 | 随机高斯 | 压力测试 |
| high_dim | 100,000 | 768 | 随机高斯 | 高维测试 |
| clustered | 100,000 | 128 | 聚类分布 | 真实场景模拟 |

### 数据生成脚本

```python
# tests/datasets/generate_datasets.py
import numpy as np
from typing import Tuple
import os

def generate_random_dataset(
    n_vectors: int, 
    dimension: int, 
    seed: int = 42
) -> np.ndarray:
    """生成随机高斯分布的数据集"""
    np.random.seed(seed)
    return np.random.randn(n_vectors, dimension).astype(np.float32)

def generate_clustered_dataset(
    n_vectors: int,
    dimension: int,
    n_clusters: int = 100,
    seed: int = 42
) -> np.ndarray:
    """生成聚类分布的数据集(更接近真实embedding)"""
    np.random.seed(seed)
    
    vectors_per_cluster = n_vectors // n_clusters
    vectors = []
    
    for i in range(n_clusters):
        # 每个聚类有一个中心
        center = np.random.randn(dimension).astype(np.float32) * 5
        # 生成围绕中心的向量
        cluster_vectors = center + np.random.randn(vectors_per_cluster, dimension).astype(np.float32) * 0.5
        vectors.append(cluster_vectors)
    
    # 处理余数
    remaining = n_vectors - n_clusters * vectors_per_cluster
    if remaining > 0:
        center = np.random.randn(dimension).astype(np.float32) * 5
        vectors.append(center + np.random.randn(remaining, dimension).astype(np.float32) * 0.5)
    
    return np.vstack(vectors)

def generate_query_set(
    database: np.ndarray,
    n_queries: int,
    in_distribution: bool = True
) -> np.ndarray:
    """生成查询集
    
    Args:
        database: 数据库向量
        n_queries: 查询数量
        in_distribution: True则从数据库采样,False则随机生成
    """
    if in_distribution:
        indices = np.random.choice(len(database), n_queries, replace=False)
        return database[indices].copy()
    else:
        return np.random.randn(n_queries, database.shape[1]).astype(np.float32)

def compute_ground_truth(
    database: np.ndarray,
    queries: np.ndarray,
    k: int = 100
) -> np.ndarray:
    """计算精确的 top-k 最近邻 (暴力搜索)"""
    n_queries = len(queries)
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)
    
    for i, query in enumerate(queries):
        # L2 距离
        distances = np.sum((database - query) ** 2, axis=1)
        ground_truth[i] = np.argsort(distances)[:k]
    
    return ground_truth

# 预定义数据集配置
DATASET_CONFIGS = {
    "small": {"n_vectors": 10_000, "dimension": 128},
    "medium": {"n_vectors": 100_000, "dimension": 128},
    "large": {"n_vectors": 1_000_000, "dimension": 128},
    "high_dim": {"n_vectors": 100_000, "dimension": 768},
    "sift_like": {"n_vectors": 100_000, "dimension": 128},
}

def save_dataset(name: str, vectors: np.ndarray, queries: np.ndarray, 
                 ground_truth: np.ndarray, output_dir: str = "./tests/datasets"):
    """保存数据集到磁盘"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/{name}_vectors.npy", vectors)
    np.save(f"{output_dir}/{name}_queries.npy", queries)
    np.save(f"{output_dir}/{name}_ground_truth.npy", ground_truth)

def load_dataset(name: str, data_dir: str = "./tests/datasets") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载数据集"""
    vectors = np.load(f"{data_dir}/{name}_vectors.npy")
    queries = np.load(f"{data_dir}/{name}_queries.npy")
    ground_truth = np.load(f"{data_dir}/{name}_ground_truth.npy")
    return vectors, queries, ground_truth

if __name__ == "__main__":
    print("Generating test datasets...")
    
    for name, config in DATASET_CONFIGS.items():
        print(f"  Generating {name}...")
        vectors = generate_clustered_dataset(**config)
        queries = generate_query_set(vectors, n_queries=1000)
        ground_truth = compute_ground_truth(vectors, queries, k=100)
        save_dataset(name, vectors, queries, ground_truth)
        print(f"    Saved {name}: {vectors.shape}, {queries.shape}")
    
    print("Done!")
```

---

## 🧪 测试用例

### 1. 索引构建性能测试

```python
# tests/benchmarks/test_indexing_perf.py
import pytest
import numpy as np
import time
from vectordb import VectorDB, VectorDBConfig, IVFIndexConfig, RaBitQConfig

class TestIndexingPerformance:
    """索引构建性能测试"""
    
    @pytest.fixture
    def dataset_sizes(self):
        return [1000, 10000, 50000, 100000]
    
    @pytest.fixture
    def dimension(self):
        return 128
    
    def test_build_time_vs_dataset_size(self, dataset_sizes, dimension, benchmark):
        """测试: 索引构建时间随数据集大小的变化"""
        results = []
        
        for n_vectors in dataset_sizes:
            vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
            
            db = VectorDB(use_memory_storage=True)
            db.add_vectors(vectors)
            
            start_time = time.time()
            db.build()
            build_time = time.time() - start_time
            
            results.append({
                "n_vectors": n_vectors,
                "dimension": dimension,
                "build_time_sec": build_time,
                "vectors_per_second": n_vectors / build_time
            })
            
            print(f"\n{n_vectors:,} vectors: {build_time:.2f}s ({n_vectors/build_time:.0f} vec/s)")
        
        return results
    
    def test_build_time_vs_nlist(self, dimension):
        """测试: 不同 nlist 对构建时间的影响"""
        n_vectors = 50000
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        nlist_values = [10, 50, 100, 200, 500]
        
        results = []
        for nlist in nlist_values:
            config = VectorDBConfig(
                ivf=IVFIndexConfig(nlist=nlist, nprobe=10)
            )
            db = VectorDB(config=config, use_memory_storage=True)
            db.add_vectors(vectors)
            
            start_time = time.time()
            db.build()
            build_time = time.time() - start_time
            
            results.append({
                "nlist": nlist,
                "build_time_sec": build_time
            })
            print(f"nlist={nlist}: {build_time:.2f}s")
        
        return results
    
    def test_build_time_vs_dimension(self):
        """测试: 不同维度对构建时间的影响"""
        n_vectors = 50000
        dimensions = [64, 128, 256, 512, 768]
        
        results = []
        for dim in dimensions:
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            
            db = VectorDB(use_memory_storage=True)
            db.add_vectors(vectors)
            
            start_time = time.time()
            db.build()
            build_time = time.time() - start_time
            
            results.append({
                "dimension": dim,
                "build_time_sec": build_time
            })
            print(f"dim={dim}: {build_time:.2f}s")
        
        return results

    def test_build_with_quantization(self, dimension):
        """测试: 量化对构建时间的影响"""
        n_vectors = 50000
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        # 不使用量化
        db_no_q = VectorDB(use_memory_storage=True)
        db_no_q.add_vectors(vectors.copy())
        
        start = time.time()
        db_no_q.build(use_quantization=False)
        time_no_q = time.time() - start
        
        # 使用量化
        db_with_q = VectorDB(use_memory_storage=True)
        db_with_q.add_vectors(vectors.copy())
        
        start = time.time()
        db_with_q.build(use_quantization=True)
        time_with_q = time.time() - start
        
        print(f"\nWithout quantization: {time_no_q:.2f}s")
        print(f"With quantization: {time_with_q:.2f}s")
        print(f"Overhead: {(time_with_q - time_no_q) / time_no_q * 100:.1f}%")
```

### 2. 搜索性能测试

```python
# tests/benchmarks/test_search_perf.py
import pytest
import numpy as np
import time
from vectordb import VectorDB, VectorDBConfig, IVFIndexConfig

class TestSearchPerformance:
    """搜索性能测试"""
    
    @pytest.fixture
    def prepared_db(self):
        """准备好的数据库 fixture"""
        n_vectors = 100000
        dimension = 128
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        db = VectorDB(use_memory_storage=True)
        db.add_vectors(vectors)
        db.build()
        
        return db, vectors
    
    def test_single_query_latency(self, prepared_db):
        """测试: 单查询延迟"""
        db, vectors = prepared_db
        query = vectors[0]
        
        # 预热
        for _ in range(10):
            db.search(query, k=10)
        
        # 测试
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            db.search(query, k=10)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        print(f"\nSingle query latency (ms):")
        print(f"  Mean: {latencies.mean():.3f}")
        print(f"  Median: {np.median(latencies):.3f}")
        print(f"  P95: {np.percentile(latencies, 95):.3f}")
        print(f"  P99: {np.percentile(latencies, 99):.3f}")
        
        return {
            "mean_ms": latencies.mean(),
            "p50_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
    
    def test_throughput_qps(self, prepared_db):
        """测试: 查询吞吐量 (QPS)"""
        db, vectors = prepared_db
        n_queries = 1000
        queries = vectors[:n_queries]
        
        # 预热
        db.search_batch(queries[:10], k=10)
        
        # 测试
        start = time.time()
        for query in queries:
            db.search(query, k=10)
        elapsed = time.time() - start
        
        qps = n_queries / elapsed
        print(f"\nThroughput: {qps:.0f} QPS ({elapsed:.2f}s for {n_queries} queries)")
        
        return {"qps": qps, "total_time_sec": elapsed}
    
    def test_nprobe_vs_latency(self, prepared_db):
        """测试: nprobe 对延迟的影响"""
        db, vectors = prepared_db
        query = vectors[0]
        
        nprobe_values = [1, 5, 10, 20, 50, 100]
        results = []
        
        for nprobe in nprobe_values:
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                db.search(query, k=10, nprobe=nprobe)
                latencies.append((time.perf_counter() - start) * 1000)
            
            mean_latency = np.mean(latencies)
            results.append({
                "nprobe": nprobe,
                "mean_latency_ms": mean_latency
            })
            print(f"nprobe={nprobe}: {mean_latency:.3f}ms")
        
        return results
    
    def test_k_vs_latency(self, prepared_db):
        """测试: k 值对延迟的影响"""
        db, vectors = prepared_db
        query = vectors[0]
        
        k_values = [1, 10, 50, 100, 500]
        results = []
        
        for k in k_values:
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                db.search(query, k=k)
                latencies.append((time.perf_counter() - start) * 1000)
            
            mean_latency = np.mean(latencies)
            results.append({
                "k": k,
                "mean_latency_ms": mean_latency
            })
            print(f"k={k}: {mean_latency:.3f}ms")
        
        return results
    
    def test_batch_vs_sequential(self, prepared_db):
        """测试: 批量搜索 vs 逐个搜索"""
        db, vectors = prepared_db
        n_queries = 100
        queries = vectors[:n_queries]
        
        # 逐个搜索
        start = time.time()
        for query in queries:
            db.search(query, k=10)
        sequential_time = time.time() - start
        
        # 批量搜索
        start = time.time()
        db.search_batch(queries, k=10)
        batch_time = time.time() - start
        
        print(f"\nSequential: {sequential_time:.3f}s")
        print(f"Batch: {batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.2f}x")
```

### 3. 内存测试

```python
# tests/benchmarks/test_memory_perf.py
import pytest
import numpy as np
import sys
from memory_profiler import profile
from vectordb import VectorDB

class TestMemoryPerformance:
    """内存使用测试"""
    
    def test_memory_usage_vs_size(self):
        """测试: 内存使用随数据集大小变化"""
        import tracemalloc
        
        sizes = [10000, 50000, 100000]
        dimension = 128
        
        results = []
        for n_vectors in sizes:
            tracemalloc.start()
            
            vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
            db = VectorDB(use_memory_storage=True)
            db.add_vectors(vectors)
            db.build()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # 理论最小内存 = vectors 本身
            theoretical_min = n_vectors * dimension * 4  # float32 = 4 bytes
            
            results.append({
                "n_vectors": n_vectors,
                "peak_memory_mb": peak / 1024 / 1024,
                "theoretical_min_mb": theoretical_min / 1024 / 1024,
                "overhead_ratio": peak / theoretical_min
            })
            
            print(f"\n{n_vectors:,} vectors:")
            print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
            print(f"  Theoretical min: {theoretical_min / 1024 / 1024:.1f} MB")
            print(f"  Overhead: {peak / theoretical_min:.2f}x")
        
        return results
    
    def test_quantization_memory_savings(self):
        """测试: 量化带来的内存节省"""
        import tracemalloc
        
        n_vectors = 100000
        dimension = 128
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        # 不使用量化
        tracemalloc.start()
        db_no_q = VectorDB(use_memory_storage=True)
        db_no_q.add_vectors(vectors.copy())
        db_no_q.build(use_quantization=False)
        _, peak_no_q = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 使用量化
        tracemalloc.start()
        db_with_q = VectorDB(use_memory_storage=True)
        db_with_q.add_vectors(vectors.copy())
        db_with_q.build(use_quantization=True)
        _, peak_with_q = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\nWithout quantization: {peak_no_q / 1024 / 1024:.1f} MB")
        print(f"With quantization: {peak_with_q / 1024 / 1024:.1f} MB")
        print(f"Savings: {(1 - peak_with_q / peak_no_q) * 100:.1f}%")
        
        return {
            "no_quantization_mb": peak_no_q / 1024 / 1024,
            "with_quantization_mb": peak_with_q / 1024 / 1024,
            "savings_percent": (1 - peak_with_q / peak_no_q) * 100
        }
```

### 4. 召回率测试 (最重要!)

```python
# tests/benchmarks/test_recall.py
import pytest
import numpy as np
from typing import List, Set
from vectordb import VectorDB, VectorDBConfig, IVFIndexConfig

def compute_recall_at_k(
    retrieved: List[int], 
    ground_truth: np.ndarray, 
    k: int
) -> float:
    """计算 Recall@k
    
    Recall@k = |retrieved ∩ ground_truth[:k]| / k
    """
    gt_set = set(ground_truth[:k].tolist())
    retrieved_set = set(retrieved[:k])
    
    intersection = len(gt_set & retrieved_set)
    return intersection / k

def compute_ground_truth_brute_force(
    database: np.ndarray,
    queries: np.ndarray,
    k: int = 100
) -> np.ndarray:
    """暴力搜索计算 ground truth"""
    n_queries = len(queries)
    ground_truth = np.zeros((n_queries, k), dtype=np.int64)
    
    for i, query in enumerate(queries):
        distances = np.sum((database - query) ** 2, axis=1)
        ground_truth[i] = np.argsort(distances)[:k]
    
    return ground_truth

class TestRecall:
    """召回率测试"""
    
    @pytest.fixture
    def test_data(self):
        """测试数据集: 100K 向量, 1K 查询"""
        n_database = 100000
        n_queries = 1000
        dimension = 128
        
        np.random.seed(42)
        database = np.random.randn(n_database, dimension).astype(np.float32)
        
        # 从数据库采样作为查询 (in-distribution)
        query_indices = np.random.choice(n_database, n_queries, replace=False)
        queries = database[query_indices].copy()
        
        # 计算 ground truth
        ground_truth = compute_ground_truth_brute_force(database, queries, k=100)
        
        return database, queries, ground_truth
    
    def test_recall_at_k(self, test_data):
        """测试: 不同 k 值的召回率"""
        database, queries, ground_truth = test_data
        
        db = VectorDB(use_memory_storage=True)
        db.add_vectors(database)
        db.build()
        
        k_values = [1, 10, 50, 100]
        
        for k in k_values:
            recalls = []
            for i, query in enumerate(queries):
                results = db.search(query, k=k)
                recall = compute_recall_at_k(results.ids, ground_truth[i], k)
                recalls.append(recall)
            
            mean_recall = np.mean(recalls)
            print(f"Recall@{k}: {mean_recall:.4f}")
            
            # 召回率阈值断言
            if k == 1:
                assert mean_recall >= 0.85, f"Recall@1 too low: {mean_recall}"
            elif k == 10:
                assert mean_recall >= 0.90, f"Recall@10 too low: {mean_recall}"
    
    def test_recall_vs_nprobe(self, test_data):
        """测试: nprobe 对召回率的影响"""
        database, queries, ground_truth = test_data
        
        nlist = 100  # 100 clusters
        nprobe_values = [1, 5, 10, 20, 50, 100]
        
        config = VectorDBConfig(
            ivf=IVFIndexConfig(nlist=nlist, nprobe=10)
        )
        db = VectorDB(config=config, use_memory_storage=True)
        db.add_vectors(database)
        db.build()
        
        k = 10
        results = []
        
        for nprobe in nprobe_values:
            recalls = []
            for i, query in enumerate(queries[:200]):  # 使用 200 个查询
                search_results = db.search(query, k=k, nprobe=nprobe)
                recall = compute_recall_at_k(search_results.ids, ground_truth[i], k)
                recalls.append(recall)
            
            mean_recall = np.mean(recalls)
            results.append({
                "nprobe": nprobe,
                "recall@10": mean_recall,
                "nprobe_ratio": nprobe / nlist
            })
            print(f"nprobe={nprobe} ({nprobe/nlist*100:.0f}%): Recall@10={mean_recall:.4f}")
        
        return results
    
    def test_recall_vs_nlist(self, test_data):
        """测试: nlist 对召回率的影响"""
        database, queries, ground_truth = test_data
        
        nlist_values = [10, 50, 100, 200, 500]
        k = 10
        
        results = []
        for nlist in nlist_values:
            nprobe = max(1, nlist // 10)  # 搜索 10% 的簇
            
            config = VectorDBConfig(
                ivf=IVFIndexConfig(nlist=nlist, nprobe=nprobe)
            )
            db = VectorDB(config=config, use_memory_storage=True)
            db.add_vectors(database)
            db.build()
            
            recalls = []
            for i, query in enumerate(queries[:200]):
                search_results = db.search(query, k=k)
                recall = compute_recall_at_k(search_results.ids, ground_truth[i], k)
                recalls.append(recall)
            
            mean_recall = np.mean(recalls)
            results.append({
                "nlist": nlist,
                "nprobe": nprobe,
                "recall@10": mean_recall
            })
            print(f"nlist={nlist}, nprobe={nprobe}: Recall@10={mean_recall:.4f}")
        
        return results
    
    def test_recall_with_quantization(self, test_data):
        """测试: 量化对召回率的影响"""
        database, queries, ground_truth = test_data
        
        k = 10
        
        # 不使用量化
        db_no_q = VectorDB(use_memory_storage=True)
        db_no_q.add_vectors(database.copy())
        db_no_q.build(use_quantization=False)
        
        recalls_no_q = []
        for i, query in enumerate(queries[:200]):
            results = db_no_q.search(query, k=k)
            recall = compute_recall_at_k(results.ids, ground_truth[i], k)
            recalls_no_q.append(recall)
        
        # 使用量化
        db_with_q = VectorDB(use_memory_storage=True)
        db_with_q.add_vectors(database.copy())
        db_with_q.build(use_quantization=True)
        
        recalls_with_q = []
        for i, query in enumerate(queries[:200]):
            results = db_with_q.search(query, k=k)
            recall = compute_recall_at_k(results.ids, ground_truth[i], k)
            recalls_with_q.append(recall)
        
        print(f"\nWithout quantization: Recall@10 = {np.mean(recalls_no_q):.4f}")
        print(f"With quantization: Recall@10 = {np.mean(recalls_with_q):.4f}")
        print(f"Recall drop: {(np.mean(recalls_no_q) - np.mean(recalls_with_q)) * 100:.2f}%")
```

---

## 📈 性能基准目标

### 索引构建

| 数据集 | 目标时间 | 可接受时间 |
|--------|----------|------------|
| 10K vectors | < 1s | < 3s |
| 100K vectors | < 10s | < 30s |
| 1M vectors | < 120s | < 300s |

### 搜索延迟

| 数据集 | 目标 P50 | 目标 P99 |
|--------|----------|----------|
| 100K vectors | < 1ms | < 5ms |
| 1M vectors | < 5ms | < 20ms |

### 吞吐量

| 数据集 | 目标 QPS |
|--------|----------|
| 100K vectors | > 1000 |
| 1M vectors | > 500 |

### 召回率

| nprobe (% of nlist) | 目标 Recall@10 |
|---------------------|----------------|
| 1% | > 0.50 |
| 10% | > 0.90 |
| 50% | > 0.98 |
| 100% | 1.00 |

---

## 🔧 运行测试

### 快速测试

```bash
# 运行所有基准测试
pytest tests/benchmarks/ -v

# 运行特定测试
pytest tests/benchmarks/test_recall.py -v

# 使用 pytest-benchmark
pytest tests/benchmarks/ --benchmark-only
```

### 完整基准测试

```bash
# 生成测试数据集
python tests/datasets/generate_datasets.py

# 运行完整基准测试 (可能需要较长时间)
pytest tests/benchmarks/ -v --benchmark-save=baseline

# 对比两次测试结果
pytest-benchmark compare baseline current
```

### 性能报告生成

```python
# tests/benchmarks/generate_report.py
import json
import matplotlib.pyplot as plt
import pandas as pd

def generate_performance_report(results: dict, output_dir: str = "./tests/results"):
    """生成性能测试报告"""
    
    # 1. 召回率 vs nprobe 图
    if "recall_vs_nprobe" in results:
        df = pd.DataFrame(results["recall_vs_nprobe"])
        plt.figure(figsize=(10, 6))
        plt.plot(df["nprobe"], df["recall@10"], marker="o")
        plt.xlabel("nprobe")
        plt.ylabel("Recall@10")
        plt.title("Recall vs nprobe")
        plt.grid(True)
        plt.savefig(f"{output_dir}/recall_vs_nprobe.png")
        plt.close()
    
    # 2. 延迟分布
    if "latency" in results:
        latencies = results["latency"]
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=50, edgecolor="black")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.title("Search Latency Distribution")
        plt.savefig(f"{output_dir}/latency_distribution.png")
        plt.close()
    
    # 3. 生成 Markdown 报告
    with open(f"{output_dir}/benchmark_report.md", "w") as f:
        f.write("# VectorDB Performance Benchmark Report\n\n")
        f.write(f"## Summary\n\n")
        f.write(json.dumps(results, indent=2))
```

---

## 📋 测试检查清单

- [ ] 索引构建性能测试
  - [ ] 数据集大小 vs 构建时间
  - [ ] 维度 vs 构建时间
  - [ ] nlist vs 构建时间
  - [ ] 量化开销测试

- [ ] 搜索性能测试
  - [ ] 单查询延迟 (P50, P95, P99)
  - [ ] 吞吐量 (QPS)
  - [ ] nprobe vs 延迟
  - [ ] k vs 延迟
  - [ ] 批量 vs 逐个搜索

- [ ] 内存测试
  - [ ] 索引内存占用
  - [ ] 量化内存节省
  - [ ] 内存泄漏检测

- [ ] 召回率测试
  - [ ] Recall@1, @10, @100
  - [ ] nprobe vs 召回率
  - [ ] nlist vs 召回率
  - [ ] 量化对召回率影响

- [ ] 边界情况测试
  - [ ] 空数据库
  - [ ] 单向量
  - [ ] 重复向量
  - [ ] 高维向量 (768+)

---

## 🎯 Next Steps

1. **实现测试脚本**: 将上述测试用例实现到 `tests/benchmarks/` 目录
2. **生成基线数据**: 运行一次完整测试作为基线
3. **设置 CI/CD**: 在 PR 中自动运行性能回归测试
4. **持续监控**: 跟踪性能指标随版本变化

---

*Performance Testing Plan v1.0*
