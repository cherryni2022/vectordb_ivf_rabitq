# VectorDB IVF + RaBitQ 优化路线图

本文档详细描述了 IVF 索引和 RaBitQ 量化实现的分阶段优化计划。

---

## 优化阶段概览

| 阶段 | 名称 | 目标 | 预期收益 |
|------|------|------|----------|
| Phase 1 | 纯 NumPy 优化 | 代码重构与算法改进 | 2-5x 加速 |
| Phase 2 | JIT 编译加速 | 引入 Numba 加速热点 | 5-10x 加速 |
| Phase 3 | GPU 加速 | 核心计算迁移至 GPU | 10-100x 加速 |
| Phase 4 | 系统级优化 | 内存管理与并行化 | 整体性能提升 |

---

## Phase 1: 纯 NumPy 优化 (1-2 周)

### 1.1 K-Means Centroid 更新向量化

**目标**: 消除 Python 循环，使用纯向量化操作

**当前问题** (`ivf_index.py` Line 155-162):
```python
for i in range(n_clusters):
    mask = assignments == i
    if np.any(mask):
        new_centroids[i] = np.mean(vectors[mask], axis=0)
```

**优化实现**:
```python
def _vectorized_centroid_update(self, vectors: np.ndarray, 
                                 assignments: np.ndarray, 
                                 n_clusters: int) -> np.ndarray:
    """
    向量化的 centroid 更新
    使用 np.bincount 和累加技巧避免循环
    """
    dimension = vectors.shape[1]
    
    # 统计每个簇的向量数量
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.float32)
    counts = np.maximum(counts, 1)  # 避免除零
    
    # 累加每个簇的向量和
    # 使用 advanced indexing 实现向量化累加
    new_centroids = np.zeros((n_clusters, dimension), dtype=np.float32)
    np.add.at(new_centroids, assignments, vectors)
    
    # 计算均值
    new_centroids /= counts[:, np.newaxis]
    
    return new_centroids
```

**编码步骤**:
1. 在 `IVFIndex` 类中添加 `_vectorized_centroid_update` 方法
2. 修改 `_kmeans_clustering` 调用新方法
3. 添加单元测试验证正确性
4. 运行基准测试对比性能

---

### 1.2 静态 Popcount 查找表

**目标**: 避免每次调用重新创建查找表

**当前问题** (`true_rabitq.py` Line 214-216):
```python
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
```

**优化实现**:
```python
class TrueRaBitQ:
    # 类级别静态查找表
    _POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    
    def compute_hamming_distance(self, query_code: np.ndarray, 
                                  database_codes: np.ndarray) -> np.ndarray:
        xor_result = np.bitwise_xor(database_codes, query_code)
        return np.sum(self._POPCOUNT_TABLE[xor_result], axis=1)
```

**编码步骤**:
1. 在 `TrueRaBitQ` 类定义顶部添加类变量 `_POPCOUNT_TABLE`
2. 修改 `compute_hamming_distance` 使用类变量
3. 验证功能正确性

---

### 1.3 分块距离计算

**目标**: 支持大规模数据的内存高效距离计算

**优化实现**:
```python
def _compute_distances_chunked(self, queries: np.ndarray, 
                                points: np.ndarray, 
                                chunk_size: int = 10000) -> np.ndarray:
    """
    分块计算距离矩阵，避免内存溢出
    """
    n_queries = queries.shape[0]
    n_points = points.shape[0]
    distances = np.empty((n_queries, n_points), dtype=np.float32)
    
    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        chunk = points[i:end]
        distances[:, i:end] = self._compute_distances_matrix(queries, chunk)
    
    return distances
```

**编码步骤**:
1. 添加 `_compute_distances_chunked` 方法
2. 在 `build` 和 `_kmeans_clustering` 中根据数据规模选择使用
3. 添加 `chunk_size` 配置参数

---

### 1.4 空簇处理优化

**目标**: 更智能的空簇重新初始化策略

**优化实现**:
```python
def _handle_empty_clusters(self, vectors: np.ndarray,
                           assignments: np.ndarray,
                           centroids: np.ndarray,
                           empty_clusters: np.ndarray) -> np.ndarray:
    """
    智能处理空簇：选择距离其所属 centroid 最远的点
    """
    for cluster_idx in empty_clusters:
        # 找到最大簇
        cluster_sizes = np.bincount(assignments, minlength=len(centroids))
        largest_cluster = np.argmax(cluster_sizes)
        
        # 找到最大簇中距离 centroid 最远的点
        mask = assignments == largest_cluster
        cluster_vectors = vectors[mask]
        cluster_centroid = centroids[largest_cluster]
        
        distances = np.sum((cluster_vectors - cluster_centroid) ** 2, axis=1)
        farthest_local_idx = np.argmax(distances)
        farthest_global_idx = np.where(mask)[0][farthest_local_idx]
        
        # 将最远点作为新 centroid
        centroids[cluster_idx] = vectors[farthest_global_idx]
        assignments[farthest_global_idx] = cluster_idx
    
    return centroids
```

**编码步骤**:
1. 添加 `_handle_empty_clusters` 方法
2. 在 `_kmeans_clustering` 迭代中检测并处理空簇
3. 添加测试用例验证边界情况

---

### 1.5 Mini-Batch K-Means

**目标**: 支持超大规模数据的高效聚类

**优化实现**:
```python
def _minibatch_kmeans(self, vectors: np.ndarray, 
                       n_clusters: int,
                       batch_size: int = 1024,
                       max_iter: int = 100,
                       tol: float = 1e-4) -> np.ndarray:
    """
    Mini-Batch K-Means 实现
    每次迭代只使用部分样本更新 centroids
    """
    n_vectors = vectors.shape[0]
    centroids = self._kmeans_plusplus_init(vectors, n_clusters)
    
    for iteration in range(max_iter):
        # 随机采样 mini-batch
        batch_indices = np.random.choice(n_vectors, 
                                         min(batch_size, n_vectors), 
                                         replace=False)
        batch = vectors[batch_indices]
        
        # 分配到最近 centroid
        distances = self._compute_distances_matrix(batch, centroids)
        assignments = np.argmin(distances, axis=1)
        
        # 增量更新 centroids (学习率衰减)
        learning_rate = 1.0 / (iteration + 1)
        for i in range(n_clusters):
            mask = assignments == i
            if np.any(mask):
                delta = np.mean(batch[mask], axis=0) - centroids[i]
                centroids[i] += learning_rate * delta
    
    return centroids
```

**编码步骤**:
1. 添加 `_minibatch_kmeans` 方法
2. 添加 `use_minibatch` 和 `batch_size` 初始化参数
3. 在 `build` 中根据数据规模自动选择算法
4. 添加性能基准测试

---

## Phase 2: JIT 编译加速 (1-2 周)

### 2.1 Numba 加速 Hamming 距离

**目标**: 使用 JIT 编译加速位运算

**优化实现**:
```python
import numba
from numba import njit, prange

# 预编译 popcount 表
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

@njit(parallel=True, cache=True)
def fast_hamming_distance(query_code: np.ndarray, 
                          database_codes: np.ndarray,
                          popcount_table: np.ndarray) -> np.ndarray:
    """
    Numba 加速的 Hamming 距离计算
    """
    n_vectors = database_codes.shape[0]
    n_bytes = query_code.shape[0]
    result = np.zeros(n_vectors, dtype=np.int32)
    
    for i in prange(n_vectors):
        count = 0
        for j in range(n_bytes):
            xor_val = database_codes[i, j] ^ query_code[j]
            count += popcount_table[xor_val]
        result[i] = count
    
    return result
```

**编码步骤**:
1. 创建 `vectordb/utils/numba_utils.py`
2. 实现 `fast_hamming_distance` 函数
3. 在 `TrueRaBitQ` 中条件导入并使用
4. 添加 fallback 机制（无 Numba 时使用 NumPy 版本）

---

### 2.2 Numba 加速距离计算

**优化实现**:
```python
@njit(parallel=True, cache=True, fastmath=True)
def fast_l2_distances(queries: np.ndarray, 
                      points: np.ndarray) -> np.ndarray:
    """
    Numba 加速的 L2 距离批量计算
    """
    n_queries = queries.shape[0]
    n_points = points.shape[0]
    dimension = queries.shape[1]
    
    result = np.zeros((n_queries, n_points), dtype=np.float32)
    
    for i in prange(n_queries):
        for j in range(n_points):
            dist = 0.0
            for k in range(dimension):
                diff = queries[i, k] - points[j, k]
                dist += diff * diff
            result[i, j] = dist
    
    return result
```

**编码步骤**:
1. 在 `numba_utils.py` 添加 `fast_l2_distances`
2. 在 `IVFIndex._compute_distances_matrix` 中集成
3. 基准测试对比 NumPy BLAS 性能

---

### 2.3 Numba 加速 K-Means 迭代

**优化实现**:
```python
@njit(parallel=True, cache=True)
def fast_kmeans_assignment(vectors: np.ndarray,
                           centroids: np.ndarray) -> np.ndarray:
    """
    Numba 加速的 K-Means 分配步骤
    """
    n_vectors = vectors.shape[0]
    assignments = np.empty(n_vectors, dtype=np.int32)
    
    for i in prange(n_vectors):
        min_dist = np.inf
        min_idx = 0
        for j in range(centroids.shape[0]):
            dist = 0.0
            for k in range(vectors.shape[1]):
                diff = vectors[i, k] - centroids[j, k]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        assignments[i] = min_idx
    
    return assignments
```

**编码步骤**:
1. 添加 `fast_kmeans_assignment` 和 `fast_centroid_update`
2. 创建 Numba 版本的完整 K-Means 实现
3. 在 `IVFIndex` 中添加 `use_jit` 选项

---

## Phase 3: GPU 加速 (2-3 周)

### 3.1 PyTorch GPU 后端

**目标**: 使用 PyTorch 实现 GPU 加速的核心操作

**优化实现**:
```python
# vectordb/backends/pytorch_backend.py

import torch
from typing import Optional

class PyTorchBackend:
    """PyTorch GPU/CPU 计算后端"""
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.is_gpu = self.device.type == 'cuda'
    
    def compute_distances(self, queries: torch.Tensor, 
                          points: torch.Tensor,
                          metric: str = 'l2') -> torch.Tensor:
        """GPU 加速的距离计算"""
        if metric == 'l2':
            return torch.cdist(queries, points, p=2).pow(2)
        elif metric == 'ip':
            return -torch.mm(queries, points.t())
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def kmeans_iteration(self, vectors: torch.Tensor,
                         centroids: torch.Tensor) -> tuple:
        """单次 K-Means 迭代"""
        # 计算距离并分配
        distances = self.compute_distances(vectors, centroids)
        assignments = torch.argmin(distances, dim=1)
        
        # 更新 centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(len(centroids), device=self.device)
        
        # 使用 scatter_add 进行向量化累加
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, vectors.shape[1]), vectors)
        counts.scatter_add_(0, assignments, torch.ones(len(vectors), device=self.device))
        
        counts = counts.clamp(min=1)
        new_centroids /= counts.unsqueeze(1)
        
        return new_centroids, assignments
    
    def to_numpy(self, tensor: torch.Tensor) -> 'np.ndarray':
        return tensor.cpu().numpy()
    
    def from_numpy(self, array: 'np.ndarray') -> torch.Tensor:
        return torch.from_numpy(array).to(self.device)
```

**编码步骤**:
1. 创建 `vectordb/backends/` 目录结构
2. 实现 `pytorch_backend.py`
3. 创建后端抽象接口 `base_backend.py`
4. 添加 NumPy 后端作为默认实现

---

### 3.2 GPU 加速的 K-Means

**优化实现**:
```python
class GPUKMeans:
    """GPU 加速的 K-Means 实现"""
    
    def __init__(self, n_clusters: int, max_iter: int = 100,
                 tol: float = 1e-4, device: str = 'cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.backend = PyTorchBackend(device)
    
    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """训练 K-Means"""
        vectors_gpu = self.backend.from_numpy(vectors.astype(np.float32))
        
        # K-Means++ 初始化 (在 GPU 上)
        centroids = self._kmeans_plusplus_gpu(vectors_gpu)
        
        for iteration in range(self.max_iter):
            new_centroids, assignments = self.backend.kmeans_iteration(
                vectors_gpu, centroids
            )
            
            # 检查收敛
            shift = torch.max(torch.norm(new_centroids - centroids, dim=1))
            if shift < self.tol:
                break
            
            centroids = new_centroids
        
        return self.backend.to_numpy(centroids)
```

**编码步骤**:
1. 创建 `vectordb/clustering/gpu_kmeans.py`
2. 实现 GPU 版本的 K-Means++ 初始化
3. 在 `IVFIndex` 中添加 GPU 后端选择逻辑
4. 添加 CUDA 可用性检测和 fallback

---

### 3.3 GPU 加速的 RaBitQ

**优化实现**:
```python
class GPURaBitQ:
    """GPU 加速的 RaBitQ 实现"""
    
    def __init__(self, dimension: int, device: str = 'cuda'):
        self.dimension = dimension
        self.device = torch.device(device)
        self.rotation_matrix: Optional[torch.Tensor] = None
    
    def encode_gpu(self, vectors: torch.Tensor) -> tuple:
        """GPU 上的批量编码"""
        # 计算 norms
        norms = torch.norm(vectors, dim=1)
        
        # 归一化
        normalized = vectors / (norms.unsqueeze(1) + 1e-8)
        
        # 旋转
        rotated = torch.mm(normalized, self.rotation_matrix)
        
        # 符号量化
        signs = (rotated > 0).to(torch.uint8)
        
        # Pack bits (在 GPU 上)
        # 使用自定义 CUDA kernel 或分块处理
        binary_codes = self._pack_bits_gpu(signs)
        
        return binary_codes, norms
    
    def compute_hamming_gpu(self, query_code: torch.Tensor,
                            database_codes: torch.Tensor) -> torch.Tensor:
        """GPU 加速的 Hamming 距离"""
        # XOR 操作
        xor_result = query_code.unsqueeze(0) ^ database_codes
        
        # Popcount (使用位操作技巧)
        # 对于 uint8，展开为 bits 再 sum
        bits = xor_result.unsqueeze(-1).bitwise_and(
            torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=self.device)
        ).ne(0).sum(dim=-1).sum(dim=-1)
        
        return bits
```

**编码步骤**:
1. 创建 `vectordb/quantization/gpu_rabitq.py`
2. 实现 GPU 编码和 Hamming 距离
3. 添加 CPU-GPU 数据传输优化（异步传输）
4. 集成到 `RaBitQWithRerank` 流程

---

### 3.4 CuPy 备选方案

**优化实现**:
```python
# vectordb/backends/cupy_backend.py

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class CuPyBackend:
    """CuPy GPU 计算后端 (更接近 NumPy API)"""
    
    def compute_distances(self, queries: cp.ndarray, 
                          points: cp.ndarray) -> cp.ndarray:
        """L2 距离计算"""
        q_norm = cp.sum(queries ** 2, axis=1, keepdims=True)
        p_norm = cp.sum(points ** 2, axis=1, keepdims=True).T
        dot = cp.dot(queries, points.T)
        return q_norm + p_norm - 2 * dot
    
    def hamming_distance(self, query_code: cp.ndarray,
                         database_codes: cp.ndarray) -> cp.ndarray:
        """Hamming 距离 (使用 CUDA popcount)"""
        xor_result = cp.bitwise_xor(database_codes, query_code)
        
        # 使用 RawKernel 调用 CUDA popcount
        kernel = cp.RawKernel(r'''
        extern "C" __global__
        void popcount_sum(const unsigned char* xor_data, 
                          int* result, int n_bytes) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int sum = 0;
            for (int i = 0; i < n_bytes; i++) {
                sum += __popc(xor_data[idx * n_bytes + i]);
            }
            result[idx] = sum;
        }
        ''', 'popcount_sum')
        
        # 执行 kernel
        ...
```

**编码步骤**:
1. 创建 `cupy_backend.py`
2. 实现核心距离和位运算
3. 添加自定义 CUDA kernel 优化 popcount
4. 创建后端统一接口

---

## Phase 4: 系统级优化 (1-2 周)

### 4.1 内存映射支持

**目标**: 支持超大规模数据集

**优化实现**:
```python
class MemoryMappedVectorStore:
    """内存映射向量存储"""
    
    def __init__(self, path: str, dimension: int, dtype=np.float32):
        self.path = path
        self.dimension = dimension
        self.dtype = dtype
        self.mmap: Optional[np.memmap] = None
    
    def create(self, n_vectors: int) -> np.memmap:
        """创建内存映射文件"""
        self.mmap = np.memmap(
            self.path, 
            dtype=self.dtype,
            mode='w+',
            shape=(n_vectors, self.dimension)
        )
        return self.mmap
    
    def load(self, mode: str = 'r') -> np.memmap:
        """加载已存在的映射"""
        self.mmap = np.memmap(self.path, dtype=self.dtype, mode=mode)
        n_vectors = len(self.mmap) // self.dimension
        self.mmap = self.mmap.reshape((n_vectors, self.dimension))
        return self.mmap
    
    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        """按需加载批量向量"""
        return np.array(self.mmap[indices])
```

**编码步骤**:
1. 创建 `vectordb/storage/mmap_store.py`
2. 修改 `IVFIndex` 支持 mmap 向量存储
3. 实现按需加载和预取机制
4. 添加缓存层提升热点数据访问

---

### 4.2 多线程批量搜索

**优化实现**:
```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ParallelSearcher:
    """多线程批量搜索"""
    
    def __init__(self, index: 'IVFIndex', n_threads: int = 4):
        self.index = index
        self.n_threads = n_threads
        self.executor = ThreadPoolExecutor(max_workers=n_threads)
    
    def search_batch(self, queries: np.ndarray, k: int) -> list:
        """并行批量搜索"""
        futures = [
            self.executor.submit(self.index.search, q, k)
            for q in queries
        ]
        return [f.result() for f in futures]
    
    def search_batch_chunked(self, queries: np.ndarray, 
                             k: int, chunk_size: int = 100) -> list:
        """分块并行搜索 (减少线程开销)"""
        n_queries = len(queries)
        results = [None] * n_queries
        
        def process_chunk(start: int, end: int):
            for i in range(start, end):
                results[i] = self.index.search(queries[i], k)
        
        chunk_starts = range(0, n_queries, chunk_size)
        futures = [
            self.executor.submit(process_chunk, s, min(s + chunk_size, n_queries))
            for s in chunk_starts
        ]
        
        for f in futures:
            f.result()
        
        return results
```

**编码步骤**:
1. 创建 `vectordb/search/parallel_searcher.py`
2. 实现线程池管理和任务分发
3. 添加 GIL 优化（使用 Numba `nogil` 或 C 扩展）
4. 基准测试不同并发级别

---

### 4.3 搜索缓存层

**优化实现**:
```python
from functools import lru_cache
import hashlib

class SearchCache:
    """查询结果缓存"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _hash_query(self, query: np.ndarray, k: int, nprobe: int) -> str:
        """计算查询哈希"""
        query_bytes = query.tobytes()
        key = hashlib.md5(query_bytes + f"{k}_{nprobe}".encode()).hexdigest()
        return key
    
    def get(self, query: np.ndarray, k: int, nprobe: int):
        """获取缓存结果"""
        key = self._hash_query(query, k, nprobe)
        if key in self.cache:
            return self.cache[key]
        return None
    
    def put(self, query: np.ndarray, k: int, nprobe: int, result):
        """存储结果"""
        if len(self.cache) >= self.max_size:
            # LRU 淘汰
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        key = self._hash_query(query, k, nprobe)
        self.cache[key] = result
        self.access_order.append(key)
```

**编码步骤**:
1. 创建缓存模块
2. 在 `IVFIndex.search` 中集成缓存逻辑
3. 添加缓存统计和监控
4. 支持缓存失效策略

---

### 4.4 动态 nprobe 调整

**优化实现**:
```python
class AdaptiveNProbe:
    """自适应 nprobe 调整"""
    
    def __init__(self, index: 'IVFIndex', target_recall: float = 0.95):
        self.index = index
        self.target_recall = target_recall
        self.history = []
    
    def estimate_nprobe(self, query: np.ndarray) -> int:
        """根据查询特征估计最优 nprobe"""
        # 计算查询到各 centroid 的距离分布
        distances = self.index._compute_distances_matrix(
            query.reshape(1, -1), 
            self.index.centroids
        )[0]
        
        sorted_dists = np.sort(distances)
        
        # 根据距离分布差异调整 nprobe
        # 如果前几个 cluster 距离接近，需要更大 nprobe
        dist_ratio = sorted_dists[1] / (sorted_dists[0] + 1e-8)
        
        if dist_ratio < 1.1:  # 距离非常接近
            return min(self.index.nprobe * 2, self.index.nlist)
        elif dist_ratio > 2.0:  # 距离差异明显
            return max(self.index.nprobe // 2, 1)
        else:
            return self.index.nprobe
```

**编码步骤**:
1. 创建自适应搜索模块
2. 收集查询统计信息
3. 实现基于统计的 nprobe 预测
4. 添加在线学习调整机制

---

## 依赖与环境配置

### 基础依赖
```bash
# requirements.txt 更新
numpy>=1.21.0
scipy>=1.7.0
```

### JIT 依赖 (Phase 2)
```bash
numba>=0.56.0
```

### GPU 依赖 (Phase 3)
```bash
# PyTorch GPU
torch>=2.0.0  # with CUDA

# 或 CuPy
cupy-cuda11x>=11.0.0  # 根据 CUDA 版本选择
```

---

## 测试与验证计划

### 每阶段验证
1. **功能测试**: 验证优化后结果与原始实现一致
2. **性能基准**: 对比优化前后的 QPS 和延迟
3. **Recall 测试**: 确保 ANN 精度不降低

### 基准测试维度
- 数据规模: 10K, 100K, 1M, 10M 向量
- 维度: 128, 256, 512, 768, 1536
- 批量大小: 1, 10, 100, 1000 查询

---

## 参考资源

1. [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
2. [RaBitQ Paper (SIGMOD'24)](https://dl.acm.org/doi/10.1145/3626734)
3. [Numba Documentation](https://numba.readthedocs.io/)
4. [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
