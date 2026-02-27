# 详细技术方案设计: 向量数据库 (Phase 1)

本文档将针对 PRD 中提出的几个关键模块（Schema 管理、数据组织与存储引擎、索引与原数据映射、并发控制、MVCC 与版本、gRPC API 设计）分别给出**可落地、可编码**的实现方案。整个单机服务端使用 Python 作为入口，核心密集型索引依赖现有的 C++/Numpy 绑定库（HNSWLib、Faiss 等）。

---

## 目录
1. [Schema 体系与管理设计](#1-schema-体系与管理设计)
2. [数据组织与存储引擎](#2-数据组织与存储引擎-storage-engine)
3. [双索引策略与实现](#3-双索引策略与实现)
4. [数据序列化与反序列化](#4-数据序列化与反序列化)
5. [gRPC API 协议定义](#5-grpc-api-协议定义)
6. [并发控制与锁设计](#6-并发控制与锁设计)
7. [MVCC 多版本控制与查询快照](#7-mvcc-多版本控制与查询快照-snapshot)

---

## 1. Schema 体系与管理设计

### 1.1 核心数据结构定义

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class FieldType(Enum):
    INT = "int"        # int64
    STRING = "string"  # 变长 UTF-8 字符串
    VECTOR = "vector"  # float32 定长向量

class MetricType(Enum):
    L2 = "L2"
    IP = "IP"          # Inner Product
    COSINE = "COSINE"

@dataclass
class FieldSchema:
    """单个字段的定义"""
    name: str
    dtype: FieldType
    is_primary: bool = False
    # 仅 VECTOR 类型字段有效
    dim: Optional[int] = None
    metric: Optional[MetricType] = None

@dataclass
class CollectionSchema:
    """Collection 的完整 Schema"""
    collection_name: str
    fields: List[FieldSchema]
    # 系统自动填充
    created_at: Optional[float] = None   # timestamp
    version: int = 0                     # schema 版本号

    def primary_field(self) -> FieldSchema:
        """返回主键字段"""
        for f in self.fields:
            if f.is_primary:
                return f
        raise ValueError("No primary field defined")

    def vector_field(self) -> FieldSchema:
        """返回第一个向量字段"""
        for f in self.fields:
            if f.dtype == FieldType.VECTOR:
                return f
        raise ValueError("No vector field defined")
```

**校验规则**:
| 规则 | 说明 |
|------|------|
| 必须有且仅有一个 `is_primary=True` 字段 | 主键类型仅允许 `INT` 或 `STRING` |
| 必须至少有一个 `VECTOR` 字段 | 且必须指定 `dim` 和 `metric` |
| 字段名不可重复 | 不允许 `_lsn`, `_deleted` 等系统保留名 |

### 1.2 元数据持久化
- **存储位置**: `{data_dir}/meta/{collection_name}/schema.json`
- **加载策略**: 服务启动时，扫描 `meta/` 下所有子目录，将 Schema 反序列化到内存 `Dict[str, CollectionSchema]`
- **修改策略**: Phase 1 不支持 Schema 变更（ALTER），如需修改只能 Drop + Recreate

---

## 2. 数据组织与存储引擎 (Storage Engine)

### 2.1 核心概念: Row 与 Segment

一个 Collection 的数据被组织为多个 **Segment（数据片段）**。每个 Segment 是一个独立的、原子的存储单元。

#### 2.1.1 Row（行）的内部表示

每一条用户 Insert 的数据，在引擎内部被表示为一个 `Row`：

```python
@dataclass
class Row:
    """引擎内部的一行数据表示"""
    pk: Union[int, str]                 # 主键值
    vector: np.ndarray                  # shape=(dim,), dtype=float32
    scalars: Dict[str, Union[int, str]] # 标量字段 {field_name: value}
    lsn: int = 0                        # 写入时分配的版本号 (系统隐藏列)
```

#### 2.1.2 Segment（数据片段）

每个 Segment 在内存和磁盘上管理一批 Row，核心在于将列式存储拆分为多个独立的物理文件/数组：

```python
@dataclass
class SegmentMeta:
    """Segment 元信息"""
    segment_id: str          # 唯一标识, 如 "seg_0001"
    collection_name: str
    state: SegmentState      # GROWING / SEALED / INDEXED
    row_count: int = 0
    max_rows: int = 100_000  # 阈值, 达到后 seal
    min_lsn: int = 0         # 此 Segment 内最小的 LSN
    max_lsn: int = 0         # 此 Segment 内最大的 LSN

class SegmentState(Enum):
    GROWING = "growing"      # 正在接收写入 (即 MemTable)
    SEALED = "sealed"        # 已封闭, 不再接收写入, 但尚未建索引
    INDEXED = "indexed"      # 已封闭且已构建向量索引
```

### 2.2 Segment 内部的物理数据布局

每个 Segment 内部将数据按**列(Column)**拆分存储。这样做的核心原因：
- 向量列是定长的 `float32` 矩阵，可以紧凑地用 `np.ndarray` 存储和 mmap
- 标量列（`int`/`string`）的存取模式独立于向量列
- 搜索时只需要读取向量列，不需要将标量列全部加载到内存

```text
Segment "seg_0001" 内部物理文件组织:
─────────────────────────────────────
{data_dir}/segments/seg_0001/
├── meta.json           ← SegmentMeta 序列化
├── vectors.npy         ← 向量列, shape=(row_count, dim), float32
├── pk.npy / pk.pkl     ← 主键列 (int 用 npy, string 用 pkl)
├── scalars.pkl         ← 其他标量列 Dict[str, List[Union[int,str]]]
├── lsn.npy             ← 隐藏列, shape=(row_count,), int64
├── delete_bitset.bin   ← 删除位图, 每个 bit 表示一行
└── index/              ← 索引文件 (仅 INDEXED 状态)
    ├── hnsw.bin         ← hnswlib 序列化文件
    └── index_meta.json  ← 索引类型/参数
```

#### 2.2.1 各列在内存中的数据结构

```python
class SegmentData:
    """Segment 在内存中的完整数据表示"""
    def __init__(self, meta: SegmentMeta, schema: CollectionSchema):
        self.meta = meta
        self.schema = schema
        dim = schema.vector_field().dim

        # ====== 列式存储 ======
        # 向量列: 预分配定长 numpy 矩阵
        # 为什么预分配? 避免每次 insert 做 np.vstack 拷贝
        self.vectors: np.ndarray = np.zeros(
            (meta.max_rows, dim), dtype=np.float32
        )
        # 主键列
        self.pks: List[Union[int, str]] = []
        # 标量列: 按字段名分开存储
        self.scalar_columns: Dict[str, List[Union[int, str]]] = {
            f.name: []
            for f in schema.fields
            if f.dtype != FieldType.VECTOR and not f.is_primary
        }
        # 隐藏列: 每行的写入版本号
        self.lsns: np.ndarray = np.zeros(meta.max_rows, dtype=np.int64)
        # 删除位图: 第 i 位为 1 表示第 i 行已被软删除
        self.delete_bitset: bytearray = bytearray(
            (meta.max_rows + 7) // 8
        )
        # 当前写入游标
        self._write_cursor: int = 0

    def append_row(self, row: Row) -> int:
        """追加一行, 返回该行在 Segment 内的偏移量 (offset)"""
        offset = self._write_cursor
        self.vectors[offset] = row.vector
        self.pks.append(row.pk)
        for fname, val in row.scalars.items():
            self.scalar_columns[fname].append(val)
        self.lsns[offset] = row.lsn
        self._write_cursor += 1
        self.meta.row_count = self._write_cursor
        return offset

    def mark_deleted(self, offset: int):
        """在位图中标记第 offset 行为已删除"""
        byte_idx = offset >> 3        # offset // 8
        bit_idx = offset & 0x07       # offset % 8
        self.delete_bitset[byte_idx] |= (1 << bit_idx)

    def is_deleted(self, offset: int) -> bool:
        byte_idx = offset >> 3
        bit_idx = offset & 0x07
        return bool(self.delete_bitset[byte_idx] & (1 << bit_idx))

    def is_full(self) -> bool:
        return self._write_cursor >= self.meta.max_rows
```

#### 2.2.2 为什么向量列使用预分配而不是动态追加

| 方案 | 每次写入开销 | 内存占用 | 说明 |
|------|------------|---------|------|
| `np.vstack` 动态拼接 | O(N×dim) 拷贝 | 紧实 | 每次追加触发全量拷贝，10万条 128 维时追加一条需拷贝 ~48MB |
| **预分配固定矩阵** | O(dim) 赋值 | 预留 max_rows | 仅在 `vectors[cursor] = vec` 赋值，O(1) 级别追加 |

我们选择**预分配**。max_rows=100,000 且 dim=128 时，预分配仅占 `100000 × 128 × 4B = 48.8MB`，可控。

### 2.3 Collection 级别的 Segment 管理

```python
class CollectionEngine:
    """管理某个 Collection 下的所有 Segment"""
    def __init__(self, schema: CollectionSchema):
        self.schema = schema
        # 当前可写入的 Segment（GROWING 状态），只有一个
        self.active_segment: SegmentData = self._new_segment()
        # 已封闭的 Segment 列表
        self.sealed_segments: List[SegmentData] = []
        # 全局主键映射: PK -> (segment_id, offset)
        self.pk_map: Dict[Union[int, str], Tuple[str, int]] = {}

    def _new_segment(self) -> SegmentData:
        seg_id = f"seg_{uuid4().hex[:8]}"
        meta = SegmentMeta(
            segment_id=seg_id,
            collection_name=self.schema.collection_name,
            state=SegmentState.GROWING,
        )
        return SegmentData(meta, self.schema)

    def seal_active_segment(self):
        """将当前活跃 Segment 封闭, 创建新的活跃 Segment"""
        self.active_segment.meta.state = SegmentState.SEALED
        self.sealed_segments.append(self.active_segment)
        self.active_segment = self._new_segment()
```

---

## 3. 双索引策略与实现

系统支持两种可选的索引策略，Collection 创建索引时二选一：

```python
class IndexStrategy(Enum):
    """Collection 级别的索引策略"""
    HNSW_PER_SEGMENT = "hnsw"        # 每 Segment 独立 HNSWLib 索引
    IVF_RABITQ_GLOBAL = "ivf_rabitq"  # 全局 IVF + RaBitQ 索引
```

| 维度 | HNSWLib Per-Segment | IVF + RaBitQ 全局 |
|------|---------------------|-------------------|
| **适用场景** | 在线实时写入 + 低延迟查询 | 批量导入完成后，高吞吐高压缩比检索 |
| **构建时机** | Segment Sealed 后自动/手动触发 | 全量数据写入完成后显式调用 |
| **数据粒度** | 每 Segment 独立 | 跨所有 Segment 的全局数据 |
| **写入影响** | 不影响，Sealed Segment 不可变 | Build 期间可写入，新数据对全局索引不可见 |
| **压缩比** | 无量化 | RaBitQ 32x 压缩 |

### 3.1 核心问题: 索引内部 ID ↔ 原始数据 Offset 的映射

底层索引引擎（`hnswlib`、`IVFIndex`）在 `build()` 和 `search()` 时使用的是**内部连续整数 ID**（即 `0, 1, 2, ... N-1`）。但在 Segment 化的存储中，真正用于定位数据的是 `(segment_id, offset)`。两种索引策略的映射方式有所不同：

```text
======= HNSWLib Per-Segment =======
用户视角:  PK (1001, "abc", ...)
              ↓  pk_map 查询
引擎视角:  (segment_id="seg_0001", offset=42)
              ↓  直接下标访问
物理存储:  segment.vectors[42], segment.pks[42], ...
              ↓  建索引时
索引引擎:  hnswlib_internal_id = 42   (等于 segment 内 offset)

关键设计: 索引内部 ID 直接等于该 Segment 内的 offset

======= IVF+RaBitQ 全局 =======
用户视角:  PK (1001, "abc", ...)
              ↓  pk_map 查询
引擎视角:  (segment_id="seg_0001", offset=42)
              ↓  build 时拼接
全局矩阵:  all_vectors[global_id=12345]
              ↓  global_id_map 反查
映射回:    global_id_map[12345] = ("seg_0001", 42)

关键设计: 维护 global_id -> (segment_id, offset) 映射表
```

---

### 3.2 策略 A: HNSWLib Per-Segment 索引

#### 3.2.1 Segment 内的索引构建接口

```python
class SegmentIndex:
    """为单个 Sealed/Indexed Segment 管理 HNSWLib 向量索引"""
    def __init__(self, segment: SegmentData):
        self.segment = segment
        self.index: Optional[hnswlib.Index] = None
        self.index_type: str = "hnsw"
        self.index_params: Dict[str, Any] = {}

    def build_index(self, params: dict):
        """
        对 Segment 内的向量构建 HNSWLib 索引
        注意: segment.meta.state 必须是 SEALED
        """
        vectors = self.segment.vectors[:self.segment.meta.row_count]
        n = self.segment.meta.row_count
        dim = vectors.shape[1]

        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(
            max_elements=n,
            ef_construction=params.get('ef_construction', 200),
            M=params.get('M', 16),
        )
        # labels 直接使用 0..N-1, 即 segment offset
        self.index.add_items(vectors, ids=np.arange(n))
        self.index.set_ef(params.get('ef_search', 50))

        self.segment.meta.state = SegmentState.INDEXED
        self.index_params = params

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        返回 List[(offset_in_segment, distance)]
        调用方需要自行过滤 delete_bitset 和 LSN
        """
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)
        return list(zip(labels[0].tolist(), distances[0].tolist()))
```

#### 3.2.2 HNSWLib 搜索链路: Scatter-Gather 归并

一次 `Search()` 请求需要同时搜索**所有 Segment**（包括未建索引的 Active Segment），然后做全局归并：

```text
Client: Search(query_vec, top_k=10)
    │
    ├──▶ Active Segment (GROWING): brute-force 搜索
    │      vectors[:cursor] 做 np.dot / cdist, 返回 local top_k
    │
    ├──▶ Sealed Segment A (INDEXED): segment_index.search(query, top_k)
    │      HNSWLib knn_query, 返回 [(offset, dist), ...]
    │
    ├──▶ Sealed Segment B (INDEXED): segment_index.search(query, top_k)
    │      HNSWLib knn_query, 返回 [(offset, dist), ...]
    │
    ├──▶ Sealed Segment C (SEALED, 未建索引): brute-force
    │
    └─── Coordinator: 对所有 local top_k 做 heap merge
         → 过滤: is_deleted(offset)? lsn > read_lsn?
         → 输出 global top_k, 回填 pk 和 scalar 数据
```

---

### 3.3 策略 B: IVF + RaBitQ 全局索引

#### 3.3.1 设计思路

- **写入阶段**: 数据写入时只写入 Segment 原始数据（MemTable → Sealed Segment），不构建任何索引。
- **构建阶段**: 批量数据写入完成后，用户显式调用 `CreateIndex(type=ivf_rabitq)` 触发全局索引构建。构建过程将所有 Segment 的向量拼接为全局矩阵，统一训练 IVF k-means 聚类 + RaBitQ 二值量化编码。
- **搜索阶段**: 使用全局 IVF+RaBitQ 索引的两阶段检索（RaBitQ 粗筛 + 精确 Rerank）搜索已覆盖的数据；Build 后新写入的数据走暴力搜索兜底，最终归并 top_k。

#### 3.3.2 全局索引构建器

```python
from vectordb.index.ivf_index import IVFIndex
from vectordb.quantization.true_rabitq import TrueRaBitQ

class GlobalIVFIndex:
    """全局 IVF + RaBitQ 索引"""
    def __init__(self):
        self.ivf: Optional[IVFIndex] = None
        self.quantizer: Optional[TrueRaBitQ] = None
        # 关键映射: global_vector_id -> (segment_id, offset)
        self.global_id_map: List[Tuple[str, int]] = []
        # 记录本次 Build 覆盖了哪些 Segment
        self.included_segments: Set[str] = set()
        self.build_lsn: int = 0        # Build 时的最大 LSN
        self.is_built: bool = False


class GlobalIVFIndexBuilder:
    """全局 IVF + RaBitQ 索引构建器"""

    @staticmethod
    def build(
        segments: List[SegmentData],
        params: Dict[str, Any],
        current_lsn: int,
    ) -> GlobalIVFIndex:
        """
        拼接所有 Segment 的向量数据，构建全局 IVF 索引 + RaBitQ 量化。

        Args:
            segments: 所有需要纳入索引的 Segment (SEALED 或 GROWING 均可)
            params: 索引参数, 如 {"nlist": 100, "nprobe": 10, "rerank_factor": 10}
            current_lsn: 当前全局 LSN, 用于记录索引覆盖的数据快照
        """
        result = GlobalIVFIndex()

        # ====== Step 1: 拼接所有 Segment 的向量，跳过已删除行 ======
        all_vectors = []
        global_id_map = []

        for seg in segments:
            n = seg.meta.row_count
            seg_id = seg.meta.segment_id
            for offset in range(n):
                # 跳过已删除的行，减小索引体积
                if not seg.is_deleted(offset):
                    all_vectors.append(seg.vectors[offset])
                    global_id_map.append((seg_id, offset))
            result.included_segments.add(seg_id)

        if len(all_vectors) == 0:
            return result

        all_vectors_np = np.array(all_vectors, dtype=np.float32)
        total_vectors = len(all_vectors_np)
        dim = all_vectors_np.shape[1]

        # ====== Step 2: 构建 IVF 索引 (全局 k-means 聚类) ======
        nlist = min(
            params.get('nlist', 100),
            max(int(np.sqrt(total_vectors)), 1)
        )
        nprobe = params.get('nprobe', 10)

        ivf = IVFIndex(nlist=nlist, nprobe=nprobe, metric="l2")
        ivf.build(all_vectors_np)

        # ====== Step 3: 构建 RaBitQ 量化器 (全局训练 + 编码) ======
        quantizer = TrueRaBitQ(dimension=dim)
        quantizer.train(all_vectors_np)
        quantizer.encode(all_vectors_np)

        # ====== Step 4: 组装结果 ======
        result.ivf = ivf
        result.quantizer = quantizer
        result.global_id_map = global_id_map
        result.build_lsn = current_lsn
        result.is_built = True

        return result
```

#### 3.3.3 全局索引搜索

```python
@dataclass
class GlobalSearchHit:
    """全局搜索结果"""
    segment_id: str
    offset: int
    distance: float


class GlobalIVFSearcher:
    """使用全局 IVF+RaBitQ 索引执行搜索"""

    @staticmethod
    def search(
        global_index: GlobalIVFIndex,
        query: np.ndarray,
        k: int,
        segments_map: Dict[str, SegmentData],  # seg_id -> SegmentData
        read_lsn: int,
        rerank_factor: int = 10,
    ) -> List[GlobalSearchHit]:
        """
        使用全局 IVF+RaBitQ 索引搜索。

        流程:
        1. IVF 找到 nprobe 个最近聚类中心
        2. RaBitQ 粗筛候选向量 (近似距离)
        3. 取 top_k * rerank_factor 个候选
        4. 精确距离 Rerank
        5. global_id_map 映射回 (seg_id, offset)
        6. 过滤已删除 / 不可见的行
        """
        # 使用已有的 IVF + RaBitQ 搜索方法
        raw_results = global_index.ivf.search_with_rabitq(
            query, k * rerank_factor, global_index.quantizer,
            rerank_factor=rerank_factor
        )

        # 映射回 (segment_id, offset) 并过滤
        hits = []
        for global_id, distance in raw_results:
            seg_id, offset = global_index.global_id_map[global_id]
            seg = segments_map.get(seg_id)
            if seg is None:
                continue
            # 可见性检查: 未删除 + LSN <= read_lsn
            if seg.is_deleted(offset):
                continue
            if seg.lsns[offset] > read_lsn:
                continue
            hits.append(GlobalSearchHit(seg_id, offset, distance))
            if len(hits) >= k:
                break

        return hits
```

#### 3.3.4 IVF+RaBitQ 搜索链路

```text
Client: Search(query_vec, top_k=10)
    │
    ├──▶ GlobalIVFIndex.search(query, top_k)
    │      1. IVF: 找到 nprobe 个最近聚类中心
    │      2. RaBitQ: 粗筛候选向量 (Hamming 近似距离)
    │      3. 取 top_k * rerank_factor 个候选
    │      4. Exact Rerank: 计算精确距离
    │      5. global_id_map 映射回 (seg_id, offset)
    │      6. 过滤 delete_bitset + LSN
    │
    ├──▶ 未被全局索引覆盖的 Segments: brute-force
    │      (Active Segment + Build 后新增的 Sealed Segments)
    │      判断条件: seg_id NOT IN global_index.included_segments
    │
    └─── Coordinator: 归并所有结果
         → heap merge → 输出 global top_k
         → 回填 pk 和 scalar 数据
```

---

### 3.4 暴力搜索 (两种策略共用)

Active Segment 和未建索引的 Segment 均使用暴力搜索，两种索引策略共用此逻辑：

```python
def brute_force_search(
    segment: SegmentData,
    query: np.ndarray,  # shape=(dim,)
    k: int,
    read_lsn: int,
) -> List[Tuple[int, float]]:
    """暴力搜索, 返回 [(offset, distance), ...] 已按距离升序"""
    n = segment._write_cursor
    if n == 0:
        return []
    vectors = segment.vectors[:n]  # shape=(n, dim)
    # L2 距离: ||q - v||^2
    diffs = vectors - query.reshape(1, -1)  # broadcast
    distances = np.sum(diffs ** 2, axis=1)  # shape=(n,)

    # 过滤已删除和不可见的行
    valid_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if segment.is_deleted(i) or segment.lsns[i] > read_lsn:
            valid_mask[i] = False

    valid_indices = np.where(valid_mask)[0]
    valid_distances = distances[valid_indices]

    # 取 top_k
    if len(valid_indices) == 0:
        return []
    top_count = min(k, len(valid_indices))
    top_local = np.argpartition(valid_distances, top_count)[:top_count]
    top_local = top_local[np.argsort(valid_distances[top_local])]

    return [
        (int(valid_indices[i]), float(valid_distances[i]))
        for i in top_local
    ]
```

### 3.5 搜索协调器: 统一入口

```python
import heapq

class SearchCoordinator:
    """根据索引策略分发搜索请求并归并结果"""

    def __init__(self, engine: CollectionEngine):
        self.engine = engine

    def search(
        self,
        query: np.ndarray,
        k: int,
        snap: ReadSnapshot,
        index_strategy: IndexStrategy,
    ) -> List[dict]:
        """
        统一搜索入口, 根据索引策略路由到不同的搜索链路。
        返回 List[{"pk": ..., "distance": ..., "scalars": {...}}]
        """
        all_candidates = []  # List[(distance, seg_id, offset)]

        if index_strategy == IndexStrategy.HNSW_PER_SEGMENT:
            all_candidates = self._search_hnsw(query, k, snap)
        elif index_strategy == IndexStrategy.IVF_RABITQ_GLOBAL:
            all_candidates = self._search_ivf_rabitq(query, k, snap)

        # 归并 top_k (按距离升序)
        top_k = heapq.nsmallest(k, all_candidates, key=lambda x: x[0])

        # 回填原始数据
        return self._assemble_results(top_k)

    def _search_hnsw(self, query, k, snap):
        """HNSWLib Per-Segment 搜索路径"""
        candidates = []

        # 1. Active Segment 暴力搜索
        bf_results = brute_force_search(
            self.engine.active_segment, query, k, snap.read_lsn
        )
        seg_id = self.engine.active_segment.meta.segment_id
        for offset, dist in bf_results:
            candidates.append((dist, seg_id, offset))

        # 2. 各 Sealed/Indexed Segment
        for seg in self.engine.sealed_segments:
            if seg.meta.state == SegmentState.INDEXED and hasattr(seg, '_index'):
                results = seg._index.search(query, k)
                for offset, dist in results:
                    if is_visible(seg, offset, snap):
                        candidates.append((dist, seg.meta.segment_id, offset))
            else:
                # SEALED 但未建索引, 暴力搜索
                bf_results = brute_force_search(seg, query, k, snap.read_lsn)
                for offset, dist in bf_results:
                    candidates.append((dist, seg.meta.segment_id, offset))

        return candidates

    def _search_ivf_rabitq(self, query, k, snap):
        """IVF+RaBitQ 全局索引搜索路径"""
        candidates = []
        global_index = self.engine.global_ivf_index  # GlobalIVFIndex 实例

        # 1. 全局 IVF+RaBitQ 搜索 (覆盖 included_segments 中的数据)
        if global_index is not None and global_index.is_built:
            segments_map = {
                seg.meta.segment_id: seg
                for seg in self.engine.sealed_segments
            }
            hits = GlobalIVFSearcher.search(
                global_index, query, k, segments_map, snap.read_lsn
            )
            for hit in hits:
                candidates.append((hit.distance, hit.segment_id, hit.offset))

        # 2. 未被全局索引覆盖的 Segment: brute-force
        covered = global_index.included_segments if global_index else set()

        # Active Segment (一定不在 covered 中)
        bf_results = brute_force_search(
            self.engine.active_segment, query, k, snap.read_lsn
        )
        seg_id = self.engine.active_segment.meta.segment_id
        for offset, dist in bf_results:
            candidates.append((dist, seg_id, offset))

        # Build 后新增的 Sealed Segments
        for seg in self.engine.sealed_segments:
            if seg.meta.segment_id not in covered:
                bf_results = brute_force_search(seg, query, k, snap.read_lsn)
                for offset, dist in bf_results:
                    candidates.append((dist, seg.meta.segment_id, offset))

        return candidates

    def _assemble_results(self, top_k):
        """回填 pk 和 scalar 数据"""
        results = []
        for dist, seg_id, offset in top_k:
            seg = self._find_segment(seg_id)
            results.append({
                "pk": seg.pks[offset],
                "distance": dist,
                "scalars": {
                    fname: col[offset]
                    for fname, col in seg.scalar_columns.items()
                },
            })
        return results

    def _find_segment(self, seg_id: str) -> SegmentData:
        if self.engine.active_segment.meta.segment_id == seg_id:
            return self.engine.active_segment
        for seg in self.engine.sealed_segments:
            if seg.meta.segment_id == seg_id:
                return seg
        raise ValueError(f"Segment {seg_id} not found")
```

---

## 4. 数据序列化与反序列化

### 4.1 各列的序列化格式选型

| 数据列 | 数据类型 | 序列化格式 | 理由 |
|--------|---------|-----------|------|
| `vectors` | `np.ndarray` float32 | `.npy` (numpy binary) | 零拷贝 mmap 兼容，无需 pickle |
| `lsns` | `np.ndarray` int64 | `.npy` | 同上 |
| `pks` (int 主键) | `List[int]` | `.npy` (int64 array) | 紧凑高效 |
| `pks` (string 主键) | `List[str]` | `.pkl` (pickle) | 变长字符串无法用 numpy |
| `scalar_columns` | `Dict[str, List]` | `.pkl` (pickle) | 混合类型，灵活 |
| `delete_bitset` | `bytearray` | `.bin` (raw bytes) | 直接 `write(bytes)` |
| `SegmentMeta` | dataclass | `.json` | 人可读，便于调试 |
| 索引 (HNSWLib) | C++ 对象 | `.bin` (hnswlib native) | 用 `index.save_index()` |
| 索引 (IVF 全局) | centroids + inverted_lists | `.pkl` (pickle) | 不重复存储 vectors，从 Segment 读取 |
| 量化器 (RaBitQ) | rotation_matrix + binary_codes | `.pkl` (pickle) | 独立序列化，便于单独更新 |
| 全局映射表 | `List[Tuple[str,int]]` | `.pkl` (pickle) | global_id → (seg_id, offset) |

### 4.2 Segment 的 flush (内存→磁盘) 实现

```python
import json, pickle, numpy as np
from pathlib import Path

class SegmentSerializer:
    """将 SegmentData 持久化到磁盘 / 从磁盘恢复"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def _seg_dir(self, seg_id: str) -> Path:
        d = self.base_dir / seg_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def flush(self, segment: SegmentData):
        """将 Segment 当前数据写入磁盘"""
        d = self._seg_dir(segment.meta.segment_id)
        n = segment.meta.row_count

        # 1) Meta
        with open(d / "meta.json", "w") as f:
            json.dump({
                "segment_id": segment.meta.segment_id,
                "collection_name": segment.meta.collection_name,
                "state": segment.meta.state.value,
                "row_count": n,
                "max_rows": segment.meta.max_rows,
                "min_lsn": int(segment.meta.min_lsn),
                "max_lsn": int(segment.meta.max_lsn),
            }, f, indent=2)

        # 2) Vectors: 只写有效行 [0:n]
        np.save(d / "vectors.npy", segment.vectors[:n])

        # 3) LSN 列
        np.save(d / "lsn.npy", segment.lsns[:n])

        # 4) PK 列 (根据类型选择格式)
        pk_field = segment.schema.primary_field()
        if pk_field.dtype == FieldType.INT:
            np.save(d / "pk.npy", np.array(segment.pks, dtype=np.int64))
        else:
            with open(d / "pk.pkl", "wb") as f:
                pickle.dump(segment.pks, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 5) 标量列
        with open(d / "scalars.pkl", "wb") as f:
            pickle.dump(segment.scalar_columns, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # 6) Delete Bitset
        with open(d / "delete_bitset.bin", "wb") as f:
            f.write(bytes(segment.delete_bitset))

    def load(self, seg_id: str, schema: CollectionSchema) -> SegmentData:
        """从磁盘恢复一个 Segment"""
        d = self._seg_dir(seg_id)

        # 1) Meta
        with open(d / "meta.json", "r") as f:
            raw = json.load(f)
        meta = SegmentMeta(
            segment_id=raw["segment_id"],
            collection_name=raw["collection_name"],
            state=SegmentState(raw["state"]),
            row_count=raw["row_count"],
            max_rows=raw["max_rows"],
            min_lsn=raw["min_lsn"],
            max_lsn=raw["max_lsn"],
        )
        seg = SegmentData(meta, schema)

        # 2) Vectors
        vectors = np.load(d / "vectors.npy")
        seg.vectors[:meta.row_count] = vectors
        seg._write_cursor = meta.row_count

        # 3) LSN
        seg.lsns[:meta.row_count] = np.load(d / "lsn.npy")

        # 4) PK
        pk_field = schema.primary_field()
        if pk_field.dtype == FieldType.INT:
            seg.pks = np.load(d / "pk.npy").tolist()
        else:
            with open(d / "pk.pkl", "rb") as f:
                seg.pks = pickle.load(f)

        # 5) Scalars
        with open(d / "scalars.pkl", "rb") as f:
            seg.scalar_columns = pickle.load(f)

        # 6) Delete Bitset
        with open(d / "delete_bitset.bin", "rb") as f:
            seg.delete_bitset = bytearray(f.read())

        return seg
```

### 4.3 总体磁盘目录结构

```text
{data_dir}/
├── meta/
│   └── product_vectors/
│       └── schema.json             ← Collection Schema
├── segments/
│   ├── seg_a1b2c3d4/              ← Sealed + Indexed Segment (HNSWLib 策略时)
│   │   ├── meta.json
│   │   ├── vectors.npy             ← 48.8 MB (100K × 128 × 4B)
│   │   ├── pk.npy                  ← 0.8 MB  (100K × 8B)
│   │   ├── scalars.pkl
│   │   ├── lsn.npy
│   │   ├── delete_bitset.bin       ← 12.5 KB (100K bits)
│   │   └── index/                  ← 仅 HNSWLib Per-Segment 策略时存在
│   │       ├── hnsw.bin
│   │       └── index_meta.json
│   └── seg_e5f6g7h8/              ← Active (GROWING) Segment
│       ├── meta.json
│       ├── vectors.npy
│       └── ...
├── global_index/                  ← 仅 IVF+RaBitQ 全局策略时存在
│   ├── index_meta.json             ← 索引类型、参数、build 时间戳、覆盖的 Segment 列表
│   ├── ivf.pkl                    ← centroids + inverted_lists (不包含 vectors)
│   ├── rabitq.pkl                 ← TrueRaBitQ 量化器状态
│   └── global_id_map.pkl          ← global_id → (seg_id, offset) 映射表
└── global/
    └── pk_map.pkl                  ← 全局 PK -> (seg_id, offset) 映射
```

### 4.4 全局 IVF+RaBitQ 索引的序列化

```python
import time

class GlobalIVFIndexSerializer:
    """全局 IVF+RaBitQ 索引的序列化/反序列化"""

    @staticmethod
    def save(base_dir: Path, global_index: GlobalIVFIndex):
        """将全局索引持久化到磁盘"""
        d = base_dir / "global_index"
        d.mkdir(parents=True, exist_ok=True)

        # 1) Meta: 索引类型、参数、Build 信息
        meta = {
            "index_type": "ivf_rabitq",
            "nlist": global_index.ivf.nlist,
            "nprobe": global_index.ivf.nprobe,
            "total_vectors": len(global_index.global_id_map),
            "build_lsn": global_index.build_lsn,
            "build_timestamp": time.time(),
            "included_segments": list(global_index.included_segments),
        }
        with open(d / "index_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 2) IVF: 只保存 centroids + inverted_lists (不保存 vectors)
        ivf_data = {
            "centroids": global_index.ivf.centroids,
            "inverted_lists": global_index.ivf.inverted_lists,
            "vector_to_cluster": global_index.ivf.vector_to_cluster,
            "nlist": global_index.ivf.nlist,
            "nprobe": global_index.ivf.nprobe,
            "metric": global_index.ivf.metric,
        }
        with open(d / "ivf.pkl", "wb") as f:
            pickle.dump(ivf_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 3) RaBitQ 量化器
        global_index.quantizer.save(str(d / "rabitq.pkl"))

        # 4) 全局映射表
        with open(d / "global_id_map.pkl", "wb") as f:
            pickle.dump(global_index.global_id_map, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(base_dir: Path) -> Optional[GlobalIVFIndex]:
        """从磁盘恢复全局索引"""
        d = base_dir / "global_index"
        if not (d / "index_meta.json").exists():
            return None

        result = GlobalIVFIndex()

        # 1) Meta
        with open(d / "index_meta.json", "r") as f:
            meta = json.load(f)
        result.build_lsn = meta["build_lsn"]
        result.included_segments = set(meta["included_segments"])

        # 2) IVF
        with open(d / "ivf.pkl", "rb") as f:
            ivf_data = pickle.load(f)
        ivf = IVFIndex(
            nlist=ivf_data["nlist"],
            nprobe=ivf_data["nprobe"],
            metric=ivf_data["metric"]
        )
        ivf.centroids = ivf_data["centroids"]
        ivf.inverted_lists = ivf_data["inverted_lists"]
        ivf.vector_to_cluster = ivf_data["vector_to_cluster"]
        ivf.is_built = True
        result.ivf = ivf

        # 3) RaBitQ
        result.quantizer = TrueRaBitQ.load(str(d / "rabitq.pkl"))

        # 4) 全局映射表
        with open(d / "global_id_map.pkl", "rb") as f:
            result.global_id_map = pickle.load(f)

        result.is_built = True
        return result
```

> **设计要点**: IVF 序列化时**不重复存储 vectors**。Segment 已经有 `vectors.npy`，搜索时的 Rerank 阶段通过 `global_id_map` 映射回 Segment 读取原始向量，避免双倍存储开销。

---

## 5. gRPC API 协议定义

### 5.1 完整 Protobuf 定义 (`vectordb_service.proto`)

```protobuf
syntax = "proto3";
package vectordb;

// ==================== Service ====================
service VectorDBService {
    // 集合管理
    rpc CreateCollection(CreateCollectionRequest) returns (StatusResponse);
    rpc DropCollection(DropCollectionRequest) returns (StatusResponse);
    rpc DescribeCollection(DescribeCollectionRequest) returns (DescribeCollectionResponse);

    // 数据读写
    rpc Insert(InsertRequest) returns (MutationResponse);
    rpc Upsert(UpsertRequest) returns (MutationResponse);
    rpc Delete(DeleteRequest) returns (MutationResponse);

    // 检索
    rpc Search(SearchRequest) returns (SearchResponse);

    // 索引管理
    rpc CreateIndex(CreateIndexRequest) returns (StatusResponse);
    rpc DropIndex(DropIndexRequest) returns (StatusResponse);
    rpc DescribeIndex(DescribeIndexRequest) returns (DescribeIndexResponse);
}

// ==================== 通用 ====================
message StatusResponse {
    int32 code = 1;       // 0=OK, 非 0=错误
    string message = 2;
}

// ==================== Schema & Collection ====================
message FieldSchemaPb {
    string name = 1;
    string dtype = 2;     // "int", "string", "vector"
    bool is_primary = 3;
    int32 dim = 4;        // vector only
    string metric = 5;    // "L2", "IP", "COSINE"
}

message CreateCollectionRequest {
    string collection_name = 1;
    repeated FieldSchemaPb fields = 2;
}
message DropCollectionRequest {
    string collection_name = 1;
}
message DescribeCollectionRequest {
    string collection_name = 1;
}
message DescribeCollectionResponse {
    StatusResponse status = 1;
    string collection_name = 2;
    repeated FieldSchemaPb fields = 3;
    int64 row_count = 4;
    int32 segment_count = 5;
}

// ==================== DML: Insert / Upsert / Delete ====================
// 使用列式传输: 每个字段单独传一个数组
message FieldData {
    string field_name = 1;
    oneof data {
        IntArray int_data = 2;
        StringArray string_data = 3;
        VectorArray vector_data = 4;
    }
}
message IntArray { repeated int64 values = 1; }
message StringArray { repeated string values = 1; }
message VectorArray {
    repeated float values = 1;  // 扁平化: [v0_d0, v0_d1,..., v1_d0, ...]
    int32 dim = 2;
}

message InsertRequest {
    string collection_name = 1;
    repeated FieldData fields = 2;
    int32 num_rows = 3;
}
message UpsertRequest {
    string collection_name = 1;
    repeated FieldData fields = 2;
    int32 num_rows = 3;
}
message DeleteRequest {
    string collection_name = 1;
    repeated int64 int_pks = 2;     // INT 主键
    repeated string string_pks = 3; // STRING 主键 (二选一)
}
message MutationResponse {
    StatusResponse status = 1;
    int64 affected_count = 2;
}

// ==================== DQL: Search ====================
message SearchRequest {
    string collection_name = 1;
    repeated float query_vector = 2;  // shape=(dim,)
    int32 top_k = 3;
    // 可选搜索参数
    int32 ef_search = 10;             // HNSW 搜索宽度
    int32 nprobe = 11;                // IVF 搜索探针
}
message SearchResponse {
    StatusResponse status = 1;
    repeated SearchHit hits = 2;
}
message SearchHit {
    FieldData pk = 1;
    float distance = 2;
    repeated FieldData scalar_fields = 3;
}

// ==================== 索引管理 ====================
message CreateIndexRequest {
    string collection_name = 1;
    string field_name = 2;
    string index_type = 3;            // "hnsw" 或 "ivf_rabitq"
    map<string, string> params = 4;   // HNSWLib: {"M": "16", "ef_construction": "200"}
                                       // IVF+RaBitQ: {"nlist": "100", "nprobe": "10"}
}
message DropIndexRequest {
    string collection_name = 1;
    string field_name = 2;
}
message DescribeIndexRequest {
    string collection_name = 1;
    string field_name = 2;
}
message DescribeIndexResponse {
    StatusResponse status = 1;
    string index_type = 2;
    map<string, string> params = 3;
    int64 indexed_row_count = 4;
    int32 indexed_segment_count = 5;
}
```

### 5.2 为什么向量用列式扁平化传输

传输 1000 条 128 维向量时：
| 方案 | Protobuf 体积 | 原因 |
|------|--------------|------|
| 每行一个 `repeated float` 嵌套消息 | ~520 KB | 1000 个子消息，每个有 tag + length 开销 |
| **所有向量扁平化到一个 `repeated float`** | ~512 KB | 单个 packed repeated field，几乎零开销 |

扁平化方案更高效。客户端和服务端通过 `dim` 参数将扁平数组 reshape 为 `(N, dim)` 矩阵。

---

## 6. 并发控制与锁设计

### 6.1 写入路径的锁模型

#### 6.1.1 分段锁 (Striped/Sharded Lock)

对 `pk_map` 使用全局 `RWLock` 开销太大，采用**分段锁**方案：

```python
import threading

class StripedLock:
    """分段锁: 将 PK 空间哈希到 N 个桶，每个桶一把互斥锁"""
    def __init__(self, num_stripes: int = 64):
        self.num_stripes = num_stripes
        self.locks = [threading.Lock() for _ in range(num_stripes)]

    def get_lock(self, pk: Union[int, str]) -> threading.Lock:
        stripe_id = hash(pk) % self.num_stripes
        return self.locks[stripe_id]
```

#### 6.1.2 完整的写入流程 (带锁)

```python
class WriteExecutor:
    def __init__(self, engine: CollectionEngine, lsn_generator):
        self.engine = engine
        self.lsn_gen = lsn_generator  # AtomicCounter
        self.striped_lock = StripedLock(num_stripes=64)
        self.segment_lock = threading.Lock()  # 保护 seal 操作

    def insert_row(self, row: Row):
        lock = self.striped_lock.get_lock(row.pk)
        with lock:
            row.lsn = self.lsn_gen.next()
            # 1. 检查 PK 是否已存在
            existing = self.engine.pk_map.get(row.pk)
            if existing is not None:
                # Upsert: 标记旧行删除
                old_seg_id, old_offset = existing
                old_seg = self._find_segment(old_seg_id)
                old_seg.mark_deleted(old_offset)

            # 2. 写入 Active Segment
            with self.segment_lock:
                if self.engine.active_segment.is_full():
                    self.engine.seal_active_segment()
                seg = self.engine.active_segment

            offset = seg.append_row(row)

            # 3. 更新 PK 映射
            self.engine.pk_map[row.pk] = (seg.meta.segment_id, offset)
```

### 6.2 读写并发保障

| 操作 | 锁策略 | 说明 |
|------|--------|------|
| Insert/Upsert | `StripedLock(pk)` + `segment_lock` | 细粒度保护 PK 去重和 Segment 切换 |
| Delete | `StripedLock(pk)` | 只修改 Bitset（原子位操作） |
| Search | **无锁** | 读取 Snapshot LSN 后只读遍历 |
| CreateIndex (HNSWLib) | **无锁** (对 Sealed Segment) | SEALED 状态的 Segment 不可变 |
| CreateIndex (IVF+RaBitQ) | **无写锁**, 完成后原子切换引用 | Build 期间读取已有数据的快照，完成后原子切换 `global_ivf_index` |

查询操作（Search）**不需要加写锁**的原因：
1. Search 只读取 SEALED/INDEXED Segment 的已有数据，这些数据不可变
2. 对 Active Segment 的暴力搜索使用快照：先读取 `_write_cursor`，只搜索 `[0:cursor]` 范围
3. `delete_bitset` 的单 bit 操作是字节级原子的，即使并发修改也不会导致 crash（最多额外过滤一个正在被删除的行）

---

## 7. MVCC 多版本控制与查询快照 (Snapshot)

### 7.1 全局 LSN 发号器

```python
import threading

class LSNGenerator:
    """全局单调递增的 Log Sequence Number 发号器"""
    def __init__(self, start: int = 0):
        self._counter = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def current(self) -> int:
        return self._counter
```

### 7.2 Snapshot Read 实现

```python
@dataclass
class ReadSnapshot:
    """一次查询请求的快照上下文"""
    read_lsn: int          # 本次查询能看到的最大 LSN
    active_cursor: int     # Active Segment 的写入游标快照

def acquire_snapshot(lsn_gen: LSNGenerator,
                     active_seg: SegmentData) -> ReadSnapshot:
    """获取当前时刻的读快照"""
    return ReadSnapshot(
        read_lsn=lsn_gen.current(),
        active_cursor=active_seg._write_cursor,
    )
```

搜索时的可见性判定:
```python
def is_visible(segment: SegmentData, offset: int, snap: ReadSnapshot) -> bool:
    """判定第 offset 行在快照 snap 下是否可见"""
    # 条件 1: 写入 LSN <= 读快照 LSN (即不是"未来"的数据)
    if segment.lsns[offset] > snap.read_lsn:
        return False
    # 条件 2: 未被删除
    if segment.is_deleted(offset):
        return False
    return True
```

### 7.3 MVCC 垃圾回收时机 (Phase 2 Compaction 预设计)

Phase 1 暂不实现自动后台 Compaction，但设计好接口便于 Phase 2 扩展:

```python
class CompactionPolicy:
    """决定哪些 Segment 需要合并的策略"""
    def should_compact(self, segments: List[SegmentData],
                       min_active_read_lsn: int) -> List[str]:
        """
        返回需要合并的 segment_id 列表
        min_active_read_lsn: 当前所有活跃查询中最小的 read_lsn,
                             低于此值的软删行可以安全物理删除
        """
        candidates = []
        for seg in segments:
            if seg.meta.state != SegmentState.INDEXED:
                continue
            deleted_count = sum(
                seg.is_deleted(i) for i in range(seg.meta.row_count)
            )
            delete_ratio = deleted_count / max(seg.meta.row_count, 1)
            # 软删比例超过 30% 且所有行的 LSN 都小于最小活跃读 LSN
            if delete_ratio > 0.3 and seg.meta.max_lsn < min_active_read_lsn:
                candidates.append(seg.meta.segment_id)
        return candidates
```

---

## 附录: 关键数据结构一览

```text
┌─────────────────────────────────────────────────────────────────┐
│                    CollectionEngine                              │
│                                                                  │
│  schema: CollectionSchema                                        │
│  index_strategy: IndexStrategy (hnsw | ivf_rabitq)               │
│  active_segment: SegmentData (GROWING)                           │
│  sealed_segments: List[SegmentData] (SEALED/INDEXED)             │
│  pk_map: Dict[PK, (seg_id, offset)]                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   SegmentData                             │   │
│  │                                                           │   │
│  │  meta: SegmentMeta {id, state, row_count, lsn_range}     │   │
│  │                                                           │   │
│  │  [Column: vectors]     np.ndarray  (max_rows, dim) f32   │   │
│  │  [Column: pks]         List[int|str]                      │   │
│  │  [Column: scalars]     Dict[field -> List[val]]           │   │
│  │  [Column: lsns]        np.ndarray  (max_rows,) i64       │   │
│  │  [System: bitset]      bytearray   (max_rows bits)       │   │
│  │                                                           │   │
│  │  ┌────────────────────────────────────┐                   │   │
│  │  │  SegmentIndex (仅 HNSWLib 策略)   │                   │   │
│  │  │  index: hnswlib.Index              │                   │   │
│  │  │  index_id == segment offset        │                   │   │
│  │  └────────────────────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GlobalIVFIndex (仅 IVF+RaBitQ 策略)                       │   │
│  │                                                           │   │
│  │  ivf: IVFIndex (centroids + inverted_lists)               │   │
│  │  quantizer: TrueRaBitQ (rotation_matrix + binary_codes)   │   │
│  │  global_id_map: List[(seg_id, offset)]                     │   │
│  │  included_segments: Set[str]                               │   │
│  │  build_lsn: int                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
