# 架构设计文档: 向量数据库 (Phase 1 单机架构)

## 1. 整体系统架构
为了支撑 Phase 1 及后续演进，向量数据库的单机架构从上到下可以划分为四层：**接入层、执行控制层、内存与存储层、索引引擎层。**

### 模块划分视图

```text
+-------------------------------------------------------------+
|                     Client (Python / Go / etc.)             |
+-------------------------------------------------------------+
                              | gRPC
+-------------------------------------------------------------+
|                     Access Layer (接入层)                    |
|   - gRPC Server               - HTTP (Optional for Admin)   |
|   - Auth / Connection Management                            |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|             Execution Layer (执行控制层 / Coordinator)        |
|   - Request Parser & Validator                              |
|   - Schema Manager (元数据管理)                               |
|   - Transaction & Version Manager (LSN 发号器 / MVCC)       |
|   - Index Strategy Manager (索引策略管理)                     |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|             Storage & Engine Layer (存储与内存管理层)         |
|   - Write Buffer (MemTable)  |  - Immutable Segments        |
|   - Deletion Bitset          |  - Segment Meta              |
|   - KV Storage (Primary key -> Offsets)                     |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                     Index Engine Layer (底层索引层)          |
|                                                              |
|  [策略 A: HNSWLib Per-Segment]  [策略 B: IVF+RaBitQ 全局]   |
|   - Per-Segment HNSWLib 索引     - 全局 IVF 聚类索引         |
|   - Scatter-Gather 多路搜索      - 全局 RaBitQ 量化编码       |
|                                  - Global ID ↔ Segment 映射  |
+-------------------------------------------------------------+
```

## 2. 核心组件说明

### 2.1 Access Layer (接入层)
基于 gRPC 的高性能服务端，将用户发过来的 Protobuf 序列化消息转为内部数据结构（如 Python Pydantic Model 或 C++ struct），并下发给具体路由。

### 2.2 Execution Layer (执行层)
- **Schema Manager**: 将 `CreateCollection` 创建的结构信息持久化在本地（如本地 SQLite 库，或 JSON 配置文件中），后续所有 Insert 请求必须在此验证（数据长度、维度、类型）是否合法。
- **Version Manager**: 为所有写入操作分配递增的全局顺序号（LSN - Log Sequence Number），用于构建快照和保证写入顺序。
- **Index Strategy Manager**: 管理 Collection 级别的索引策略（HNSWLib Per-Segment 或 IVF+RaBitQ 全局），并根据策略路由搜索请求到对应的索引引擎。

### 2.3 Storage Layer (存储层)
基于简化的 LSM-Tree 理念：
- **MemTable (内存表)**: 最新写入的数据存储在此，MemTable 为追加写。当 MemTable 达到阈值（如 10 万条）时，转换为不可变的 Immutable Segment，并可触发写入磁盘（Phase 2）。
- **Segment (数据块)**: 每 10 万或 N 万条数据独立成块。搜索时，根据索引策略的不同走不同的搜索链路。
- **Bitset**: 每个 Segment 维护一个 Bitset 数组，用于记录已被删除行的偏移量或 ID。

### 2.4 Index Layer (索引层)

系统提供两种可选的索引策略，Collection 创建索引时二选一：

#### 策略 A: HNSWLib Per-Segment 索引
- **定位**: 在线实时写入 + 低延迟查询的增量索引方案。
- **机制**: 每个 Immutable Segment Sealed 后，独立为该 Segment 内的向量构建 HNSWLib 索引。搜索时并发查询所有 Segment，Scatter-Gather 归并结果。
- **特点**: 索引粒度与 Segment 一致，不影响写入、不阻塞读取。Phase 2 通过 Compaction 合并 Segment 后重建索引。

#### 策略 B: IVF + RaBitQ 全局索引
- **定位**: 批量数据导入完成后，追求高吞吐、高压缩比的离线索引方案。
- **机制**: 数据写入时只写入 Segment 原始数据，不构建索引。全量数据写入完成后，用户显式调用 `CreateIndex(type=ivf_rabitq)` 触发全局索引构建。构建过程拼接所有 Segment 的向量，统一训练 IVF k-means 聚类 + RaBitQ 二值量化编码。
- **特点**: 全局聚类质量更高；RaBitQ 提供 32x 压缩比；搜索时使用两阶段检索（RaBitQ 粗筛 + 精确 Rerank）。Build 后新写入的数据走暴力搜索兜底。

## 3. 读写架构流转

### 3.1 写入链路 (Write Path)
1. 客户端发送 `Insert(List<Rows>)` 请求。
2. Request Validator 查询 Schema，校验所有行的数据类型与维度。
3. Version Manager 生成一个最新 LSN。
4. 检查 Primary Key（目前可以使用互斥锁或者基于哈希的分段锁），在系统内部的 KV Map（存储 PK 到 MemTable Offset 的映射）中登记。若存在则是 Upsert。
5. 将数据追加到当前活跃的 **MemTable** 尾部，并在 KV Map 中记录 `PK -> {SegmentID, Offset, LSN}`。
6. (如果是 Update，同时需要把旧 Segment 对应条目的 Bitset 更新为 "Deleted"，并写入新 MemTable)。
7. 返回写入成功。

> **注意**: 写入链路与索引策略无关。两种策略下写入逻辑完全相同，索引构建是独立的后置操作。

### 3.2 搜索链路 (Read Path)

搜索链路根据 Collection 的索引策略分为两条路径：

#### 3.2.1 HNSWLib Per-Segment 搜索链路

```text
Client: Search(query_vec, top_k=10)
    │
    ├──▶ Active Segment (GROWING): brute-force 暴力搜索
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

#### 3.2.2 IVF + RaBitQ 全局索引搜索链路

```text
Client: Search(query_vec, top_k=10)
    │
    ├──▶ GlobalIVFIndex.search(query, top_k)
    │      1. 找到 nprobe 个最近聚类中心
    │      2. RaBitQ 粗筛候选向量 (近似距离)
    │      3. 取 top_k * rerank_factor 个候选
    │      4. 精确距离 Rerank → 返回 [(global_id, dist)]
    │      5. global_id_map 映射回 (seg_id, offset)
    │
    ├──▶ 未被全局索引覆盖的 Segments: brute-force
    │      (Active Segment + Build 后新增的 Sealed Segments)
    │
    └─── Coordinator: 归并所有结果
         → 过滤: is_deleted(offset)? lsn > read_lsn?
         → 输出 global top_k, 回填 pk 和 scalar 数据
```
