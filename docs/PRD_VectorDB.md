# 产品需求文档 (PRD): 单机向量数据库 (Phase 1)

## 1. 背景与目标
当前 `study_vectordb` 仅作为一个包含（IVF、RaBitQ、HNSWLib）等向量索引算法的依赖库。为了能够在生产环境和实际业务中落地，需要将其演进为一个**高可用、高性能的向量数据库服务**。
本期（Phase 1）目标是实现一个**单机版本的向量数据库服务**，具备独立提供 HTTP/gRPC 网络接口的能力，并解决数据schema管理、并发读写、版本控制等基础数据库能力。

## 2. 演进路线图 (Roadmap)
- **Phase 1: 单机向量数据库服务 (当前范围)**
  - Schema 管理（支持向量与基础标量）
  - gRPC 通信协议引擎
  - 并发读写与基础事务支持
  - MVCC 多版本数据控制与软删除
  - 双索引策略：HNSWLib Per-Segment 索引 / IVF+RaBitQ 全局索引
- **Phase 2: 企业级单机引擎**
  - 混合检索与标量过滤 (Hybrid Search)
  - 持久化灾备 (WAL / Mmap)
  - 后台数据合并任务 (Background Compaction)
  - HNSWLib 索引的 Segment 合并与索引 Rebuild
- **Phase 3: 分布式与高可用**
  - 分布式集群部署 (Sharding)
  - 副本与数据一致性 (Replication & Consistency)

## 3. 功能需求说明 (Phase 1)

### 3.1 数据 Schema 管理
- **需求描述**: 支持对 Collection（集合）的列定义进行管理。
- **支持的数据类型**:
  - `vector`: 表示向量数据（需指定维度 `dim` 和 距离度量方式 `metric`: L2, IP, Cosine等）。
  - 标量数据：目前仅支持 `string` 类型与 `int` 类型。
- **约束**: 一个 Collection 只能有一个主键（PK）列。必须有至少一个 `vector` 类型的列。

### 3.2 单机 Server 服务能力入口
- **需求描述**: 提供标准化的网络通信协议供客户端调用。
- **具体接口设计 (基于 gRPC)**:
  - **集合管理**:
    - `CreateCollection()`: 定义并创建包含 Schema 的表。
    - `DropCollection()`: 删除集合并释放资源。
  - **数据读写 (DML)**:
    - `Insert() / Upsert()`: 单条/批量写入，需根据 Schema 严格校验数据格式与类型，主键相同的 `Upsert` 执行覆写。
    - `Delete()`: 单条/批量通过主键删除。
  - **查询检索 (DQL)**:
    - `Search()`: 接收 Target 向量、`top_k` 等参数进行 KNN 或 ANN 检索。
  - **索引管理**:
    - `CreateIndex()`: 对某一个 Vector 字段构建索引。支持两种策略：
      - `hnsw`: 对已 Sealed 的 Segment 构建 HNSWLib 索引（Per-Segment 粒度）。
      - `ivf_rabitq`: 触发全局 IVF 索引构建 + RaBitQ 量化编码（跨所有 Segment）。
    - `DropIndex() / DescribeIndex()`: 删除与查询索引信息。

### 3.3 并发操作保障机制
- **需求描述**: 服务端必须支持同时处理多个写入和查询请求，且不能出现数据竞态或脏读故障。
- **业务场景**:
  - **并发写**: 多端并发触发 Insert，需保证主键的唯一性和写入原子性。
  - **边写边读**: 写入的数据需要在合适时机对查询可见，且不能阻塞正在进行的查询扫图过程。

### 3.4 版本控制与多版本并发控制 (MVCC)
- **需求描述**: 提供无锁化或细粒度锁读写支持。
- **技术点**:
  - **Segment 化**: 新写入数据进内存表（MemTable），不实时进行复杂全量图的索引重构。
  - **软删除标记**: Delete操作使用 Bitset （位图）标记为已删除，代替物理摘除（物理摘除图节点开销过大）。查询时先召回，再依靠 Bitset 滤除已删除元素。
  - **Snapshot 视图**: 查询请求附带或隐式获取当前数据的时间戳（Version ID/LSN），查询只读取该 Version 及之前可见的数据结构。

### 3.5 双索引策略
- **需求描述**: 系统支持两种索引方案，Collection 创建索引时二选一。
- **方案一: HNSWLib Per-Segment**
  - **适用场景**: 在线实时写入 + 低延迟查询场景。
  - **构建时机**: 每个 Segment 达到阈值 Sealed 后，自动或手动触发为该 Segment 构建 HNSWLib 索引。
  - **搜索方式**: 并发搜索所有 Segment（INDEXED Segment 走 HNSWLib、Active Segment 走暴力搜索），归并 top_k。
  - **Phase 2 演进**: 后台 Compaction 任务将多个小 Segment 合并为大 Segment 并重建索引。
- **方案二: IVF + RaBitQ 全局索引**
  - **适用场景**: 批量数据导入完成后，追求高吞吐、高压缩比的大规模检索场景。
  - **构建时机**: 数据批量写入完成后，用户显式调用 `CreateIndex(type=ivf_rabitq)` 触发全局索引构建。
  - **写入与索引关系**: 写入数据时只写入 Segment 原始数据，不构建索引。全局索引构建时拼接所有 Segment 的向量数据，统一训练 IVF 聚类 + RaBitQ 量化。
  - **搜索方式**: 全局 IVF+RaBitQ 索引覆盖已 Build 的数据；Build 后新写入的数据走暴力搜索兜底，最终归并 top_k。
