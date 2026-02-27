# 开发计划: 向量数据库 Phase 1 单机服务

> 基于 [Technical_Design_VectorDB.md](./Technical_Design_VectorDB.md) 制定，目标是在现有索引库（IVF、HNSWLib、RaBitQ）之上，构建完整的单机向量数据库服务。

## 现状分析

现有 `vectordb/` 包已经提供了完整的**底层索引能力**：

| 模块 | 文件 | 说明 |
|------|------|------|
| 索引 | `vectordb/index/ivf_index.py` | IVF 聚类索引，支持 `search_with_rabitq` |
| 索引 | `vectordb/index/hnswlib_index.py` | HNSWLib 图索引包装器 |
| 量化 | `vectordb/quantization/true_rabitq.py` | RaBitQ 二值量化 |
| 入口 | `vectordb/core/vector_db.py` | 当前 MVP：单集合、无并发、无 Schema |

**缺失的部分**（全部需新建）：

- Schema 管理层（多 Collection、字段校验）
- Segment 存储引擎（MemTable → Sealed → Indexed 状态机）
- 并发控制（LSN + 分段锁 + MVCC Snapshot）
- 序列化层（Segment flush/load、全局索引持久化）
- 双索引策略路由（HNSWLib Per-Segment / IVF+RaBitQ Global）
- gRPC 服务层（Protobuf 定义 + Server 实现）

---

## 模块结构规划

新建 `vectordb/server/` 目录，作为服务端核心引擎：

```
vectordb/server/
├── __init__.py
├── schema.py          # FieldType, FieldSchema, CollectionSchema
├── segment.py         # Row, SegmentData, SegmentMeta, SegmentState, CollectionEngine
├── lsn.py             # LSNGenerator, ReadSnapshot
├── mvcc.py            # acquire_snapshot, is_visible
├── index_strategy.py  # IndexStrategy, SegmentIndex, GlobalIVFIndex, GlobalIVFIndexBuilder,
│                      #   GlobalIVFSearcher, brute_force_search, SearchCoordinator
├── serializer.py      # SegmentSerializer, GlobalIVFIndexSerializer, SchemaSerializer
├── concurrency.py     # StripedLock, WriteExecutor
└── grpc/
    ├── __init__.py
    ├── vectordb_service.proto
    ├── vectordb_pb2.py        (generated)
    ├── vectordb_pb2_grpc.py   (generated)
    └── servicer.py            # VectorDBServicer 实现
```

---

## 开发阶段规划

### Sprint 1：核心存储引擎（无并发、无 gRPC）

**目标**：实现内存中的 Schema + Segment + 基础读写，能够跑通端到端的 Insert → Seal → Search 流程。

- [ ] `schema.py`：FieldType, FieldSchema, CollectionSchema, 校验逻辑
- [ ] `segment.py`：Row, SegmentMeta, SegmentState, SegmentData, CollectionEngine
- [ ] `lsn.py` + `mvcc.py`：LSNGenerator, ReadSnapshot, is_visible
- [ ] `index_strategy.py`（基础版）：brute_force_search, SegmentIndex (HNSWLib), SearchCoordinator (仅 HNSW 路径)
- [ ] 端到端集成测试（无 gRPC）

**交付物**：可在 Python 中直接实例化 `CollectionEngine`，完成 Insert → Seal → CreateIndex(hnsw) → Search 全流程。

---

### Sprint 2：序列化层

**目标**：将 Sprint 1 的内存引擎持久化到磁盘，并能从磁盘恢复。

- [ ] `serializer.py`：`SegmentSerializer.flush()` / `load()`
- [ ] `serializer.py`：`SchemaSerializer` (schema.json)
- [ ] `serializer.py`：HNSWLib 索引的 save/load（`index_dir/hnsw.bin`）
- [ ] `CollectionEngine.recover_from_disk()`：启动时扫描 `meta/` 和 `segments/`，重建内存状态
- [ ] 序列化集成测试

---

### Sprint 3：IVF+RaBitQ 全局索引策略

**目标**：实现第二种索引策略，复用现有 `IVFIndex` 和 `TrueRaBitQ`。

- [ ] `index_strategy.py`：GlobalIVFIndex, GlobalIVFIndexBuilder, GlobalIVFSearcher
- [ ] `SearchCoordinator._search_ivf_rabitq()`：IVF 全局搜索 + brute-force 兜底
- [ ] `serializer.py`：`GlobalIVFIndexSerializer.save()` / `load()`（global_index/ 目录）
- [ ] 全局索引构建 + 搜索 + 序列化集成测试

---

### Sprint 4：并发控制

**目标**：为写入路径加锁，保证并发安全。

- [ ] `concurrency.py`：StripedLock（64 分段锁）
- [ ] `concurrency.py`：WriteExecutor（带锁的 insert_row 流程）
- [ ] 并发写 + 并发读的集成测试（`concurrent.futures.ThreadPoolExecutor`）

---

### Sprint 5：gRPC 服务层

**目标**：将引擎暴露为标准 gRPC 服务。

- [ ] 编写 `vectordb_service.proto`（参考技术设计文档 5.1）
- [ ] 运行 `grpc_tools.protoc` 生成 `_pb2.py` / `_pb2_grpc.py`
- [ ] `servicer.py`：实现所有 9 个 RPC（CreateCollection, DropCollection, DescribeCollection, Insert, Upsert, Delete, Search, CreateIndex, DropIndex）
- [ ] gRPC Server 启动入口 `vectordb/server/main.py`
- [ ] gRPC 端到端测试（Python gRPC client 测试）

---

### Sprint 6：测试完善与收尾

**目标**：补全测试覆盖，完善错误处理和文档。

- [ ] 错误处理：Schema 校验异常、索引未建异常、并发冲突
- [ ] 补全单元测试（pytest, `tests/server/` 目录）
- [ ] 性能冒烟测试（10 万条 128 维向量，HNSW vs IVF+RaBitQ 搜索耗时对比）
- [ ] 更新 README 和 CLAUDE.md

---

## 关键依赖

| 依赖 | 用途 | 已有？ |
|------|------|--------|
| `hnswlib` | HNSWLib Per-Segment 索引 | ✅ |
| `numpy` | 向量存储和计算 | ✅ |
| `grpcio` + `grpcio-tools` | gRPC 服务 | ❌ 需安装 |
| `protobuf` | Protobuf 序列化 | ❌ 需安装 |
| `pytest` | 测试框架 | ✅ |

---

## 风险与注意事项

| 风险 | 缓解 |
|------|------|
| 全局 IVF Build 时内存峰值高（所有向量拼接为大矩阵） | Build 完成后释放临时变量；Sprint 2+ 可考虑分块 |
| Build 期间新写入的数据只能暴力搜索 | 设计上已接受，fallback 路径已在 SearchCoordinator 中处理 |
| gRPC protoc 代码生成依赖 grpcio-tools 版本 | 固定版本到 pyproject.toml |
