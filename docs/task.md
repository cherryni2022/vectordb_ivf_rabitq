# 开发任务清单: 向量数据库 Phase 1

> 对应 [plan.md](./plan.md) 的详细任务拆解。每个任务为最小可独立测试的编码单元。

---

## Sprint 1：核心存储引擎

> 目标：内存中完整跑通 Insert → Seal → CreateIndex(hnsw) → Search 流程。

### 1.1 Schema 层 [`vectordb/server/schema.py`]

- [ ] **1.1.1** 定义 `FieldType` 枚举（INT / STRING / VECTOR）
- [ ] **1.1.2** 定义 `MetricType` 枚举（L2 / IP / COSINE）
- [ ] **1.1.3** 定义 `FieldSchema` dataclass（name, dtype, is_primary, dim, metric）
- [ ] **1.1.4** 定义 `CollectionSchema` dataclass（collection_name, fields, created_at, version）
  - `primary_field()` 方法
  - `vector_field()` 方法
- [ ] **1.1.5** `validate_schema(schema: CollectionSchema)` 工厂函数
  - 检查：唯一主键、至少一个 VECTOR 字段、字段名不重复、VECTOR 字段必须有 dim 和 metric

### 1.2 存储引擎层 [`vectordb/server/segment.py`]

- [ ] **1.2.1** 定义 `SegmentState` 枚举（GROWING / SEALED / INDEXED）
- [ ] **1.2.2** 定义 `SegmentMeta` dataclass（segment_id, collection_name, state, row_count, max_rows, min_lsn, max_lsn）
- [ ] **1.2.3** 定义 `Row` dataclass（pk, vector, scalars, lsn）
- [ ] **1.2.4** 实现 `SegmentData` 类
  - `__init__`: 预分配 vectors numpy 矩阵、pks list、scalar_columns dict、lsns、delete_bitset
  - `append_row(row) -> int`：追加数据，返回 offset
  - `mark_deleted(offset: int)`：位图软删除
  - `is_deleted(offset: int) -> bool`
  - `is_full() -> bool`
- [ ] **1.2.5** 实现 `CollectionEngine` 类
  - `active_segment`、`sealed_segments`、`pk_map` 字段
  - `seal_active_segment()`：关闭当前 segment，新建 active
  - `_new_segment()` 私有方法

### 1.3 LSN 与 MVCC [`vectordb/server/lsn.py` + `vectordb/server/mvcc.py`]

- [ ] **1.3.1** `LSNGenerator`：线程安全的递增计数器（`next()` / `current()`）
- [ ] **1.3.2** `ReadSnapshot` dataclass（read_lsn, active_cursor）
- [ ] **1.3.3** `acquire_snapshot(lsn_gen, active_seg) -> ReadSnapshot`
- [ ] **1.3.4** `is_visible(segment, offset, snap) -> bool`（LSN 可见 + 未删除）

### 1.4 HNSWLib Per-Segment 索引策略 [`vectordb/server/index_strategy.py`]

- [ ] **1.4.1** 定义 `IndexStrategy` 枚举（HNSW_PER_SEGMENT / IVF_RABITQ_GLOBAL）
- [ ] **1.4.2** 实现 `SegmentIndex` 类（包装 hnswlib）
  - `build_index(params: dict)`：调用 `hnswlib.Index`，labels = segment offset
  - `search(query, k) -> List[(offset, dist)]`
- [ ] **1.4.3** 实现 `brute_force_search(segment, query, k, read_lsn) -> List[(offset, dist)]`
- [ ] **1.4.4** 实现 `SearchCoordinator` 类（仅 HNSW 路径，IVF 路径暂时 stub）
  - `search(query, k, snap, index_strategy) -> List[dict]`
  - `_search_hnsw()`
  - `_assemble_results()`

### 1.5 Sprint 1 集成测试 [`tests/server/test_sprint1.py`]

- [ ] **1.5.1** 测试 Schema 校验（正常 + 异常 case）
- [ ] **1.5.2** 测试 SegmentData append → seal → is_full 状态流转
- [ ] **1.5.3** 测试 mark_deleted → is_deleted 位图操作
- [ ] **1.5.4** 测试 LSNGenerator 线程安全（多线程并发调用 next()）
- [ ] **1.5.5** 测试 ReadSnapshot + is_visible 可见性过滤
- [ ] **1.5.6** 端到端测试：Insert 1000 条 → seal → build_index(hnsw) → search → 验证结果正确性

---

## Sprint 2：序列化层

> 目标：引擎数据持久化到磁盘，重启后可从磁盘恢复完整状态。

### 2.1 Schema 序列化 [`vectordb/server/serializer.py`]

- [ ] **2.1.1** `SchemaSerializer.save(base_dir, schema)`：写 `meta/{collection_name}/schema.json`
- [ ] **2.1.2** `SchemaSerializer.load(base_dir, collection_name) -> CollectionSchema`：读 JSON 反序列化
- [ ] **2.1.3** `SchemaSerializer.list_collections(base_dir) -> List[str]`：扫描 meta/ 目录

### 2.2 Segment 序列化 [`vectordb/server/serializer.py`]

- [ ] **2.2.1** `SegmentSerializer.flush(segment)`：写 meta.json / vectors.npy / lsn.npy / pk.npy(或 pk.pkl) / scalars.pkl / delete_bitset.bin
- [ ] **2.2.2** `SegmentSerializer.load(seg_id, schema) -> SegmentData`：从磁盘重建 SegmentData
- [ ] **2.2.3** HNSWLib 索引 save：`SegmentSerializer.save_hnsw_index(seg_id, segment_index)`
- [ ] **2.2.4** HNSWLib 索引 load：`SegmentSerializer.load_hnsw_index(seg_id, segment) -> SegmentIndex`

### 2.3 CollectionEngine 启动恢复

- [ ] **2.3.1** `CollectionEngine.recover_from_disk(base_dir)`：
  - 加载 schema.json
  - 扫描 segments/ 目录，按 meta.json 中的 state 恢复各 Segment
  - 对 INDEXED 状态的 segment，加载 hnsw.bin
  - 重建 `pk_map`（遍历所有 segment 的 pks）

### 2.4 Sprint 2 集成测试 [`tests/server/test_sprint2.py`]

- [ ] **2.4.1** Insert → flush → 清空内存 → load → 验证数据一致性
- [ ] **2.4.2** flush → load HNSWLib 索引 → search → 结果与内存搜索一致
- [ ] **2.4.3** recover_from_disk 完整测试（含多个 Segment）

---

## Sprint 3：IVF+RaBitQ 全局索引策略

> 目标：实现全局索引构建、搜索和持久化，复用现有 IVFIndex 和 TrueRaBitQ。

### 3.1 全局索引数据结构 [`vectordb/server/index_strategy.py`]

- [ ] **3.1.1** 定义 `GlobalIVFIndex` dataclass（ivf, quantizer, global_id_map, included_segments, build_lsn, is_built）
- [ ] **3.1.2** 定义 `GlobalSearchHit` dataclass（segment_id, offset, distance）

### 3.2 GlobalIVFIndexBuilder [`vectordb/server/index_strategy.py`]

- [ ] **3.2.1** `GlobalIVFIndexBuilder.build(segments, params, current_lsn) -> GlobalIVFIndex`
  - Step 1: 遍历所有 Segment，跳过已删除行，拼接 all_vectors + 构建 global_id_map
  - Step 2: 调用 `IVFIndex(nlist=..., nprobe=...)` 训练
  - Step 3: 调用 `TrueRaBitQ(dimension=...)` train + encode
  - Step 4: 组装 GlobalIVFIndex

### 3.3 GlobalIVFSearcher [`vectordb/server/index_strategy.py`]

- [ ] **3.3.1** `GlobalIVFSearcher.search(global_index, query, k, segments_map, read_lsn, rerank_factor) -> List[GlobalSearchHit]`
  - 调用 `ivf.search_with_rabitq` 得到 raw_results
  - 通过 global_id_map 映射回 (seg_id, offset)
  - 过滤 is_deleted + LSN 可见性
  - 返回前 k 个

### 3.4 SearchCoordinator IVF 路径补全 [`vectordb/server/index_strategy.py`]

- [ ] **3.4.1** `SearchCoordinator._search_ivf_rabitq()`：全局 IVF 搜索 + 遍历未覆盖 Segment 的 brute-force 兜底
- [ ] **3.4.2** `CollectionEngine` 新增 `global_ivf_index: Optional[GlobalIVFIndex]` 字段
- [ ] **3.4.3** `CollectionEngine.build_global_index(params) -> GlobalIVFIndex`：调用 builder，更新 `self.global_ivf_index`

### 3.5 全局索引序列化 [`vectordb/server/serializer.py`]

- [ ] **3.5.1** `GlobalIVFIndexSerializer.save(base_dir, global_index)`：写 `global_index/index_meta.json` / `ivf.pkl` / `rabitq.pkl` / `global_id_map.pkl`
- [ ] **3.5.2** `GlobalIVFIndexSerializer.load(base_dir) -> Optional[GlobalIVFIndex]`：从磁盘恢复

### 3.6 Sprint 3 集成测试 [`tests/server/test_sprint3.py`]

- [ ] **3.6.1** 构建全局索引 → 搜索 → 验证结果（与 brute-force 结果对比 recall@10）
- [ ] **3.6.2** build 后新增数据 → search → 验证新数据通过 brute-force 路径被召回
- [ ] **3.6.3** 全局索引 save → load → search → 结果一致性验证
- [ ] **3.6.4** 软删除一条已纳入全局索引的向量 → search → 被删除的 pk 不在结果中

---

## Sprint 4：并发控制

> 目标：写入路径线程安全，读写可并发无死锁。

### 4.1 分段锁 [`vectordb/server/concurrency.py`]

- [ ] **4.1.1** `StripedLock` 类（64 分段，`get_lock(pk) -> threading.Lock`）

### 4.2 WriteExecutor [`vectordb/server/concurrency.py`]

- [ ] **4.2.1** `WriteExecutor.__init__`（engine, lsn_gen, StripedLock, segment_lock）
- [ ] **4.2.2** `WriteExecutor.insert_row(row)`：
  - StripedLock(pk) 加锁
  - 分配 LSN
  - 检查 PK 是否存在（Upsert 场景：旧行 mark_deleted）
  - segment_lock 保护 seal 操作
  - `seg.append_row(row)` → 更新 `pk_map`
- [ ] **4.2.3** `WriteExecutor.delete_row(pk)`：StripedLock(pk) 加锁 → mark_deleted → 更新 pk_map
- [ ] **4.2.4** `WriteExecutor.insert_batch(rows)`：批量调用 insert_row

### 4.3 Sprint 4 并发测试 [`tests/server/test_sprint4.py`]

- [ ] **4.3.1** 100 线程并发 Insert（各不同 PK） → 验证最终数量无丢失
- [ ] **4.3.2** 并发 Upsert 同一 PK → 最终只存在一条记录
- [ ] **4.3.3** 写入同时并发 Search（无锁读） → 验证无异常崩溃
- [ ] **4.3.4** 并发 Insert + Delete → 验证最终状态正确

---

## Sprint 5：gRPC 服务层

> 目标：将引擎暴露为 gRPC 服务，通过 Python gRPC client 完成端到端验证。

### 5.1 Protobuf 定义 [`vectordb/server/grpc/vectordb_service.proto`]

- [ ] **5.1.1** 定义所有 message（FieldSchemaPb, StatusResponse, InsertRequest, SearchRequest 等）
- [ ] **5.1.2** 定义 `VectorDBService`（9 个 RPC：CreateCollection, DropCollection, DescribeCollection, Insert, Upsert, Delete, Search, CreateIndex, DropIndex）
- [ ] **5.1.3** 运行 `python -m grpc_tools.protoc` 生成 `_pb2.py` / `_pb2_grpc.py`

### 5.2 gRPC Servicer 实现 [`vectordb/server/grpc/servicer.py`]

- [ ] **5.2.1** `VectorDBServicer` 类，持有 `Dict[str, CollectionEngine]` 和 `WriteExecutor`
- [ ] **5.2.2** `CreateCollection` RPC：解析 Schema → 校验 → 创建 Engine → 持久化 schema.json
- [ ] **5.2.3** `DropCollection` RPC：删除引擎内存状态 + 磁盘目录（可选）
- [ ] **5.2.4** `DescribeCollection` RPC：返回 Schema + 行数 + Segment 数
- [ ] **5.2.5** `Insert` RPC：解析列式 FieldData → 构建 Row 列表 → WriteExecutor.insert_batch
- [ ] **5.2.6** `Upsert` RPC：同 Insert，WriteExecutor 内部自动处理 Upsert 逻辑
- [ ] **5.2.7** `Delete` RPC：解析 pk 列表 → WriteExecutor.delete_row
- [ ] **5.2.8** `Search` RPC：acquire_snapshot → SearchCoordinator.search → 构造 SearchResponse
- [ ] **5.2.9** `CreateIndex` RPC：根据 index_type 触发 SegmentIndex 构建或 GlobalIVFIndexBuilder 构建
- [ ] **5.2.10** `DropIndex` RPC：清除索引状态

### 5.3 gRPC Server 启动入口 [`vectordb/server/main.py`]

- [ ] **5.3.1** 解析命令行参数（`--host`, `--port`, `--data-dir`）
- [ ] **5.3.2** 启动时调用 `recover_from_disk()`，加载所有已有 Collection
- [ ] **5.3.3** 创建 `grpc.server(ThreadPoolExecutor)` → add_servicer → server.start()

### 5.4 安装依赖

- [ ] **5.4.1** `uv add grpcio grpcio-tools protobuf`（更新 pyproject.toml）

### 5.5 Sprint 5 gRPC 端到端测试 [`tests/server/test_sprint5_grpc.py`]

- [ ] **5.5.1** 启动本地 gRPC server → CreateCollection → Insert 1000 条 → CreateIndex(hnsw) → Search → 验证结果
- [ ] **5.5.2** gRPC Delete → Search → 被删除 pk 不在结果中
- [ ] **5.5.3** 重启 server（recover_from_disk）→ Search → 结果与重启前一致
- [ ] **5.5.4** CreateIndex(ivf_rabitq) → Search → recall@10 ≥ 0.8

---

## Sprint 6：测试完善与收尾

### 6.1 错误处理完善

- [ ] **6.1.1** Schema 校验异常返回 gRPC 错误码（非 0）
- [ ] **6.1.2** 对不存在的 Collection 操作返回 NOT_FOUND
- [ ] **6.1.3** Insert 维度不匹配返回 INVALID_ARGUMENT
- [ ] **6.1.4** Search 前未建索引时，降级为 brute-force（或返回提示）

### 6.2 性能冒烟测试 [`tests/server/test_performance.py`]

- [ ] **6.2.1** 10 万条 128 维向量写入耗时（应 < 30s 单线程）
- [ ] **6.2.2** HNSWLib 索引建立耗时（应 < 60s on 10 万条）
- [ ] **6.2.3** HNSW Search(top_10) QPS（目标 > 1000 QPS 单进程）
- [ ] **6.2.4** IVF+RaBitQ 全局索引 Build 耗时（10 万条 128 维）
- [ ] **6.2.5** IVF+RaBitQ Search recall@10 验证（目标 ≥ 0.8）

### 6.3 文档与总结

- [ ] **6.3.1** 更新 `README.md`：新增 Server 模式的快速启动说明
- [ ] **6.3.2** 更新 `CLAUDE.md`：说明新 `vectordb/server/` 模块结构和测试命令
- [ ] **6.3.3** 打标记：将所有 Sprint 任务状态更新为完成

---

## 任务依赖图

```
Sprint 1 (存储引擎)
    ↓
Sprint 2 (序列化)   ←── 并行 ──→   Sprint 3 (IVF+RaBitQ 全局索引)
    ↓                                       ↓
Sprint 4 (并发控制) ←───────────────────────┘
    ↓
Sprint 5 (gRPC 服务)
    ↓
Sprint 6 (测试 & 收尾)
```

> Sprint 2 和 Sprint 3 可并行开发（前提是 Sprint 1 完成）。Sprint 4 依赖 Sprint 1，但不强依赖 Sprint 2/3（可以先做并发再做序列化）。

---

## 文件新建清单

| 文件路径 | Sprint | 说明 |
|----------|--------|------|
| `vectordb/server/__init__.py` | S1 | 包入口 |
| `vectordb/server/schema.py` | S1 | Schema 定义与校验 |
| `vectordb/server/segment.py` | S1 | Row / SegmentData / CollectionEngine |
| `vectordb/server/lsn.py` | S1 | LSNGenerator / ReadSnapshot |
| `vectordb/server/mvcc.py` | S1 | acquire_snapshot / is_visible |
| `vectordb/server/index_strategy.py` | S1/S3 | 双索引策略实现 |
| `vectordb/server/serializer.py` | S2/S3 | Segment + 全局索引序列化 |
| `vectordb/server/concurrency.py` | S4 | StripedLock / WriteExecutor |
| `vectordb/server/grpc/__init__.py` | S5 | gRPC 包入口 |
| `vectordb/server/grpc/vectordb_service.proto` | S5 | Protobuf 定义 |
| `vectordb/server/grpc/servicer.py` | S5 | gRPC Servicer 实现 |
| `vectordb/server/main.py` | S5 | Server 启动入口 |
| `tests/server/__init__.py` | S1 | 测试包入口 |
| `tests/server/test_sprint1.py` | S1 | Sprint 1 测试 |
| `tests/server/test_sprint2.py` | S2 | Sprint 2 测试 |
| `tests/server/test_sprint3.py` | S3 | Sprint 3 测试 |
| `tests/server/test_sprint4.py` | S4 | Sprint 4 并发测试 |
| `tests/server/test_sprint5_grpc.py` | S5 | gRPC 端到端测试 |
| `tests/server/test_performance.py` | S6 | 性能冒烟测试 |
