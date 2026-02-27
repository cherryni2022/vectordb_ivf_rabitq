# 开发计划: 向量数据库 Phase 1 单机服务

> 基于 [Technical_Design_VectorDB.md](./Technical_Design_VectorDB.md) 制定，目标是在现有索引库（IVF、HNSWLib、RaBitQ）之上，构建完整的单机向量数据库服务。

---

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

**技术设计参照**：
- [§1 Schema 体系与管理设计](./Technical_Design_VectorDB.md#1-schema-体系与管理设计)
- [§2 数据组织与存储引擎](./Technical_Design_VectorDB.md#2-数据组织与存储引擎-storage-engine)
- [§3.1-3.2 HNSWLib Per-Segment 索引](./Technical_Design_VectorDB.md#31-核心问题-索引内部-id--原始数据-offset-的映射)
- [§3.4-3.5 暴力搜索 + 搜索协调器](./Technical_Design_VectorDB.md#34-暴力搜索-两种策略共用)
- [§7 MVCC 多版本控制与查询快照](./Technical_Design_VectorDB.md#7-mvcc-多版本控制与查询快照-snapshot)

**具体实现方案**：

#### 1. Schema 层 (`schema.py`)
- **实现内容**：`FieldType` 枚举、`MetricType` 枚举、`FieldSchema` dataclass、`CollectionSchema` dataclass
- **设计来源**：技术设计 §1.1 核心数据结构定义（代码清单 L22-L69）
- **校验规则**：遵循 §1.1 校验规则表（唯一主键、至少一个 VECTOR 字段、字段名不重复、保留名检查）
- **元数据路径**：遵循 §1.2 元数据持久化（`{data_dir}/meta/{collection_name}/schema.json`）

#### 2. 存储引擎层 (`segment.py`)
- **实现内容**：`Row` dataclass、`SegmentState` 枚举、`SegmentMeta` dataclass、`SegmentData` 类、`CollectionEngine` 类
- **设计来源**：技术设计 §2.1 核心概念: Row 与 Segment（L92-L125）、§2.2 物理数据布局（L127-L207）、§2.3 Collection 级别的 Segment 管理（L218-L246）
- **关键设计决策**：
  - 向量列使用**预分配固定矩阵**（`np.zeros((max_rows, dim), dtype=np.float32)`），原因见 §2.2.2 对比表
  - 删除使用 **bitset 位图软删除**（`mark_deleted`/`is_deleted`），见 §2.2.1 L194-L203
  - `CollectionEngine` 维护唯一 `active_segment`（GROWING）和 `sealed_segments` 列表，见 §2.3 L220-L246

#### 3. LSN 与 MVCC (`lsn.py` + `mvcc.py`)
- **实现内容**：`LSNGenerator` 线程安全计数器、`ReadSnapshot` dataclass、`acquire_snapshot()`、`is_visible()`
- **设计来源**：技术设计 §7 MVCC 多版本控制（L1181-L1230）
- **关键设计**：
  - `LSNGenerator` 使用 `threading.Lock` 保护递增（§7.1 L1186-L1198）
  - `ReadSnapshot` 包含 `read_lsn` + `active_cursor`（§7.2 L1204-L1216）
  - `is_visible` 双重检查：LSN ≤ read_lsn 且未删除（§7.2 L1221-L1229）

#### 4. HNSWLib Per-Segment 索引 (`index_strategy.py` 基础版)
- **实现内容**：`IndexStrategy` 枚举、`SegmentIndex` 类、`brute_force_search()` 函数、`SearchCoordinator` 类（仅 HNSW 路径）
- **设计来源**：技术设计 §3.2 策略 A: HNSWLib Per-Segment 索引（L299-L366）、§3.4 暴力搜索（L543-L583）、§3.5 搜索协调器（L585-L706）
- **关键设计**：
  - 索引内部 ID = segment offset，直接映射，无需额外映射表（§3.1 L273-L283）
  - `SegmentIndex.build_index()` 使用 `hnswlib.Index`，labels 直接使用 `np.arange(n)`（§3.2.1 L303-L341）
  - `brute_force_search` 使用 L2 距离 + `np.argpartition` 取 top_k（§3.4 L548-L583）
  - `SearchCoordinator._search_hnsw()` 实现 Scatter-Gather 归并：Active Segment 暴力搜索 + 各 Sealed/Indexed Segment 分别搜索 + heap merge（§3.2.2 L343-L366、§3.5 L620-L645）

#### 5. Sprint 1 集成测试
- 覆盖：Schema 校验、SegmentData 完整状态流转、位图操作、LSN 线程安全、快照可见性过滤、端到端 Insert → Seal → Build → Search

**交付物**：可在 Python 中直接实例化 `CollectionEngine`，完成 Insert → Seal → CreateIndex(hnsw) → Search 全流程。

---

### Sprint 2：序列化层

**目标**：将 Sprint 1 的内存引擎持久化到磁盘，并能从磁盘恢复。

**技术设计参照**：
- [§4 数据序列化与反序列化](./Technical_Design_VectorDB.md#4-数据序列化与反序列化)
- [§1.2 元数据持久化](./Technical_Design_VectorDB.md#12-元数据持久化)

**具体实现方案**：

#### 1. Schema 序列化 (`serializer.py`)
- **实现内容**：`SchemaSerializer.save()`/`load()`/`list_collections()`
- **设计来源**：技术设计 §1.2 元数据持久化
- **存储位置**：`{data_dir}/meta/{collection_name}/schema.json`

#### 2. Segment 序列化 (`serializer.py`)
- **实现内容**：`SegmentSerializer.flush()`/`load()`
- **设计来源**：技术设计 §4.1 各列的序列化格式选型表（L712-L726）、§4.2 Segment flush 实现（L728-L827）
- **格式选型**：
  - `vectors` → `.npy`（零拷贝 mmap 兼容）
  - `lsns` → `.npy`
  - `pks`(int) → `.npy`(int64)，`pks`(string) → `.pkl`
  - `scalar_columns` → `.pkl`
  - `delete_bitset` → `.bin`（raw bytes）
  - `SegmentMeta` → `.json`
- **磁盘布局**：遵循 §4.3 总体磁盘目录结构（L829-L858）

#### 3. HNSWLib 索引序列化
- **实现内容**：`save_hnsw_index()`/`load_hnsw_index()`
- **设计来源**：§4.1 表（索引 HNSWLib → `.bin`，使用 `index.save_index()` / `index.load_index()`）
- **存储路径**：`{data_dir}/segments/{seg_id}/index/hnsw.bin` + `index_meta.json`（§4.3 L844-L846）

#### 4. CollectionEngine 启动恢复
- **实现内容**：`CollectionEngine.recover_from_disk(base_dir)`
- **恢复流程**：
  1. `SchemaSerializer.load()` 加载 schema.json
  2. 扫描 `segments/` 目录，按 `meta.json` 中 state 恢复各 SegmentData
  3. INDEXED 状态的 segment 加载 `hnsw.bin`
  4. 遍历所有 segment 的 pks 重建 `pk_map`

**交付物**：Engine 可 flush 到磁盘，重启后通过 `recover_from_disk()` 恢复完整状态，数据和索引与重启前完全一致。

---

### Sprint 3：IVF+RaBitQ 全局索引策略

**目标**：实现第二种索引策略，复用现有 `IVFIndex` 和 `TrueRaBitQ`。

**技术设计参照**：
- [§3.3 策略 B: IVF + RaBitQ 全局索引](./Technical_Design_VectorDB.md#33-策略-b-ivf--rabitq-全局索引)
- [§3.5 搜索协调器 IVF 路径](./Technical_Design_VectorDB.md#35-搜索协调器-统一入口)
- [§4.4 全局 IVF+RaBitQ 索引的序列化](./Technical_Design_VectorDB.md#44-全局-ivfrabitq-索引的序列化)

**具体实现方案**：

#### 1. 全局索引数据结构 (`index_strategy.py`)
- **实现内容**：`GlobalIVFIndex` 数据类、`GlobalSearchHit` dataclass
- **设计来源**：技术设计 §3.3.2 L382-L392、§3.3.3 L463-L468
- **关键字段**：`ivf`、`quantizer`、`global_id_map: List[Tuple[str, int]]`、`included_segments: Set[str]`、`build_lsn`

#### 2. GlobalIVFIndexBuilder (`index_strategy.py`)
- **实现内容**：`GlobalIVFIndexBuilder.build()` 静态方法
- **设计来源**：技术设计 §3.3.2 L395-L458
- **构建流程**（4 步）：
  1. 遍历所有 Segment，跳过已删除行，拼接 `all_vectors` + 构建 `global_id_map`（L414-L431）
  2. 构建 IVF 索引（`IVFIndex(nlist, nprobe, metric="l2")` → `ivf.build(all_vectors_np)`）（L435-L443）
  3. 构建 RaBitQ 量化器（`TrueRaBitQ(dimension=dim)` → `train` + `encode`）（L445-L448）
  4. 组装 GlobalIVFIndex（L450-L457）
- **关键映射**：`global_id_map[global_vector_id] = (segment_id, offset)`，解决全局 ID ↔ Segment Offset 映射（§3.1 L285-L294）

#### 3. GlobalIVFSearcher (`index_strategy.py`)
- **实现内容**：`GlobalIVFSearcher.search()` 静态方法
- **设计来源**：技术设计 §3.3.3 L471-L517
- **搜索流程**：
  1. 调用 `ivf.search_with_rabitq()` 得到粗筛结果
  2. 通过 `global_id_map` 映射回 `(seg_id, offset)`
  3. 可见性过滤：`is_deleted` + `lsn ≤ read_lsn`
  4. 返回前 k 个 `GlobalSearchHit`

#### 4. SearchCoordinator IVF 路径补全 (`index_strategy.py`)
- **实现内容**：`SearchCoordinator._search_ivf_rabitq()`
- **设计来源**：技术设计 §3.3.4 搜索链路（L519-L539）、§3.5 L647-L682
- **搜索链路**：
  1. 全局 IVF+RaBitQ 搜索（覆盖 `included_segments` 中的数据）
  2. 未被覆盖的 Segment（Active + Build 后新增 Sealed）走 brute-force 兜底
  3. heap merge 归并所有结果

#### 5. 全局索引序列化 (`serializer.py`)
- **实现内容**：`GlobalIVFIndexSerializer.save()`/`load()`
- **设计来源**：技术设计 §4.4 L862-L944
- **存储路径**：`{data_dir}/global_index/`（`index_meta.json` + `ivf.pkl` + `rabitq.pkl` + `global_id_map.pkl`）
- **关键设计**：IVF 序列化**不重复存储 vectors**，Rerank 时通过 `global_id_map` 从 Segment 读取原始向量（§4.4 设计要点 L947）

**交付物**：可完整调用 `build_global_index()` → `search(strategy=IVF_RABITQ_GLOBAL)` → 验证 recall@10 ≥ 0.8。

---

### Sprint 4：并发控制

**目标**：为写入路径加锁，保证并发安全。

**技术设计参照**：
- [§6 并发控制与锁设计](./Technical_Design_VectorDB.md#6-并发控制与锁设计)

**具体实现方案**：

#### 1. 分段锁 (`concurrency.py`)
- **实现内容**：`StripedLock` 类（64 分段，`get_lock(pk) -> threading.Lock`）
- **设计来源**：技术设计 §6.1.1 L1110-L1126
- **原理**：将 PK 空间 hash 到 64 个桶，每个桶一把互斥锁，避免全局 RWLock 开销

#### 2. WriteExecutor (`concurrency.py`)
- **实现内容**：`WriteExecutor` 类，包含 `insert_row()`、`delete_row()`、`insert_batch()`
- **设计来源**：技术设计 §6.1.2 完整写入流程（L1128-L1160）
- **写入流程**：
  1. `StripedLock(pk)` 加锁
  2. 分配 LSN（`lsn_gen.next()`）
  3. 检查 PK 是否已存在 → Upsert：旧行 `mark_deleted`
  4. `segment_lock` 保护 seal 操作（如 `active_segment.is_full()` 则 seal）
  5. `seg.append_row(row)` → 更新 `pk_map`

#### 3. 读写并发保障
- **设计来源**：技术设计 §6.2 读写并发保障表（L1162-L1176）
- **关键设计**：
  - Search **无锁**读取：SEALED/INDEXED Segment 不可变 + Active Segment 使用快照游标
  - `delete_bitset` 的单 bit 操作是字节级原子的，并发安全
  - CreateIndex(HNSWLib) 无锁（对 Sealed Segment 操作）
  - CreateIndex(IVF+RaBitQ) 无写锁，完成后原子切换 `global_ivf_index` 引用

**交付物**：多线程并发 Insert/Upsert/Delete/Search 无死锁、无数据丢失。

---

### Sprint 5：gRPC 服务层

**目标**：将引擎暴露为标准 gRPC 服务。

**技术设计参照**：
- [§5 gRPC API 协议定义](./Technical_Design_VectorDB.md#5-grpc-api-协议定义)

**具体实现方案**：

#### 1. Protobuf 定义 (`vectordb_service.proto`)
- **实现内容**：完整 Protobuf 定义，包含 10 个 RPC、所有 message 类型
- **设计来源**：技术设计 §5.1 完整 Protobuf 定义（L955-L1092）
- **直接复制**技术设计中的 proto 文件，包含：
  - Service：10 个 RPC（CreateCollection, DropCollection, DescribeCollection, Insert, Upsert, Delete, Search, CreateIndex, DropIndex, DescribeIndex）
  - 列式传输：`FieldData` 使用 `oneof` 区分 int/string/vector，向量扁平化传输（§5.2 L1094-L1102）

#### 2. gRPC Servicer (`servicer.py`)
- **实现内容**：`VectorDBServicer` 类，实现所有 RPC
- **内部持有**：`Dict[str, CollectionEngine]`（多 Collection 管理）、`WriteExecutor`、`LSNGenerator`
- **每个 RPC 的实现逻辑**：
  - `CreateCollection`：解析 FieldSchemaPb → 构建 CollectionSchema → validate → 创建 Engine → 持久化 schema.json
  - `Insert/Upsert`：解析列式 FieldData → 构建 Row 列表 → `WriteExecutor.insert_batch()`
  - `Delete`：解析 pk 列表 → `WriteExecutor.delete_row()`
  - `Search`：`acquire_snapshot()` → `SearchCoordinator.search()` → 构造 SearchResponse
  - `CreateIndex`：根据 `index_type` ("hnsw" / "ivf_rabitq") 路由到对应构建逻辑
  - `DropIndex`：清除索引状态 + 可选删除磁盘文件

#### 3. Server 启动入口 (`main.py`)
- **命令行参数**：`--host`、`--port`、`--data-dir`
- **启动流程**：`recover_from_disk()` → `grpc.server(ThreadPoolExecutor)` → `add_servicer` → `server.start()`

**交付物**：可通过 Python gRPC client 完成 CreateCollection → Insert → CreateIndex → Search 全流程。

---

### Sprint 6：测试完善与收尾

**目标**：补全测试覆盖，完善错误处理和文档。

**实现内容**：

#### 1. 错误处理完善
- Schema 校验异常返回 gRPC 错误码（非 0）
- 不存在的 Collection 操作返回 NOT_FOUND
- Insert 维度不匹配返回 INVALID_ARGUMENT
- Search 前未建索引时降级为 brute-force

#### 2. 性能冒烟测试
- 10 万条 128 维向量写入耗时（应 < 30s 单线程）
- HNSWLib 索引建立耗时（应 < 60s on 10 万条）
- HNSW Search(top_10) QPS（目标 > 1000 QPS 单进程）
- IVF+RaBitQ 全局索引 Build 耗时
- IVF+RaBitQ Search recall@10 验证（目标 ≥ 0.8）

#### 3. 文档与总结
- 更新 `README.md`：新增 Server 模式快速启动说明
- 更新 `CLAUDE.md`：说明新 `vectordb/server/` 模块结构和测试命令

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
| HNSWLib Per-Segment Scatter-Gather 归并在 Segment 数量多时延迟增大 | Phase 1 控制 max_rows=100K，Segment 数量可控 |

---

## 任务依赖关系

```
Sprint 1 (核心存储引擎)
    ↓
Sprint 2 (序列化)   ←── 可并行 ──→   Sprint 3 (IVF+RaBitQ 全局索引)
    ↓                                       ↓
Sprint 4 (并发控制) ←───────────────────────┘
    ↓
Sprint 5 (gRPC 服务)
    ↓
Sprint 6 (测试 & 收尾)
```

> Sprint 2 和 Sprint 3 可并行开发（前提是 Sprint 1 完成）。Sprint 4 依赖 Sprint 1，但不强依赖 Sprint 2/3（可以先做并发再做序列化）。
