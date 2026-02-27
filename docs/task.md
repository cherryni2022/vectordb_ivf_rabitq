# 开发任务清单: 向量数据库 Phase 1

> 对应 [plan.md](./plan.md) 的详细任务拆解。每个任务为最小可独立测试的编码单元。
> 所有任务实现方案均参照 [Technical_Design_VectorDB.md](./Technical_Design_VectorDB.md)。

**进度统计**：0 / 70 完成

---

## Sprint 1：核心存储引擎

> **目标**：内存中完整跑通 Insert → Seal → CreateIndex(hnsw) → Search 流程。
> **参照**：技术设计 §1, §2, §3.1-3.2, §3.4-3.5, §7

### 1.1 Schema 层 [`vectordb/server/schema.py`]

> 参照：技术设计 §1.1 核心数据结构定义（L22-L69）、§1.1 校验规则表（L71-L76）

- [ ] **1.1.1** 定义 `FieldType` 枚举（INT / STRING / VECTOR）
  - 实现：`class FieldType(Enum): INT = "int"; STRING = "string"; VECTOR = "vector"`
  - 参照：技术设计 L27-L30

- [ ] **1.1.2** 定义 `MetricType` 枚举（L2 / IP / COSINE）
  - 实现：`class MetricType(Enum): L2 = "L2"; IP = "IP"; COSINE = "COSINE"`
  - 参照：技术设计 L32-L35

- [ ] **1.1.3** 定义 `FieldSchema` dataclass
  - 字段：`name: str`, `dtype: FieldType`, `is_primary: bool`, `dim: Optional[int]`, `metric: Optional[MetricType]`
  - 参照：技术设计 L37-L45

- [ ] **1.1.4** 定义 `CollectionSchema` dataclass
  - 字段：`collection_name: str`, `fields: List[FieldSchema]`, `created_at: Optional[float]`, `version: int`
  - 方法：`primary_field()` → 返回 `is_primary=True` 的字段
  - 方法：`vector_field()` → 返回第一个 `dtype==VECTOR` 的字段
  - 参照：技术设计 L48-L68

- [ ] **1.1.5** `validate_schema(schema: CollectionSchema)` 校验函数
  - 规则 1: 必须有且仅有一个 `is_primary=True` 字段，主键类型仅允许 INT 或 STRING
  - 规则 2: 必须至少有一个 VECTOR 字段，且必须指定 `dim` 和 `metric`
  - 规则 3: 字段名不可重复，不允许 `_lsn`, `_deleted` 等系统保留名
  - 参照：技术设计 §1.1 校验规则表（L71-L76）

### 1.2 存储引擎层 [`vectordb/server/segment.py`]

> 参照：技术设计 §2.1 核心概念（L87-L125）、§2.2 物理数据布局（L127-L207）、§2.3 Segment 管理（L218-L246）

- [ ] **1.2.1** 定义 `SegmentState` 枚举（GROWING / SEALED / INDEXED）
  - 参照：技术设计 L121-L124

- [ ] **1.2.2** 定义 `SegmentMeta` dataclass
  - 字段：`segment_id: str`, `collection_name: str`, `state: SegmentState`, `row_count: int`, `max_rows: int`, `min_lsn: int`, `max_lsn: int`
  - 参照：技术设计 L111-L119

- [ ] **1.2.3** 定义 `Row` dataclass
  - 字段：`pk: Union[int, str]`, `vector: np.ndarray`, `scalars: Dict[str, Union[int, str]]`, `lsn: int`
  - 参照：技术设计 L96-L103

- [ ] **1.2.4** 实现 `SegmentData` 类
  - `__init__`: 预分配 `vectors` numpy 矩阵 `np.zeros((max_rows, dim), dtype=np.float32)`、`pks` list、`scalar_columns` dict（按字段名分开）、`lsns` numpy 数组、`delete_bitset` bytearray、`_write_cursor` 计数器
  - `append_row(row: Row) -> int`: 追加数据到各列，递增 `_write_cursor`，返回 offset
  - `mark_deleted(offset: int)`: 位图软删除 `delete_bitset[offset >> 3] |= (1 << (offset & 0x07))`
  - `is_deleted(offset: int) -> bool`: 检查位图
  - `is_full() -> bool`: `_write_cursor >= max_rows`
  - 参照：技术设计 §2.2.1 L149-L207（完整代码清单）
  - **设计决策**：使用预分配矩阵而非 `np.vstack`，避免 O(N×dim) 拷贝（技术设计 §2.2.2 L209-L216）

- [ ] **1.2.5** 实现 `CollectionEngine` 类
  - 字段：`schema`, `active_segment`（GROWING 状态唯一实例）、`sealed_segments` 列表、`pk_map: Dict[pk, (seg_id, offset)]`
  - `_new_segment()`: 使用 `uuid4().hex[:8]` 生成 seg_id
  - `seal_active_segment()`: 将 `active_segment` 改为 SEALED 状态 → 移入 `sealed_segments` → 创建新 `active_segment`
  - 参照：技术设计 §2.3 L220-L246

### 1.3 LSN 与 MVCC [`vectordb/server/lsn.py` + `vectordb/server/mvcc.py`]

> 参照：技术设计 §7 MVCC 多版本控制与查询快照（L1179-L1258）

- [ ] **1.3.1** `LSNGenerator` 类
  - `__init__(start: int = 0)`: 初始化计数器和 `threading.Lock`
  - `next() -> int`: 加锁递增并返回
  - `current() -> int`: 返回当前值（无锁读）
  - 参照：技术设计 §7.1 L1186-L1198

- [ ] **1.3.2** `ReadSnapshot` dataclass
  - 字段：`read_lsn: int`（能看到的最大 LSN）、`active_cursor: int`（Active Segment 写入游标快照）
  - 参照：技术设计 §7.2 L1204-L1208

- [ ] **1.3.3** `acquire_snapshot(lsn_gen, active_seg) -> ReadSnapshot`
  - 实现：`ReadSnapshot(read_lsn=lsn_gen.current(), active_cursor=active_seg._write_cursor)`
  - 参照：技术设计 §7.2 L1210-L1216

- [ ] **1.3.4** `is_visible(segment, offset, snap) -> bool`
  - 条件 1: `segment.lsns[offset] <= snap.read_lsn`（不是"未来"数据）
  - 条件 2: `not segment.is_deleted(offset)`（未删除）
  - 参照：技术设计 §7.2 L1221-L1229

### 1.4 HNSWLib Per-Segment 索引策略 [`vectordb/server/index_strategy.py`]

> 参照：技术设计 §3.1-3.2（L250-L366）、§3.4-3.5（L543-L706）

- [ ] **1.4.1** 定义 `IndexStrategy` 枚举（HNSW_PER_SEGMENT / IVF_RABITQ_GLOBAL）
  - 参照：技术设计 L254-L259

- [ ] **1.4.2** 实现 `SegmentIndex` 类
  - `__init__(segment: SegmentData)`: 持有 `segment` 引用、`index: Optional[hnswlib.Index]`、`index_params: dict`
  - `build_index(params: dict)`: 
    - 取 `segment.vectors[:row_count]`
    - 创建 `hnswlib.Index(space='l2', dim=dim)`
    - `init_index(max_elements=n, ef_construction=params['ef_construction'], M=params['M'])`
    - `add_items(vectors, ids=np.arange(n))`（**labels = segment offset**）
    - `set_ef(ef_search)`
    - 更新 `segment.meta.state = INDEXED`
  - `search(query, k) -> List[(offset, dist)]`: `index.knn_query(query, k)` → 返回 `(offset, distance)` 列表
  - 参照：技术设计 §3.2.1 L303-L341

- [ ] **1.4.3** 实现 `brute_force_search(segment, query, k, read_lsn)`
  - L2 距离计算：`np.sum((vectors[:n] - query.reshape(1,-1))**2, axis=1)`
  - 过滤：`is_deleted(i)` 或 `lsns[i] > read_lsn` 的行标记为无效
  - 取 top_k：`np.argpartition` + `np.argsort`
  - 返回 `List[(offset, distance)]` 升序排列
  - 参照：技术设计 §3.4 L548-L583（完整代码清单）

- [ ] **1.4.4** 实现 `SearchCoordinator` 类（HNSW 路径，IVF 暂 stub）
  - `__init__(engine: CollectionEngine)`: 持有引擎引用
  - `search(query, k, snap, index_strategy) -> List[dict]`:
    - 根据 `index_strategy` 路由到 `_search_hnsw()` 或 `_search_ivf_rabitq()`（暂 stub）
    - `heapq.nsmallest(k, all_candidates)` 归并
    - `_assemble_results()` 回填 pk 和 scalar 数据
  - `_search_hnsw(query, k, snap)`:
    1. Active Segment → `brute_force_search()`
    2. 各 Sealed/Indexed Segment → `SegmentIndex.search()` 或 `brute_force_search()`
    3. 所有候选合并为 `(dist, seg_id, offset)` 列表
  - `_assemble_results(top_k)`: 根据 `(seg_id, offset)` 从 Segment 读取 pk + scalars
  - `_find_segment(seg_id)`: 在 active + sealed 中查找 Segment
  - 参照：技术设计 §3.5 L587-L706（完整代码清单）、§3.2.2 Scatter-Gather 归并流程图（L343-L364）

### 1.5 Sprint 1 集成测试 [`tests/server/test_sprint1.py`]

- [ ] **1.5.1** 测试 Schema 校验（正常 + 异常 case）
  - 正常：含 1 个 INT 主键 + 1 个 VECTOR 字段 + 1 个 STRING 字段
  - 异常：无主键、多主键、无 VECTOR、重复字段名、VECTOR 缺少 dim

- [ ] **1.5.2** 测试 SegmentData append → seal → is_full 状态流转
  - 写入至 max_rows → 验证 `is_full()` 为 True
  - 验证 `append_row` 返回的 offset 正确递增

- [ ] **1.5.3** 测试 mark_deleted → is_deleted 位图操作
  - 对连续和非连续 offset 做 mark_deleted → 验证 is_deleted 正确
  - 未标记的 offset 应返回 False

- [ ] **1.5.4** 测试 LSNGenerator 线程安全
  - 10 线程并发调用 `next()` 各 1000 次 → 最终 `current()` 应为 10000
  - 无重复 LSN

- [ ] **1.5.5** 测试 ReadSnapshot + is_visible 可见性过滤
  - 写入 row(lsn=1) → 获取 snap(read_lsn=1) → row 可见
  - 写入 row(lsn=2) → 对 snap(read_lsn=1) 不可见
  - 标记删除 → 不可见

- [ ] **1.5.6** 端到端测试：Insert 1000 条 → seal → build_index(hnsw) → search
  - 插入 1000 条随机 128 维向量
  - seal active segment → 验证状态变为 SEALED
  - `SegmentIndex.build_index()` → 状态变为 INDEXED
  - `SearchCoordinator.search()` → top_10 结果与 brute-force numpy 对比一致

---

## Sprint 2：序列化层

> **目标**：Segment 数据持久化到磁盘，重启后可恢复完整状态。
> **参照**：技术设计 §4 数据序列化与反序列化（L710-L947）

### 2.1 Schema 序列化 [`vectordb/server/serializer.py`]

> 参照：技术设计 §1.2 元数据持久化

- [ ] **2.1.1** `SchemaSerializer.save(base_dir, schema)`
  - 写入路径：`{base_dir}/meta/{collection_name}/schema.json`
  - JSON 格式：序列化 `CollectionSchema` 的所有字段
  - 自动创建目录

- [ ] **2.1.2** `SchemaSerializer.load(base_dir, collection_name) -> CollectionSchema`
  - 从 `schema.json` 反序列化重建 `CollectionSchema` 对象
  - 类型还原：字符串 → `FieldType`/`MetricType` 枚举

- [ ] **2.1.3** `SchemaSerializer.list_collections(base_dir) -> List[str]`
  - 扫描 `{base_dir}/meta/` 下所有子目录名

### 2.2 Segment 序列化 [`vectordb/server/serializer.py`]

> 参照：技术设计 §4.1 序列化格式选型表（L712-L726）、§4.2 flush 实现（L728-L827）

- [ ] **2.2.1** `SegmentSerializer.flush(segment: SegmentData)`
  - 文件清单（参照技术设计 §4.3 磁盘目录结构 L831-L858）：
    - `meta.json`: SegmentMeta 序列化（JSON）
    - `vectors.npy`: `np.save(vectors[:row_count])`
    - `lsn.npy`: `np.save(lsns[:row_count])`
    - `pk.npy` / `pk.pkl`: 根据主键类型选择格式（INT → npy，STRING → pkl）
    - `scalars.pkl`: `pickle.dump(scalar_columns)`
    - `delete_bitset.bin`: `write(bytes(delete_bitset))`
  - 参照：技术设计 §4.2 L744-L782（完整代码清单）

- [ ] **2.2.2** `SegmentSerializer.load(seg_id, schema) -> SegmentData`
  - 从磁盘恢复 Segment，重建所有列数据
  - 恢复 `_write_cursor = meta.row_count`
  - 参照：技术设计 §4.2 L784-L826（完整代码清单）

- [ ] **2.2.3** `SegmentSerializer.save_hnsw_index(seg_id, segment_index: SegmentIndex)`
  - 使用 `hnswlib.Index.save_index()` 保存到 `{seg_dir}/index/hnsw.bin`
  - 写入 `{seg_dir}/index/index_meta.json`（索引类型、参数）
  - 参照：技术设计 §4.1 表（索引 HNSWLib → `.bin`）、§4.3 L844-L846

- [ ] **2.2.4** `SegmentSerializer.load_hnsw_index(seg_id, segment) -> SegmentIndex`
  - `hnswlib.Index.load_index()` 恢复索引
  - 读取 `index_meta.json` 恢复参数
  - 返回 `SegmentIndex` 实例

### 2.3 CollectionEngine 启动恢复

- [ ] **2.3.1** `CollectionEngine.recover_from_disk(base_dir)` 类方法
  - 步骤 1: `SchemaSerializer.load()` 加载 schema.json
  - 步骤 2: 扫描 `{base_dir}/segments/` 目录，对每个子目录读取 `meta.json`
  - 步骤 3: 根据 `state` 分类恢复：
    - `GROWING` → 设为 `active_segment`
    - `SEALED` → 加入 `sealed_segments`
    - `INDEXED` → 加入 `sealed_segments` + 加载 `hnsw.bin`
  - 步骤 4: 遍历所有 segment 的 pks 重建 `pk_map`
  - 步骤 5: (可选) 加载 `global/pk_map.pkl` 作为优化

### 2.4 Sprint 2 集成测试 [`tests/server/test_sprint2.py`]

- [ ] **2.4.1** Insert → flush → 清空内存 → load → 验证数据一致性
  - vectors、pks、scalars、lsns、delete_bitset 全部逐一对比

- [ ] **2.4.2** flush → load HNSWLib 索引 → search → 结果与内存搜索一致
  - 内存中搜索结果 vs 磁盘恢复后搜索结果完全相同

- [ ] **2.4.3** `recover_from_disk` 完整测试（含多个 Segment）
  - 创建 3 个 Segment（1 GROWING + 1 SEALED + 1 INDEXED）→ 全部 flush → recover → 验证状态

---

## Sprint 3：IVF+RaBitQ 全局索引策略

> **目标**：实现全局索引构建、搜索和持久化，复用现有 `IVFIndex` 和 `TrueRaBitQ`。
> **参照**：技术设计 §3.3 策略 B（L368-L539）、§4.4 全局索引序列化（L860-L947）

### 3.1 全局索引数据结构 [`vectordb/server/index_strategy.py`]

> 参照：技术设计 §3.3.2 L382-L392

- [ ] **3.1.1** 定义 `GlobalIVFIndex` 数据类
  - 字段：`ivf: Optional[IVFIndex]`, `quantizer: Optional[TrueRaBitQ]`, `global_id_map: List[Tuple[str, int]]`, `included_segments: Set[str]`, `build_lsn: int`, `is_built: bool`
  - 参照：技术设计 L382-L392

- [ ] **3.1.2** 定义 `GlobalSearchHit` dataclass
  - 字段：`segment_id: str`, `offset: int`, `distance: float`
  - 参照：技术设计 L463-L468

### 3.2 GlobalIVFIndexBuilder [`vectordb/server/index_strategy.py`]

> 参照：技术设计 §3.3.2 L395-L458

- [ ] **3.2.1** `GlobalIVFIndexBuilder.build(segments, params, current_lsn) -> GlobalIVFIndex`
  - **Step 1**: 遍历所有 Segment，对每行检查 `is_deleted(offset)`，跳过已删除行，拼接 `all_vectors` + 构建 `global_id_map: [(seg_id, offset), ...]`
    - 参照：技术设计 L414-L431
  - **Step 2**: 构建 IVF → `nlist = min(params['nlist'], max(int(np.sqrt(total_vectors)), 1))`→ `IVFIndex(nlist, nprobe, metric="l2")` → `ivf.build(all_vectors_np)`
    - 参照：技术设计 L435-L443
  - **Step 3**: 构建 RaBitQ → `TrueRaBitQ(dimension=dim)` → `quantizer.train(all_vectors_np)` → `quantizer.encode(all_vectors_np)`
    - 参照：技术设计 L445-L448
  - **Step 4**: 组装 → 设置 `result.ivf`, `result.quantizer`, `result.global_id_map`, `result.build_lsn`, `result.is_built = True`
    - 参照：技术设计 L450-L457

### 3.3 GlobalIVFSearcher [`vectordb/server/index_strategy.py`]

> 参照：技术设计 §3.3.3 L471-L517

- [ ] **3.3.1** `GlobalIVFSearcher.search(global_index, query, k, segments_map, read_lsn, rerank_factor=10)`
  - 调用 `global_index.ivf.search_with_rabitq(query, k * rerank_factor, global_index.quantizer, rerank_factor=rerank_factor)` 得到 `raw_results`
  - 对每个 `(global_id, distance)`：
    - `seg_id, offset = global_index.global_id_map[global_id]`
    - 检查 `seg.is_deleted(offset)` → 跳过
    - 检查 `seg.lsns[offset] > read_lsn` → 跳过
    - 添加到 `hits`
  - 收集到 `k` 个即停止
  - 返回 `List[GlobalSearchHit]`
  - 参照：技术设计 L474-L517（完整代码清单）

### 3.4 SearchCoordinator IVF 路径补全 [`vectordb/server/index_strategy.py`]

> 参照：技术设计 §3.3.4 搜索链路（L519-L539）、§3.5 L647-L682

- [ ] **3.4.1** `SearchCoordinator._search_ivf_rabitq(query, k, snap)`
  - 路径 1: 全局 IVF+RaBitQ 搜索 → `GlobalIVFSearcher.search()`
  - 路径 2: 未被 `included_segments` 覆盖的 Segment 走 brute-force：
    - Active Segment（一定不在 covered 中）
    - Build 后新增的 Sealed Segments（`seg_id NOT IN covered`）
  - 合并所有候选到 `candidates: List[(dist, seg_id, offset)]`
  - 参照：技术设计 L647-L682（完整代码清单）

- [ ] **3.4.2** `CollectionEngine` 新增字段和方法
  - 新增字段：`global_ivf_index: Optional[GlobalIVFIndex] = None`
  - 新增`index_strategy: IndexStrategy` 字段（标记当前集合使用的索引策略）

- [ ] **3.4.3** `CollectionEngine.build_global_index(params) -> GlobalIVFIndex`
  - 收集所有 sealed_segments + active_segment
  - 调用 `GlobalIVFIndexBuilder.build(segments, params, lsn_gen.current())`
  - 原子赋值 `self.global_ivf_index = result`

### 3.5 全局索引序列化 [`vectordb/server/serializer.py`]

> 参照：技术设计 §4.4 全局 IVF+RaBitQ 索引的序列化（L860-L947）

- [ ] **3.5.1** `GlobalIVFIndexSerializer.save(base_dir, global_index)`
  - 存储路径：`{base_dir}/global_index/`
  - 文件清单：
    - `index_meta.json`: `{"index_type": "ivf_rabitq", "nlist", "nprobe", "total_vectors", "build_lsn", "build_timestamp", "included_segments"}`
    - `ivf.pkl`: `pickle.dump({centroids, inverted_lists, vector_to_cluster, nlist, nprobe, metric})`
    - `rabitq.pkl`: `global_index.quantizer.save()`
    - `global_id_map.pkl`: `pickle.dump(global_id_map)`
  - **关键设计**：IVF 序列化不重复存储 vectors（技术设计 L947 设计要点）
  - 参照：技术设计 L868-L905（完整代码清单）

- [ ] **3.5.2** `GlobalIVFIndexSerializer.load(base_dir) -> Optional[GlobalIVFIndex]`
  - 检查 `global_index/index_meta.json` 是否存在
  - 恢复 IVF：反序列化 centroids + inverted_lists → 重建 `IVFIndex`
  - 恢复 RaBitQ：`TrueRaBitQ.load()`
  - 恢复 global_id_map
  - 设置 `is_built = True`
  - 参照：技术设计 L907-L944（完整代码清单）

### 3.6 Sprint 3 集成测试 [`tests/server/test_sprint3.py`]

- [ ] **3.6.1** 构建全局索引 → 搜索 → 验证结果
  - 写入 5000 条 128 维向量 → 全部 seal → `build_global_index()` → search(top_10)
  - 与纯 brute-force 对比 recall@10 ≥ 0.8

- [ ] **3.6.2** Build 后新增数据 → search → 验证新数据通过 brute-force 路径被召回
  - Build → 再写入 100 条新数据（不 seal）→ search → 新数据的 pk 应出现在结果中

- [ ] **3.6.3** 全局索引 save → load → search → 结果一致性验证
  - `GlobalIVFIndexSerializer.save()` → `load()` → search 结果与 save 前完全一致

- [ ] **3.6.4** 软删除一条已纳入全局索引的向量 → search → 验证
  - 删除 top_1 结果的 pk → 再次 search → 该 pk 不在新结果中

---

## Sprint 4：并发控制

> **目标**：写入路径线程安全，读写可并发无死锁。
> **参照**：技术设计 §6 并发控制与锁设计（L1106-L1176）

### 4.1 分段锁 [`vectordb/server/concurrency.py`]

> 参照：技术设计 §6.1.1 分段锁（L1110-L1126）

- [ ] **4.1.1** `StripedLock` 类
  - `__init__(num_stripes: int = 64)`: 创建 64 把 `threading.Lock`
  - `get_lock(pk: Union[int, str]) -> threading.Lock`: `hash(pk) % num_stripes`
  - 参照：技术设计 L1117-L1125

### 4.2 WriteExecutor [`vectordb/server/concurrency.py`]

> 参照：技术设计 §6.1.2 完整写入流程（L1128-L1160）

- [ ] **4.2.1** `WriteExecutor.__init__(engine, lsn_gen, striped_lock, segment_lock)`
  - 持有：`engine: CollectionEngine`, `lsn_gen: LSNGenerator`, `striped_lock: StripedLock`, `segment_lock: threading.Lock`

- [ ] **4.2.2** `WriteExecutor.insert_row(row: Row)`
  - 完整流程（参照技术设计 L1138-L1159）：
    1. `lock = striped_lock.get_lock(row.pk)` → `with lock:`
    2. `row.lsn = lsn_gen.next()`
    3. 检查 PK 存在性：`existing = engine.pk_map.get(row.pk)` → 如有，旧行 `mark_deleted`
    4. `with segment_lock:` → 若 `active_segment.is_full()` 则 `seal_active_segment()`
    5. `seg.append_row(row)` → 更新 `engine.pk_map[row.pk] = (seg_id, offset)`

- [ ] **4.2.3** `WriteExecutor.delete_row(pk)`
  - `StripedLock(pk)` 加锁 → 查找 `pk_map[pk]` → `seg.mark_deleted(offset)` → 删除 `pk_map[pk]`

- [ ] **4.2.4** `WriteExecutor.insert_batch(rows: List[Row])`
  - 遍历调用 `insert_row(row)` 逐条写入
  - 返回成功写入的数量

### 4.3 Sprint 4 并发测试 [`tests/server/test_sprint4.py`]

> 参照：技术设计 §6.2 读写并发保障表（L1162-L1176）

- [ ] **4.3.1** 100 线程并发 Insert（各不同 PK）
  - 使用 `concurrent.futures.ThreadPoolExecutor(max_workers=100)`
  - 每线程写入 100 条不同 PK → 最终 `sum(row_count)` == 10000，无丢失

- [ ] **4.3.2** 并发 Upsert 同一 PK
  - 10 线程向同一 PK 写入不同 vector → 最终 `pk_map` 只有一条，且 vector 值为最后写入的

- [ ] **4.3.3** 写入同时并发 Search（无锁读）
  - 一组线程持续 Insert，另一组线程持续 Search → 无异常崩溃
  - 验证：Search **不加写锁**即可安全执行

- [ ] **4.3.4** 并发 Insert + Delete
  - 先写入 1000 条 → 并发 Delete 其中 500 条 + 同时 Insert 500 条新数据 → 最终状态正确

---

## Sprint 5：gRPC 服务层

> **目标**：将引擎暴露为 gRPC 服务，通过 Python gRPC client 完成端到端验证。
> **参照**：技术设计 §5 gRPC API 协议定义（L951-L1102）

### 5.1 Protobuf 定义 [`vectordb/server/grpc/vectordb_service.proto`]

> 参照：技术设计 §5.1 完整 Protobuf 定义（L955-L1092）

- [ ] **5.1.1** 定义所有 message 类型
  - `StatusResponse`(code, message)
  - `FieldSchemaPb`(name, dtype, is_primary, dim, metric)
  - `CreateCollectionRequest`/`DropCollectionRequest`/`DescribeCollectionRequest`/`DescribeCollectionResponse`
  - `FieldData`(oneof: IntArray, StringArray, VectorArray)、`IntArray`、`StringArray`、`VectorArray`(values, dim)
  - `InsertRequest`/`UpsertRequest`/`DeleteRequest`/`MutationResponse`
  - `SearchRequest`(query_vector, top_k, ef_search, nprobe)、`SearchResponse`、`SearchHit`
  - `CreateIndexRequest`(index_type, params map)、`DropIndexRequest`、`DescribeIndexRequest`/`DescribeIndexResponse`
  - 直接参照技术设计 L955-L1092 的完整 proto 定义

- [ ] **5.1.2** 定义 `VectorDBService`（10 个 RPC）
  - CreateCollection, DropCollection, DescribeCollection
  - Insert, Upsert, Delete
  - Search
  - CreateIndex, DropIndex, DescribeIndex
  - 参照：技术设计 L960-L978

- [ ] **5.1.3** 运行 `python -m grpc_tools.protoc` 生成 Python 代码
  - 生成 `vectordb_pb2.py` + `vectordb_pb2_grpc.py`
  - 确保生成的代码可正常 import

### 5.2 gRPC Servicer 实现 [`vectordb/server/grpc/servicer.py`]

- [ ] **5.2.1** `VectorDBServicer` 类初始化
  - 持有：`engines: Dict[str, CollectionEngine]`、`write_executors: Dict[str, WriteExecutor]`、`lsn_gen: LSNGenerator`、`data_dir: str`

- [ ] **5.2.2** `CreateCollection` RPC 实现
  - 解析 `FieldSchemaPb` → 构建 `CollectionSchema` → `validate_schema()` → 创建 `CollectionEngine` → `SchemaSerializer.save()`
  - 异常返回 StatusResponse(code=1, message=error_detail)

- [ ] **5.2.3** `DropCollection` RPC 实现
  - 删除 `engines[name]` + `write_executors[name]`
  - 可选：删除磁盘目录

- [ ] **5.2.4** `DescribeCollection` RPC 实现
  - 返回 Schema 信息 + `row_count`（所有 segment 总行数）+ `segment_count`

- [ ] **5.2.5** `Insert` RPC 实现
  - 解析列式 `FieldData` → reshape VectorArray(`values` / `dim`) → 构建 Row 列表
  - 参照技术设计 §5.2（L1094-L1102）向量扁平化传输的 reshape 逻辑
  - 调用 `WriteExecutor.insert_batch(rows)`

- [ ] **5.2.6** `Upsert` RPC 实现
  - 同 Insert 逻辑，WriteExecutor 内部通过 `pk_map` 检查自动处理 Upsert

- [ ] **5.2.7** `Delete` RPC 实现
  - 解析 `int_pks` 或 `string_pks` 列表 → 逐条 `WriteExecutor.delete_row(pk)`
  - 返回 `affected_count`

- [ ] **5.2.8** `Search` RPC 实现
  - `acquire_snapshot()` → `SearchCoordinator.search(query, top_k, snap, index_strategy)` → 构造 `SearchResponse`
  - 可选参数：`ef_search`（HNSW）、`nprobe`（IVF）

- [ ] **5.2.9** `CreateIndex` RPC 实现
  - 根据 `index_type`:
    - `"hnsw"`: 遍历所有 SEALED segment → `SegmentIndex.build_index(params)` → `SegmentSerializer.save_hnsw_index()`
    - `"ivf_rabitq"`: `CollectionEngine.build_global_index(params)` → `GlobalIVFIndexSerializer.save()`

- [ ] **5.2.10** `DropIndex` RPC 实现
  - 清除索引内存状态：INDEXED → SEALED，`global_ivf_index = None`
  - 可选：删除磁盘索引文件

### 5.3 gRPC Server 启动入口 [`vectordb/server/main.py`]

- [ ] **5.3.1** 解析命令行参数
  - `--host` (默认 `0.0.0.0`)
  - `--port` (默认 `50051`)
  - `--data-dir` (默认 `./vectordb_data`)

- [ ] **5.3.2** 启动时恢复状态
  - 调用 `recover_from_disk(data_dir)` 加载所有已有 Collection
  - 为每个 Collection 创建 `WriteExecutor`

- [ ] **5.3.3** 启动 gRPC Server
  - `grpc.server(ThreadPoolExecutor(max_workers=10))` → `add_VectorDBServicer_to_server()` → `server.start()` → `server.wait_for_termination()`

### 5.4 安装依赖

- [ ] **5.4.1** `uv add grpcio grpcio-tools protobuf`
  - 更新 `pyproject.toml`

### 5.5 Sprint 5 gRPC 端到端测试 [`tests/server/test_sprint5_grpc.py`]

- [ ] **5.5.1** 完整 HNSW 流程测试
  - 启动本地 gRPC server → CreateCollection → Insert 1000 条 → CreateIndex(hnsw) → Search → 验证结果

- [ ] **5.5.2** Delete + Search 测试
  - gRPC Delete 若干 PK → Search → 被删除 pk 不在结果中

- [ ] **5.5.3** 重启恢复测试
  - 重启 server（recover_from_disk）→ Search → 结果与重启前一致

- [ ] **5.5.4** IVF+RaBitQ 测试
  - CreateIndex(ivf_rabitq) → Search → recall@10 ≥ 0.8

---

## Sprint 6：测试完善与收尾

> **目标**：补全测试、错误处理和文档。

### 6.1 错误处理完善

- [ ] **6.1.1** Schema 校验异常返回 gRPC 错误码
  - `validate_schema()` 抛出 `ValueError` → Servicer 捕获 → `StatusResponse(code=3, message=detail)` (INVALID_ARGUMENT)

- [ ] **6.1.2** 对不存在的 Collection 操作返回 NOT_FOUND
  - `StatusResponse(code=5, message="Collection not found")`

- [ ] **6.1.3** Insert 维度不匹配返回 INVALID_ARGUMENT
  - 检查 `vector.shape[0] != schema.vector_field().dim` → 返回错误

- [ ] **6.1.4** Search 前未建索引时降级为 brute-force
  - 无 INDEXED segment 且无 `global_ivf_index` → 全部走 `brute_force_search` → 正常返回结果

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
Sprint 1 (存储引擎) ──────────────────── 18 个任务
    ↓
Sprint 2 (序列化)   ←── 可并行 ──→   Sprint 3 (IVF+RaBitQ 全局索引)
  10 个任务                            11 个任务
    ↓                                       ↓
Sprint 4 (并发控制) ←───────────────────────┘
  8 个任务
    ↓
Sprint 5 (gRPC 服务)
  17 个任务
    ↓
Sprint 6 (测试 & 收尾)
  8 个任务
```

> Sprint 2 和 Sprint 3 可并行开发（前提是 Sprint 1 完成）。
> Sprint 4 依赖 Sprint 1，但不强依赖 Sprint 2/3（可以先做并发再做序列化）。

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

---

## 技术设计文档交叉引用索引

| 模块 | 技术设计章节 | 行号范围 |
|------|-------------|---------|
| Schema (FieldType, CollectionSchema) | §1.1 核心数据结构定义 | L22-L69 |
| Schema 校验规则 | §1.1 校验规则表 | L71-L76 |
| Schema 持久化 | §1.2 元数据持久化 | L78-L82 |
| Row / SegmentMeta / SegmentState | §2.1 核心概念 | L87-L125 |
| SegmentData 列式存储 | §2.2 物理数据布局 | L127-L207 |
| 预分配 vs 动态追加 | §2.2.2 设计决策 | L209-L216 |
| CollectionEngine | §2.3 Segment 管理 | L218-L246 |
| 索引 ID ↔ Offset 映射 | §3.1 核心问题 | L269-L295 |
| HNSWLib Per-Segment | §3.2 策略 A | L299-L366 |
| IVF+RaBitQ 全局索引 | §3.3 策略 B | L368-L539 |
| brute_force_search | §3.4 暴力搜索 | L543-L583 |
| SearchCoordinator | §3.5 搜索协调器 | L585-L706 |
| 序列化格式选型 | §4.1 格式选型表 | L712-L726 |
| Segment flush/load | §4.2 实现代码 | L728-L827 |
| 磁盘目录结构 | §4.3 总体结构 | L829-L858 |
| 全局索引序列化 | §4.4 实现代码 | L860-L947 |
| gRPC Protobuf 定义 | §5.1 完整定义 | L955-L1092 |
| 向量扁平化传输 | §5.2 设计理由 | L1094-L1102 |
| StripedLock 分段锁 | §6.1.1 | L1110-L1126 |
| WriteExecutor 写入流程 | §6.1.2 | L1128-L1160 |
| 读写并发保障 | §6.2 保障表 | L1162-L1176 |
| LSNGenerator | §7.1 | L1186-L1198 |
| ReadSnapshot + acquire_snapshot | §7.2 | L1204-L1216 |
| is_visible 可见性判定 | §7.2 | L1221-L1229 |
| CompactionPolicy (Phase 2 预设计) | §7.3 | L1232-L1258 |
