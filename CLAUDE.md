# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-machine vector database service implementing IVF (Inverted File) indexing and HNSW (Hierarchical Navigable Small World) with **True RaBitQ** (Randomized Binary Quantization) for efficient and accelerated similarity search.

**Version**: 1.0.0 (Phase 1: Single-machine vector database service)

### Architecture Overview

The system is organized into four layers:

1. **Access Layer** (`vectordb/server/`) - gRPC service for network communication
2. **Execution Layer** (`vectordb/coordinator/`) - Schema management, transaction control, LSN generation
3. **Storage Layer** (`vectordb/storage/`) - Segment-based storage with MemTable and immutable segments
4. **Index Layer** (`vectordb/index/`) - Dual index strategies (HNSWLib Per-Segment or IVF+RaBitQ Global)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client (Python / Go / etc.)                 │
└─────────────────────────────────────────────────────────────────┘
                              | gRPC
┌─────────────────────────────────────────────────────────────────┐
│              Access Layer (gRPC Server)                          │
└─────────────────────────────────────────────────────────────────┘
                              |
┌─────────────────────────────────────────────────────────────────┐
│       Execution Layer (Schema, Transaction, LSN, Index Mgmt)     │
└─────────────────────────────────────────────────────────────────┘
                              |
┌─────────────────────────────────────────────────────────────────┐
│       Storage Layer (MemTable, Segments, Bitset, KV Map)        │
└─────────────────────────────────────────────────────────────────┘
                              |
┌─────────────────────────────────────────────────────────────────┐
│       Index Layer (HNSWLib Per-Segment / IVF+RaBitQ Global)     │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Index Types (`vectordb/index/`)

**IVF Index** (`ivf_index.py`) - Partitions vectors into clusters using k-means clustering. During search, only a subset of closest clusters (nprobe) are examined.

**HNSW Index** (`hnsw_index.py` and `hnswlib_index.py`) - Hierarchical Navigable Small World graph-based index. Pure Python (educational) and HNSWLib C++ binding (production).

**True RaBitQ** (`quantization/true_rabitq.py`) - True RaBitQ implementation based on SIGMOD'24 paper:
   - Random orthogonal transformation for even information distribution
   - 1-bit sign quantization per dimension (32x compression)
   - Hamming distance approximation for L2/IP distances
   - Two-stage search: coarse RaBitQ filtering + exact reranking

### 2. Storage Engine (`vectordb/storage/`)

**Segment-based Storage** - Data organized into Segments with three states:
- `GROWING` - Active MemTable receiving writes
- `SEALED` - No longer receiving writes, not yet indexed
- `INDEXED` - Has vector index built (HNSWLib or part of global IVF+RaBitQ)

**SegmentData** - Columnar storage with pre-allocated arrays:
- `vectors` - `np.ndarray` (max_rows, dim) float32
- `pks` - Primary key list (int or string)
- `scalars` - Scalar columns as dict of lists
- `lsns` - LSN (Log Sequence Number) per row
- `delete_bitset` - Bitset for soft deletion

### 3. Dual Index Strategies

**Strategy A: HNSWLib Per-Segment**
- Each Segment gets its own HNSWLib index after being Sealed
- Scatter-Gather search across all segments
- Best for: Real-time writes + low latency queries

**Strategy B: IVF + RaBitQ Global**
- Single global index built on demand across all segments
- RaBitQ provides 32x compression
- Two-stage search: RaBitQ filtering + exact rerank
- Best for: Batch import + high throughput/high compression

### 4. MVCC & Concurrency (`vectordb/coordinator/`)

**LSNGenerator** - Atomic counter for assigning global Log Sequence Numbers

**StripedLock** - Sharded locking for PK uniqueness (64 stripes by default)

**ReadSnapshot** - Query-level snapshot with read_lsn and active_cursor

**Soft Deletion** - Uses bitset to mark rows deleted (no physical removal)

## Data Types & Schema

### FieldType
- `INT` - int64
- `STRING` - Variable-length UTF-8
- `VECTOR` - float32 fixed-length vector with specified dimension

### MetricType
- `L2` - Euclidean distance
- `IP` - Inner Product
- `COSINE` - Cosine similarity

### FieldSchema
```python
@dataclass
class FieldSchema:
    name: str
    dtype: FieldType
    is_primary: bool = False
    dim: Optional[int] = None      # VECTOR only
    metric: Optional[MetricType] = None  # VECTOR only
```

## Environment Setup

```bash
# Install Python 3.12 (if not already installed)
uv python install 3.12

# Create virtual environment
uv venv .venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Sync dependencies (including dev dependencies)
uv sync --dev
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_ivf_index.py -v
pytest tests/test_hnsw.py -v
pytest tests/test_hnswlib.py -v
pytest tests/test_true_rabitq.py -v
pytest tests/test_rabitq.py -v
pytest tests/test_vector_db.py -v

# Run Phase 1 tests (schema, segments, storage, MVCC)
pytest tests/test_schema.py -v
pytest tests/test_segment.py -v
pytest tests/test_storage_engine.py -v
pytest tests/test_mvcc.py -v
pytest tests/test_concurrency.py -v

# Run optimization tests
python tests/test_phase1_optimizations.py
python tests/test_all_optimizations.py

# Run benchmarks
python benchmarks/rabitq_benchmark.py
python benchmarks/benchmark_ivf_vs_hnsw.py --skip-python-hnsw

# Run example usage script
python example_usage.py

# Start gRPC server
python -m vectordb.server.main --port 50051
```

## Project Rules (from .agent/rules.md)

### Git Commit Rule
Every code or documentation change must be committed to git. Use conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

```bash
git add <changed-files>
git commit -m "type: brief description"
```

### Testing Rule
Every new feature must include unit tests and benchmark comparisons. Create tests in `tests/test_<feature>.py` and benchmarks in `benchmarks/`.

### Dependency Management Rule
Third-party dependencies must be added to `pyproject.toml`. Use appropriate dependency groups:
- `dependencies` - Core required packages
- `dev` - Development/testing tools
- `jit` - JIT acceleration (numba)
- `gpu` - GPU acceleration (torch)
- `hnsw` - HNSWLib C++ binding
- `grpc` - gRPC server dependencies
- `accel` - All acceleration features (numba + torch)

## gRPC API

### Service Methods

**Collection Management**
- `CreateCollection` - Define and create a collection with schema
- `DropCollection` - Delete collection and release resources
- `DescribeCollection` - Get collection metadata

**Data Operations (DML)**
- `Insert` - Batch insert rows with schema validation
- `Upsert` - Insert or update if PK exists
- `Delete` - Soft delete rows by primary key

**Query (DQL)**
- `Search` - KNN/ANN search with top_k

**Index Management**
- `CreateIndex` - Build index (type="hnsw" or "ivf_rabitq")
- `DropIndex` - Remove index
- `DescribeIndex` - Get index metadata

## Write Path Flow

1. Client sends `Insert(List<Rows>)` request
2. Request Validator checks Schema for data types and dimensions
3. Version Manager generates LSN
4. StripedLock ensures PK uniqueness (Upsert marks old row deleted)
5. Data appended to Active MemTable
6. PK -> (segment_id, offset) mapping updated
7. Segment sealed if full, new Active Segment created

## Search Path Flow

### HNSWLib Per-Segment
```
Active Segment (GROWING) → brute-force search
Sealed Indexed Segments → HNSWLib search per segment
Sealed Non-Indexed Segments → brute-force search
Coordinator → Scatter-Gather merge → filter deleted/LSN → return top_k
```

### IVF + RaBitQ Global
```
Global IVF+RaBitQ Index → two-stage search → map back to (seg_id, offset)
Uncovered Segments → brute-force search
Coordinator → merge all results → return top_k
```

## Configuration Classes

### Schema Configuration
```python
FieldSchema(
    name="embedding",
    dtype=FieldType.VECTOR,
    dim=128,
    metric=MetricType.L2
)
```

### IVFIndexConfig
```python
IVFIndexConfig(
    nlist=100,                # Number of clusters
    nprobe=10,                # Clusters to search
    metric="l2"
)
```

### HNSWIndexConfig
```python
HNSWIndexConfig(
    M=16,                    # Connections per node
    ef_construction=200,      # Build candidate list size
    ef_search=50,            # Search candidate list size
    metric="l2"
)
```

### SegmentConfig
```python
SegmentConfig(
    max_rows=100_000,        # Rows per segment
    seal_threshold=100_000,   # Auto-seal threshold
)
```

### VectorDBConfig (Phase 1)
```python
VectorDBConfig(
    data_dir="./data",
    index_strategy="hnsw",    # "hnsw" or "ivf_rabitq"
    segment_max_rows=100_000,
)
```

## True RaBitQ - Key Concepts

### Compression
- 32x compression: Each dimension compressed from 32-bit float to 1-bit sign
- Storage: `ceil(dimension/8)` bytes for binary codes + 4 bytes for norm
- For 128-dim vectors: 512 bytes → ~20 bytes (including norm)

### Distance Approximation
```
cos(θ) ≈ 1 - 2 * hamming_distance / dimension
||x - y||² ≈ ||x||² + ||y||² - 2 * ||x|| * ||y|| * cos(θ)
```

### Two-Stage Search
1. **Coarse stage**: Use Hamming distance on binary codes for fast candidate selection
2. **Rerank stage**: Compute exact L2 distances for top-k * rerank_factor candidates
3. **Return**: Final top-k results with exact distances

## Important Implementation Details

### Distance Computation
- L2 distance: Uses efficient matrix computation `||q||^2 + ||p||^2 - 2*q*p`
- Inner product: Returns `-dot(q, p)` (negative because search minimizes distance)
- All L2 distances are clipped to non-negative values to handle floating point errors

### K-means Clustering (IVF)
- Uses k-means++ initialization for better cluster quality
- Handles edge cases: duplicate vectors (all distances zero), empty clusters (re-initialize from largest cluster)
- Mini-batch K-Means available for large-scale datasets (> 100K vectors)

### Persistence

**Schema**: `{data_dir}/meta/{collection_name}/schema.json`

**Segments**: `{data_dir}/segments/{segment_id}/`
- `meta.json` - Segment metadata
- `vectors.npy` - Vector column
- `pk.npy` or `pk.pkl` - Primary key column
- `scalars.pkl` - Scalar columns
- `lsn.npy` - LSN column
- `delete_bitset.bin` - Soft deletion bitset
- `index/hnsw.bin` - HNSWLib index (HNSW strategy only)

**Global Index** (IVF+RaBitQ only): `{data_dir}/global_index/`
- `index_meta.json` - Index metadata
- `ivf.pkl` - IVF centroids + inverted lists
- `rabitq.pkl` - RaBitQ quantizer
- `global_id_map.pkl` - Global ID to (segment_id, offset) mapping

### MVCC Visibility Rules
A row is visible to a query with `read_lsn` if:
1. `row.lsn <= read_lsn` (not from future)
2. `!is_deleted(offset)` (not soft-deleted)
3. `offset < active_cursor` (in active segment)

### Concurrency Model

| Operation | Lock Strategy |
|-----------|---------------|
| Insert/Upsert | `StripedLock(pk)` + `segment_lock` |
| Delete | `StripedLock(pk)` |
| Search | Lock-free (read snapshot) |
| CreateIndex (HNSWLib) | Lock-free (operates on SEALED segments) |
| CreateIndex (IVF+RaBitQ) | Lock-free build, atomic reference swap |

## Performance Tuning

### IVF Parameters
- `nlist`: Number of clusters (default 100, use sqrt(N) heuristic)
- `nprobe`: Clusters to search (nlist/10 for 90%+ recall)
- `rerank_factor`: Candidates for reranking (10 for good quality)

### HNSW Parameters
- `M`: Connections per node (16 default, higher = better recall, more memory)
- `ef_construction`: Build candidates (200 default)
- `ef_search`: Search candidates (50-100 for good quality)

### Segment Tuning
- `max_rows`: Rows per segment (100,000 default)
- Larger segments = fewer index files, longer seal time
- Smaller segments = more granular indexing, more files

### Compression Trade-offs
- True RaBitQ: 32x compression with ~5-10% recall loss
- Best for: Large datasets where memory is a concern
- Not suitable for: High-precision applications requiring exact nearest neighbors

## Disk Directory Structure

```
{data_dir}/
├── meta/
│   └── {collection_name}/
│       └── schema.json
├── segments/
│   └── {segment_id}/
│       ├── meta.json
│       ├── vectors.npy
│       ├── pk.npy or pk.pkl
│       ├── scalars.pkl
│       ├── lsn.npy
│       ├── delete_bitset.bin
│       └── index/ (HNSW strategy only)
│           ├── hnsw.bin
│           └── index_meta.json
├── global_index/ (IVF+RaBitQ strategy only)
│   ├── index_meta.json
│   ├── ivf.pkl
│   ├── rabitq.pkl
│   └── global_id_map.pkl
└── global/
    └── pk_map.pkl
```

## Roadmap

- **Phase 1** (Current): Single-machine vector database service with gRPC, schema management, MVCC, dual index strategies
- **Phase 2**: Enterprise features - Hybrid search, WAL/mmap persistence, background compaction, segment merge
- **Phase 3**: Distributed - Sharding, replication, consistency
