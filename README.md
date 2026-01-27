# VectorDB - IVF + RaBitQ Vector Database

A lightweight, single-machine vector database MVP implementing IVF (Inverted File) indexing with optional RaBitQ (Randomized Binary/Product Quantization) for efficient similarity search.

## Features

- **IVF Indexing**: Partitions vectors into clusters using k-means for fast approximate search
- **RaBitQ Quantization**: Product Quantization with optional binary encoding for memory efficiency
- **Top-k Search**: Retrieve the k most similar vectors for a query embedding
- **Metadata Support**: Store and retrieve metadata alongside vectors
- **Persistence**: Save and load the database from disk
- **Batch Operations**: Support for batch adding and searching vectors

## Architecture

```
VectorDB
├── IVF Index (Inverted File)
│   ├── K-means clustering (nlist clusters)
│   ├── Inverted lists (vector indices per cluster)
│   └── Search with nprobe clusters
└── RaBitQ Quantization (optional)
    ├── Product Quantization (splits vectors into sub-vectors)
    ├── Codebook for each sub-quantizer
    └── Asymmetric distance computation
```

## Installation

### Prerequisites

- Python 3.12+ (recommended: Anaconda Python 3.12)

### Setup Steps

```bash
# Clone the repository
git clone <repository-url>
cd study_vectordb

# Create virtual environment using Anaconda Python 3.12
/Users/niwen/anaconda3/bin/python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package with dev dependencies (editable mode)
pip install -e ".[dev]"
```

### Alternative: Using uv (recommended)

```bash
# Install Python 3.12 (if not already installed)
uv python install 3.12

# Create virtual environment
uv venv .venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Sync dependencies
uv sync --dev
```

## Quick Start

```python
import numpy as np
from vectordb import VectorDB

# Create an in-memory database
db = VectorDB(use_memory_storage=True)

# Generate some random embeddings
vectors = np.random.randn(1000, 128).astype(np.float32)

# Add vectors to the database
db.add_vectors(vectors)

# Build the IVF index
db.build()

# Search for similar vectors
query = vectors[0]
results = db.search(query, k=5)

print(f"Found {len(results.ids)} similar vectors")
for id_, dist in zip(results.ids, results.distances):
    print(f"  ID: {id_}, Distance: {dist:.4f}")
```

## Configuration

### Basic Configuration

```python
from vectordb import VectorDB, IVFIndexConfig, RaBitQConfig, VectorDBConfig

config = VectorDBConfig(
    ivf=IVFIndexConfig(
        nlist=100,      # Number of clusters
        nprobe=10,      # Clusters to search
        metric="l2"     # "l2" or "ip" (inner product)
    ),
    raq=RaBitQConfig(
        nsubq=8,        # Number of sub-quantizers
        nbits=8,        # Bits per sub-quantizer
        use_rabitq=True # Enable RaBitQ
    ),
    dimension=128,
    use_quantization=True
)

db = VectorDB(config=config)
```

### Convenience Function

```python
from vectordb.core.vector_db import create_vector_db

db = create_vector_db(
    dimension=128,
    nlist=50,
    nprobe=10,
    use_quantization=True
)
```

## API Reference

### VectorDB

| Method | Description |
|--------|-------------|
| `add_vectors(vectors, ids, metadata)` | Add vectors to the database |
| `build(use_quantization)` | Build IVF index |
| `search(query, k, nprobe)` | Search for top-k similar vectors |
| `search_batch(queries, k, nprobe)` | Batch search for multiple queries |
| `get_vector(vector_id)` | Retrieve a single vector |
| `save(path)` | Save database to disk |
| `load(path)` | Load database from disk |
| `get_stats()` | Get database statistics |

### SearchResults

| Attribute | Description |
|-----------|-------------|
| `ids` | List of vector IDs |
| `distances` | List of distances |
| `metadata` | List of metadata dicts |

## Key Concepts

### IVF (Inverted File Index)

IVF partitions vectors into clusters using k-means:
- **nlist**: Number of clusters - more clusters = smaller clusters per search
- **nprobe**: Number of clusters searched during query - higher = better recall, slower
- Trade-off: Memory vs. Speed vs. Recall

### RaBitQ (Randomized Binary/Product Quantization)

RaBitQ compresses vectors for memory efficiency:
- **nsubq**: Number of sub-vectors for product quantization
- **nbits**: Bits per sub-quantizer (codebook size = 2^nbits)
- **use_rabitq**: Use RaBitQ vs standard PQ

Compression ratio: `compression_ratio = (nsubq * nbits) / (dimension * 32)`

## Performance Tips

1. **Choose nlist based on dataset size**: ~sqrt(n_vectors) is a good starting point
2. **Tune nprobe for recall vs speed**: Start with nprobe = nlist // 10
3. **Use quantization for large datasets**: Can reduce memory by 90%+
4. **Batch search when possible**: More efficient than individual searches

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage
- Metadata handling
- Batch search
- Custom configuration
- Nprobe tuning
- Quantization effects
- Persistence
- And more...

Run examples:
```bash
python example_usage.py
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_vector_db.py

# Run with verbose output
pytest tests/ -v
```

## Roadmap

Future enhancements:
- [ ] GPU support for indexing and search
- [ ] Additional distance metrics (cosine, etc.)
- [ ] Index pruning and optimization
- [ ] Distributed version
- [ ] HTTP API server
- [ ] Streaming ingestion

## References

- [IVF Indexing](https://gaocegege.com/Blog/genai/hnsw) - Reference article
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [Product Quantization](https://lear.inrialpes.fr/pubs/2011/JegouAM11/JegouAM11.pdf) - Original PQ paper

## License

MIT License
