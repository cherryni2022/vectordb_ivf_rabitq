"""
VectorDB - A simple vector database with IVF indexing and RaBitQ quantization

MVP Features:
1. Build IVF index for full embeddings
2. Query top-k similar vectors based on query embedding
3. Accelerated search with True RaBitQ (32x compression)

Optimization Features (v0.3.0):
- Vectorized K-Means with NumPy optimizations
- Mini-batch K-Means for large datasets
- Memory-mapped storage for large-scale data
- Parallel batch search with caching
- Optional Numba JIT acceleration
- Optional GPU acceleration (PyTorch CUDA)
"""

__version__ = "0.3.0"

from vectordb.core.vector_db import VectorDB
from vectordb.core.config import IVFIndexConfig, RaBitQConfig, VectorDBConfig
from vectordb.quantization.true_rabitq import TrueRaBitQ, TrueRaBitQStats

# New optimization modules
from vectordb.storage import (
    VectorStorage,
    MemoryMappedVectorStore,
    ChunkedVectorLoader,
)
from vectordb.search import (
    ParallelSearcher,
    SearchCache,
    AdaptiveNProbe,
)

__all__ = [
    # Core
    "VectorDB",
    "IVFIndexConfig",
    "RaBitQConfig",
    "VectorDBConfig",
    "TrueRaBitQ",
    "TrueRaBitQStats",
    # Storage
    "VectorStorage",
    "MemoryMappedVectorStore",
    "ChunkedVectorLoader",
    # Search
    "ParallelSearcher",
    "SearchCache",
    "AdaptiveNProbe",
]

