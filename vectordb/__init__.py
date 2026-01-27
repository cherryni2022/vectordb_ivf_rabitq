"""
VectorDB - A simple vector database with IVF indexing and RaBitQ quantization

MVP Features:
1. Build IVF index for full embeddings
2. Query top-k similar vectors based on query embedding
3. Accelerated search with True RaBitQ (32x compression)
"""

__version__ = "0.2.0"

from vectordb.core.vector_db import VectorDB
from vectordb.core.config import IVFIndexConfig, RaBitQConfig, VectorDBConfig
from vectordb.quantization.true_rabitq import TrueRaBitQ, TrueRaBitQStats

__all__ = [
    "VectorDB",
    "IVFIndexConfig",
    "RaBitQConfig",
    "VectorDBConfig",
    "TrueRaBitQ",
    "TrueRaBitQStats",
]
