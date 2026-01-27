"""
VectorDB - Main class combining IVF indexing and RaBitQ quantization

This is the main entry point for the vector database MVP.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

from vectordb.core.config import VectorDBConfig, IVFIndexConfig, RaBitQConfig
from vectordb.index.ivf_index import IVFIndex, IVFIndexStats
from vectordb.quantization.rabitq import RaBitQ, RaBitQStats
from vectordb.quantization.true_rabitq import TrueRaBitQ, TrueRaBitQStats
from vectordb.storage.vector_storage import VectorStorage, InMemoryVectorStorage


@dataclass
class SearchResults:
    """Results from a search operation"""
    ids: List[Any]
    distances: List[float]
    metadata: List[Dict[str, Any]]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for easy serialization"""
        return [
            {"id": id_, "distance": dist, "metadata": meta}
            for id_, dist, meta in zip(self.ids, self.distances, self.metadata)
        ]


class VectorDB:
    """
    Vector Database with IVF indexing and optional RaBitQ quantization

    MVP Features:
    1. Build IVF index for full embeddings
    2. Query top-k similar vectors based on query embedding
    3. Optional RaBitQ quantization for memory efficiency

    Parameters:
        config: VectorDBConfig containing IVF and RaBitQ settings
        storage_path: Path to store persisted data
        use_memory_storage: Use in-memory storage (no persistence)
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        storage_path: str = "./vectordb_data",
        use_memory_storage: bool = False
    ):
        self.config = config or VectorDBConfig()

        # Initialize storage
        if use_memory_storage:
            self.storage = InMemoryVectorStorage()
        else:
            self.storage = VectorStorage(base_path=storage_path)

        # Initialize components
        self.ivf_index = IVFIndex(
            nlist=self.config.ivf.nlist,
            nprobe=self.config.ivf.nprobe,
            metric=self.config.ivf.metric
        )
        self.quantizer: Optional[RaBitQ] = None
        self.true_rabitq: Optional[TrueRaBitQ] = None  # New: True RaBitQ for accelerated search

        # State
        self.is_built = False
        self.dimension: Optional[int] = None

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add vectors to the database

        Note: After adding new vectors, you should call build() to rebuild the index.

        Args:
            vectors: Shape (n_vectors, dimension)
            ids: Optional list of IDs for each vector
            metadata: Optional metadata for each vector
        """
        n_vectors, dimension = vectors.shape

        # Validate dimension
        if self.dimension is None:
            self.dimension = dimension
        elif self.dimension != dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {dimension}")

        # Add to storage
        self.storage.add_vectors(vectors, ids, metadata)

        # Note: We need to rebuild the index after adding vectors
        self.is_built = False

    def build(self, use_quantization: Optional[bool] = None, use_true_rabitq: bool = True) -> None:
        """
        Build the IVF index from all stored vectors

        Optionally builds quantization for memory efficiency and accelerated search.

        Args:
            use_quantization: Override config setting for old-style PQ quantization
            use_true_rabitq: Enable True RaBitQ for accelerated search (default: True)
        """
        vectors = self.storage.get_all_vectors()
        if vectors is None or len(vectors) == 0:
            raise ValueError("No vectors in database. Use add_vectors() first.")

        use_quant = use_quantization if use_quantization is not None else self.config.use_quantization

        # Build IVF index
        self.ivf_index.build(vectors)

        # Build old-style PQ quantizer if enabled (for backward compatibility)
        if use_quant:
            self.quantizer = RaBitQ(
                dimension=self.dimension,
                nsubq=self.config.raq.nsubq,
                nbits=self.config.raq.nbits,
                use_binary=not self.config.raq.use_rabitq,
                random_seed=self.config.raq.random_seed
            )
            self.quantizer.train(vectors)
            self.quantizer.encode(vectors)  # Important: encode all vectors

        # Build True RaBitQ for accelerated search
        if use_true_rabitq and self.dimension is not None:
            self.true_rabitq = TrueRaBitQ(
                dimension=self.dimension,
                random_seed=self.config.raq.random_seed
            )
            self.true_rabitq.train(vectors)
            self.true_rabitq.encode(vectors)  # Encode all vectors for fast search

        self.is_built = True

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        accelerated: bool = True,
        rerank_factor: int = 10
    ) -> SearchResults:
        """
        Search for top-k similar vectors

        Args:
            query: Query vector, shape (dimension,)
            k: Number of results to return
            nprobe: Override number of clusters to probe
            accelerated: Use RaBitQ acceleration if available (default: True)
            rerank_factor: For accelerated search, multiply k by this for candidate selection

        Returns:
            SearchResults containing IDs, distances, and metadata
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # Override nprobe if provided
        if nprobe is not None:
            self.ivf_index.nprobe = nprobe

        # Choose search method
        if accelerated and self.true_rabitq is not None:
            # Use IVF + RaBitQ accelerated search
            results = self.ivf_index.search_with_rabitq(
                query, k, self.true_rabitq, rerank_factor
            )
        else:
            # Fall back to standard IVF search
            results = self.ivf_index.search(query, k)

        # Map indices back to IDs and get metadata
        ids = []
        distances = []
        metadata_list = []

        for vector_idx, distance in results:
            vector_id = self.storage.index_to_id.get(vector_idx)
            if vector_id is not None:
                ids.append(vector_id)
                distances.append(float(distance))
                metadata_list.append(self.storage.metadata.get(vector_id, {}))

        return SearchResults(ids=ids, distances=distances, metadata=metadata_list)

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None
    ) -> List[SearchResults]:
        """
        Batch search for multiple queries

        Args:
            queries: Query vectors, shape (n_queries, dimension)
            k: Number of results per query
            nprobe: Override number of clusters to probe

        Returns:
            List of SearchResults, one per query
        """
        return [self.search(q, k, nprobe) for q in queries]

    def get_vector(self, vector_id: Any) -> Optional[np.ndarray]:
        """Get a vector by its ID"""
        return self.storage.get_vector(vector_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary containing statistics about the database
        """
        storage_stats = self.storage.get_stats()
        ivf_stats = self.ivf_index.get_stats() if self.is_built else IVFIndexStats()
        raq_stats = self.quantizer.get_stats() if self.quantizer else RaBitQStats()
        true_raq_stats = self.true_rabitq.get_stats() if self.true_rabitq else None

        return {
            "total_vectors": storage_stats.total_vectors,
            "dimension": self.dimension,
            "is_built": self.is_built,
            "use_quantization": self.config.use_quantization,
            "accelerated_search_enabled": self.true_rabitq is not None,
            "storage": {
                "total_bytes": storage_stats.total_bytes,
                "metadata_count": storage_stats.metadata_count,
            },
            "ivf_index": {
                "num_clusters": ivf_stats.num_clusters,
                "avg_vectors_per_cluster": ivf_stats.avg_vectors_per_cluster,
                "nlist": self.config.ivf.nlist,
                "nprobe": self.config.ivf.nprobe,
                "metric": self.config.ivf.metric,
            },
            "quantization": {
                "nsubq": raq_stats.nsubq,
                "nbits": raq_stats.nbits,
                "compression_ratio": raq_stats.compression_ratio,
            } if self.quantizer else None,
            "true_rabitq": {
                "bits_per_vector": true_raq_stats.bits_per_vector,
                "bytes_per_vector": true_raq_stats.bytes_per_vector,
                "compression_ratio": true_raq_stats.compression_ratio,
                "n_vectors_encoded": true_raq_stats.n_vectors_encoded,
            } if true_raq_stats else None,
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the entire database to disk

        Args:
            path: Optional path to save to (uses storage path if not provided)
        """
        # Save storage
        self.storage.save()

        # Save index
        if self.is_built:
            save_path = Path(path) if path else self.storage.base_path
            self.ivf_index.save(str(save_path / "index.npz"))

            # Save old-style quantizer
            if self.quantizer:
                self.quantizer.save(str(save_path / "quantizer.npz"))
            
            # Save True RaBitQ
            if self.true_rabitq:
                self.true_rabitq.save(str(save_path / "true_rabitq.pkl"))

    def load(self, path: Optional[str] = None) -> None:
        """
        Load the database from disk

        Args:
            path: Optional path to load from (uses storage path if not provided)
        """
        # Load storage
        self.storage.load()

        vectors = self.storage.get_all_vectors()
        if vectors is not None and len(vectors) > 0:
            self.dimension = vectors.shape[1]

        # Load index
        load_path = Path(path) if path else self.storage.base_path
        index_file = load_path / "index.npz"

        if index_file.exists():
            self.ivf_index = IVFIndex.load(str(index_file))
            self.is_built = True

        # Load old-style quantizer
        quantizer_file = load_path / "quantizer.npz"
        if quantizer_file.exists():
            self.quantizer = RaBitQ.load(str(quantizer_file))
        
        # Load True RaBitQ
        true_rabitq_file = load_path / "true_rabitq.pkl"
        if true_rabitq_file.exists():
            self.true_rabitq = TrueRaBitQ.load(str(true_rabitq_file))

    def clear(self) -> None:
        """Clear all data from the database"""
        self.storage.clear()
        self.ivf_index = IVFIndex(
            nlist=self.config.ivf.nlist,
            nprobe=self.config.ivf.nprobe,
            metric=self.config.ivf.metric
        )
        self.quantizer = None
        self.true_rabitq = None
        self.is_built = False
        self.dimension = None


def create_vector_db(
    dimension: int,
    nlist: int = 100,
    nprobe: int = 10,
    use_quantization: bool = True,
    storage_path: str = "./vectordb_data"
) -> VectorDB:
    """
    Convenience function to create a VectorDB with common settings

    Args:
        dimension: Vector dimension
        nlist: Number of IVF clusters
        nprobe: Number of clusters to search
        use_quantization: Enable RaBitQ quantization
        storage_path: Path for persistence

    Returns:
        Configured VectorDB instance
    """
    config = VectorDBConfig(
        ivf=IVFIndexConfig(nlist=nlist, nprobe=nprobe),
        raq=RaBitQConfig(),
        dimension=dimension,
        use_quantization=use_quantization
    )

    return VectorDB(config=config, storage_path=storage_path)
