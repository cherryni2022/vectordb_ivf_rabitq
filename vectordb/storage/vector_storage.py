"""
Vector Storage Module

Handles persistence of vectors, metadata, and indexes to disk.
Supports different backends (filesystem, memory-only, etc.)
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class VectorMetadata:
    """Metadata for a single vector"""
    id: Any
    labels: Optional[Dict[str, Any]] = None


@dataclass
class StorageStats:
    """Statistics for storage"""
    total_vectors: int = 0
    total_bytes: int = 0
    metadata_count: int = 0


class VectorStorage:
    """
    Persistent storage for vectors and their metadata

    Provides:
    - Vector storage in numpy format (.npy)
    - Metadata storage in pickle format
    - Index persistence
    """

    def __init__(self, base_path: str = "./vectordb_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.vectors_path = self.base_path / "vectors.npy"
        self.metadata_path = self.base_path / "metadata.pkl"
        self.index_path = self.base_path / "index.npz"
        self.quantizer_path = self.base_path / "quantizer.npz"

        # In-memory storage
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[Any, Dict[str, Any]] = {}
        self.id_to_index: Dict[Any, int] = {}
        self.index_to_id: Dict[int, Any] = {}

        self.stats = StorageStats()

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add vectors to storage

        Args:
            vectors: Shape (n_vectors, dimension)
            ids: List of IDs for each vector (default: auto-increment integers)
            metadata: List of metadata dicts for each vector
        """
        n_vectors = vectors.shape[0]

        # Initialize storage if first add
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.id_to_index)
            ids = list(range(start_id, start_id + n_vectors))

        # Add IDs and metadata
        for i, vector_id in enumerate(ids):
            idx = len(self.id_to_index)
            self.id_to_index[vector_id] = idx
            self.index_to_id[idx] = vector_id

            if metadata is not None and i < len(metadata):
                self.metadata[vector_id] = metadata[i]
            else:
                self.metadata[vector_id] = {}

        self._update_stats()

    def get_vector(self, vector_id: Any) -> Optional[np.ndarray]:
        """Get a vector by its ID"""
        idx = self.id_to_index.get(vector_id)
        if idx is None or self.vectors is None:
            return None
        return self.vectors[idx]

    def get_vectors(self, vector_ids: List[Any]) -> Optional[np.ndarray]:
        """Get multiple vectors by their IDs"""
        indices = [self.id_to_index.get(vid) for vid in vector_ids]
        if None in indices or self.vectors is None:
            return None
        return self.vectors[indices]

    def get_metadata(self, vector_id: Any) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector"""
        return self.metadata.get(vector_id)

    def get_all_vectors(self) -> Optional[np.ndarray]:
        """Get all vectors"""
        return self.vectors

    def get_all_metadata(self) -> Dict[Any, Dict[str, Any]]:
        """Get all metadata"""
        return self.metadata

    def delete_vector(self, vector_id: Any) -> bool:
        """
        Delete a vector by its ID

        Note: For MVP, this marks as deleted rather than removing
        """
        if vector_id in self.metadata:
            self.metadata[vector_id]["_deleted"] = True
            return True
        return False

    def _update_stats(self) -> None:
        """Update storage statistics"""
        if self.vectors is not None:
            self.stats.total_vectors = len(self.vectors)
            self.stats.total_bytes = self.vectors.nbytes
        self.stats.metadata_count = len(self.metadata)

    def save(self) -> None:
        """Save all data to disk"""
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)

        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
            }, f)

    def load(self) -> None:
        """Load all data from disk"""
        if self.vectors_path.exists():
            self.vectors = np.load(self.vectors_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]

        self._update_stats()

    def clear(self) -> None:
        """Clear all data from storage"""
        self.vectors = None
        self.metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.stats = StorageStats()

    def get_stats(self) -> StorageStats:
        """Get storage statistics"""
        return self.stats


class InMemoryVectorStorage(VectorStorage):
    """
    In-memory only storage (no disk persistence)
    Useful for testing and temporary use cases
    """

    def __init__(self):
        # Skip parent __init__ to avoid filesystem operations
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[Any, Dict[str, Any]] = {}
        self.id_to_index: Dict[Any, int] = {}
        self.index_to_id: Dict[int, Any] = {}
        self.stats = StorageStats()
        self.base_path = Path("/dev/null")  # Not used, just for compatibility

    def save(self) -> None:
        """No-op for in-memory storage"""
        pass

    def load(self) -> None:
        """No-op for in-memory storage"""
        pass
