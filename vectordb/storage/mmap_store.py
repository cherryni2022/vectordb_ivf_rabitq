"""
Memory-mapped vector storage for large-scale datasets

Allows working with datasets larger than available RAM by using
memory-mapped files.
"""
import numpy as np
import os
from typing import Optional, Union
from pathlib import Path


class MemoryMappedVectorStore:
    """
    Memory-mapped vector storage
    
    Uses numpy memmap to allow working with vectors that don't fit in RAM.
    The operating system handles paging data in and out of memory as needed.
    
    Usage:
        # Create new store
        store = MemoryMappedVectorStore("vectors.mmap", dimension=128)
        mmap = store.create(n_vectors=1000000)
        mmap[:] = your_vectors
        
        # Load existing store
        store = MemoryMappedVectorStore("vectors.mmap", dimension=128)
        vectors = store.load()
        subset = store.get_batch([0, 1, 2, 100, 200])
    
    Args:
        path: Path to the memory-mapped file
        dimension: Vector dimension
        dtype: Data type (default: float32)
    """
    
    def __init__(self, path: Union[str, Path], dimension: int, 
                 dtype: type = np.float32):
        self.path = Path(path)
        self.dimension = dimension
        self.dtype = dtype
        self.mmap: Optional[np.memmap] = None
        self._n_vectors: int = 0
    
    def create(self, n_vectors: int, mode: str = 'w+') -> np.memmap:
        """
        Create a new memory-mapped file
        
        Args:
            n_vectors: Number of vectors to store
            mode: File mode ('w+' for create/overwrite)
            
        Returns:
            np.memmap array with shape (n_vectors, dimension)
        """
        self.mmap = np.memmap(
            str(self.path),
            dtype=self.dtype,
            mode=mode,
            shape=(n_vectors, self.dimension)
        )
        self._n_vectors = n_vectors
        assert self.mmap is not None
        return self.mmap
    
    def load(self, mode: str = 'r') -> np.memmap:
        """
        Load an existing memory-mapped file
        
        Args:
            mode: File mode ('r' for read-only, 'r+' for read-write)
            
        Returns:
            np.memmap array
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Memory-mapped file not found: {self.path}")
        
        # Get file size to determine number of vectors
        file_size = os.path.getsize(self.path)
        bytes_per_element = np.dtype(self.dtype).itemsize
        bytes_per_vector = self.dimension * bytes_per_element
        n_vectors = file_size // bytes_per_vector
        
        self.mmap = np.memmap(
            str(self.path),
            dtype=self.dtype,
            mode=mode,
            shape=(n_vectors, self.dimension)
        )
        self._n_vectors = n_vectors
        assert self.mmap is not None
        return self.mmap
    
    def get_batch(self, indices: Union[np.ndarray, list]) -> np.ndarray:
        """
        Load a batch of vectors by indices
        
        This copies the data to a regular numpy array for processing.
        
        Args:
            indices: Vector indices to load
            
        Returns:
            numpy array with the requested vectors
        """
        if self.mmap is None:
            raise RuntimeError("No memory-mapped file loaded. Call create() or load() first.")
        
        return np.array(self.mmap[indices])
    
    def get_slice(self, start: int, end: int) -> np.ndarray:
        """
        Get a contiguous slice of vectors
        
        More efficient than random access for sequential reads.
        
        Args:
            start: Start index
            end: End index (exclusive)
            
        Returns:
            numpy array with vectors[start:end]
        """
        if self.mmap is None:
            raise RuntimeError("No memory-mapped file loaded.")
        
        return np.array(self.mmap[start:end])
    
    def flush(self):
        """Flush changes to disk"""
        if self.mmap is not None:
            self.mmap.flush()
    
    def close(self):
        """Close the memory-mapped file"""
        if self.mmap is not None:
            del self.mmap
            self.mmap = None
    
    @property
    def n_vectors(self) -> int:
        """Number of vectors in the store"""
        return self._n_vectors
    
    @property
    def shape(self) -> tuple:
        """Shape of the stored data"""
        return (self._n_vectors, self.dimension)
    
    def __len__(self) -> int:
        return self._n_vectors
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ChunkedVectorLoader:
    """
    Efficient chunked loading for memory-mapped or large arrays
    
    Iterates over vectors in chunks to balance memory usage and performance.
    
    Usage:
        loader = ChunkedVectorLoader(vectors, chunk_size=10000)
        for chunk_vectors, start_idx in loader:
            process(chunk_vectors)
    """
    
    def __init__(self, vectors: Union[np.ndarray, np.memmap], 
                 chunk_size: int = 10000):
        """
        Args:
            vectors: Vector array or memmap
            chunk_size: Number of vectors per chunk
        """
        self.vectors = vectors
        self.chunk_size = chunk_size
        self.n_vectors = len(vectors)
    
    def __iter__(self):
        """Iterate over chunks"""
        for start in range(0, self.n_vectors, self.chunk_size):
            end = min(start + self.chunk_size, self.n_vectors)
            chunk = np.array(self.vectors[start:end])
            yield chunk, start
    
    def __len__(self):
        """Number of chunks"""
        return (self.n_vectors + self.chunk_size - 1) // self.chunk_size
    
    def get_chunk(self, chunk_idx: int) -> tuple:
        """
        Get a specific chunk by index
        
        Args:
            chunk_idx: Chunk index
            
        Returns:
            (chunk_vectors, start_idx) tuple
        """
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.n_vectors)
        return np.array(self.vectors[start:end]), start
