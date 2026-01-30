"""
Abstract base backend for VectorDB computations

Defines the interface that all compute backends must implement.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Any, Dict


class BaseBackend(ABC):
    """
    Abstract base class for compute backends
    
    All backends (NumPy, PyTorch, CuPy) should implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name"""
        pass
    
    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """Whether this backend uses GPU"""
        pass
    
    @abstractmethod
    def compute_l2_distances(self, queries: np.ndarray, 
                             points: np.ndarray) -> np.ndarray:
        """
        Compute squared L2 distances between queries and points
        
        Args:
            queries: Shape (n_queries, dimension)
            points: Shape (n_points, dimension)
            
        Returns:
            distances: Shape (n_queries, n_points) as numpy array
        """
        pass
    
    @abstractmethod
    def compute_ip_distances(self, queries: np.ndarray,
                             points: np.ndarray) -> np.ndarray:
        """
        Compute negative inner product (for ranking)
        
        Args:
            queries: Shape (n_queries, dimension)
            points: Shape (n_points, dimension)
            
        Returns:
            distances: Shape (n_queries, n_points) as numpy array
        """
        pass
    
    @abstractmethod
    def kmeans_iteration(self, vectors: np.ndarray,
                         centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single K-Means iteration
        
        Args:
            vectors: Shape (n_vectors, dimension)
            centroids: Shape (n_clusters, dimension)
            
        Returns:
            new_centroids: Shape (n_clusters, dimension) as numpy array
            assignments: Shape (n_vectors,) as numpy array
        """
        pass
    
    @abstractmethod
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication
        
        Args:
            a: Shape (m, k)
            b: Shape (k, n)
            
        Returns:
            result: Shape (m, n) as numpy array
        """
        pass
    
    @abstractmethod
    def argmin(self, array: np.ndarray, axis: int) -> np.ndarray:
        """
        Argmin along axis
        
        Args:
            array: Input array
            axis: Axis to reduce
            
        Returns:
            indices: Argmin result as numpy array
        """
        pass
    
    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer array to device (GPU for GPU backends)
        
        Default implementation returns array unchanged.
        """
        return array
    
    def to_numpy(self, array: Any) -> np.ndarray:
        """
        Transfer array back to CPU numpy
        
        Default implementation returns array unchanged.
        """
        return array


class BackendManager:
    """
    Manager for compute backends
    
    Provides a unified interface to select and use different backends.
    """
    
    _backends: Dict[str, BaseBackend] = {}
    _default_backend: Optional[str] = None
    
    @classmethod
    def register(cls, backend: BaseBackend):
        """Register a backend"""
        cls._backends[backend.name] = backend
        if cls._default_backend is None:
            cls._default_backend = backend.name
    
    @classmethod
    def get(cls, name: Optional[str] = None) -> BaseBackend:
        """Get a backend by name, or the default"""
        if name is None:
            name = cls._default_backend
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not found. Available: {list(cls._backends.keys())}")
        return cls._backends[name]
    
    @classmethod
    def set_default(cls, name: str):
        """Set the default backend"""
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not found")
        cls._default_backend = name
    
    @classmethod
    def list_backends(cls) -> list:
        """List available backends"""
        return list(cls._backends.keys())
    
    @classmethod
    def get_best_gpu_backend(cls) -> Optional[BaseBackend]:
        """Get the best available GPU backend"""
        for name in ['pytorch_gpu', 'cupy']:
            if name in cls._backends:
                return cls._backends[name]
        return None
