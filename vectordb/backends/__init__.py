"""
VectorDB compute backends

Provides unified interface for different compute backends:
- numpy: Default CPU backend
- pytorch_cuda: GPU backend using PyTorch CUDA
- pytorch_cpu: CPU backend using PyTorch

Usage:
    from vectordb.backends import BackendManager, get_best_backend
    
    # Get the best available backend
    backend = get_best_backend()
    
    # Compute distances
    distances = backend.compute_l2_distances(queries, points)
"""

from .base_backend import BaseBackend, BackendManager
from .numpy_backend import NumpyBackend

# Try to import optional backends
try:
    from .pytorch_backend import (
        PyTorchBackend, 
        is_pytorch_available, 
        is_cuda_available,
        get_pytorch_device_info,
    )
except ImportError:
    is_pytorch_available = lambda: False
    is_cuda_available = lambda: False
    get_pytorch_device_info = lambda: {'available': False}
    PyTorchBackend = None


def get_best_backend(prefer_gpu: bool = True) -> BaseBackend:
    """
    Get the best available backend
    
    Args:
        prefer_gpu: If True, prefer GPU backends when available
        
    Returns:
        The best available backend
    """
    if prefer_gpu:
        gpu_backend = BackendManager.get_best_gpu_backend()
        if gpu_backend is not None:
            return gpu_backend
    
    return BackendManager.get('numpy')


def get_backend(name: str) -> BaseBackend:
    """
    Get a specific backend by name
    
    Args:
        name: Backend name ('numpy', 'pytorch_cuda', 'pytorch_cpu')
        
    Returns:
        The requested backend
    """
    return BackendManager.get(name)


def list_backends() -> list:
    """List all available backends"""
    return BackendManager.list_backends()


__all__ = [
    'BaseBackend',
    'BackendManager',
    'NumpyBackend',
    'PyTorchBackend',
    'get_best_backend',
    'get_backend',
    'list_backends',
    'is_pytorch_available',
    'is_cuda_available',
    'get_pytorch_device_info',
]
