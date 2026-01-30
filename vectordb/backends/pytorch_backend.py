"""
PyTorch GPU backend for VectorDB computations

Provides GPU-accelerated operations using PyTorch CUDA.
"""
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from .base_backend import BaseBackend, BackendManager

# Check if PyTorch with CUDA is available
try:
    import torch
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None  # type: ignore


# Only define PyTorchBackend if torch is available
if PYTORCH_AVAILABLE:
    class PyTorchBackend(BaseBackend):
        """
        PyTorch-based compute backend
        
        Uses CUDA for GPU acceleration when available.
        
        Args:
            device: 'cuda', 'cpu', or specific device like 'cuda:0'
                    Defaults to 'cuda' if available, else 'cpu'
        """
        
        def __init__(self, device: Optional[str] = None):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Install with: pip install torch")
            
            if device is None:
                device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            
            self.device = torch.device(device)
            self._is_gpu = self.device.type == 'cuda'
        
        @property
        def name(self) -> str:
            return f"pytorch_{self.device.type}"
        
        @property
        def is_gpu(self) -> bool:
            return self._is_gpu
        
        def to_device(self, array: np.ndarray):
            """Transfer numpy array to PyTorch tensor on device"""
            return torch.from_numpy(array).to(self.device)
        
        def to_numpy(self, tensor) -> np.ndarray:
            """Transfer PyTorch tensor back to numpy"""
            return tensor.cpu().numpy()
        
        def compute_l2_distances(self, queries: np.ndarray,
                                 points: np.ndarray) -> np.ndarray:
            """
            Compute squared L2 distances using torch.cdist
            
            torch.cdist is highly optimized and uses CUDA efficiently.
            """
            queries_t = torch.from_numpy(queries.astype(np.float32)).to(self.device)
            points_t = torch.from_numpy(points.astype(np.float32)).to(self.device)
            
            # cdist computes Euclidean distance, we need squared
            distances = torch.cdist(queries_t, points_t, p=2).pow(2)
            
            return distances.cpu().numpy()
        
        def compute_ip_distances(self, queries: np.ndarray,
                                 points: np.ndarray) -> np.ndarray:
            """Compute negative inner product"""
            queries_t = torch.from_numpy(queries.astype(np.float32)).to(self.device)
            points_t = torch.from_numpy(points.astype(np.float32)).to(self.device)
            
            result = -torch.mm(queries_t, points_t.t())
            
            return result.cpu().numpy()
        
        def kmeans_iteration(self, vectors: np.ndarray,
                             centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Single K-Means iteration on GPU
            
            Uses efficient scatter_add for centroid updates.
            """
            n_clusters = len(centroids)
            dimension = vectors.shape[1]
            
            vectors_t = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
            centroids_t = torch.from_numpy(centroids.astype(np.float32)).to(self.device)
            
            # Assignment step
            distances = torch.cdist(vectors_t, centroids_t, p=2).pow(2)
            assignments = torch.argmin(distances, dim=1)
            
            # Update step using scatter_add
            new_centroids = torch.zeros(n_clusters, dimension, 
                                        dtype=torch.float32, device=self.device)
            counts = torch.zeros(n_clusters, dtype=torch.float32, device=self.device)
            
            # Expand assignments for scatter_add
            assignments_expanded = assignments.unsqueeze(1).expand(-1, dimension)
            new_centroids.scatter_add_(0, assignments_expanded, vectors_t)
            counts.scatter_add_(0, assignments, 
                               torch.ones(len(vectors), dtype=torch.float32, device=self.device))
            
            # Avoid division by zero
            counts = torch.clamp(counts, min=1)
            new_centroids /= counts.unsqueeze(1)
            
            return new_centroids.cpu().numpy(), assignments.cpu().numpy().astype(np.int32)
        
        def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Matrix multiplication on GPU"""
            a_t = torch.from_numpy(a.astype(np.float32)).to(self.device)
            b_t = torch.from_numpy(b.astype(np.float32)).to(self.device)
            
            result = torch.mm(a_t, b_t)
            
            return result.cpu().numpy()
        
        def argmin(self, array: np.ndarray, axis: int) -> np.ndarray:
            """Argmin on GPU"""
            tensor = torch.from_numpy(array).to(self.device)
            result = torch.argmin(tensor, dim=axis)
            return result.cpu().numpy()
        
        def run_kmeans(self, vectors: np.ndarray, n_clusters: int,
                       max_iter: int = 100, tol: float = 1e-4,
                       init_centroids: Optional[np.ndarray] = None) -> np.ndarray:
            """
            Full K-Means clustering on GPU
            
            Args:
                vectors: Shape (n_vectors, dimension)
                n_clusters: Number of clusters
                max_iter: Maximum iterations
                tol: Convergence tolerance
                init_centroids: Optional initial centroids
                
            Returns:
                centroids: Shape (n_clusters, dimension)
            """
            vectors_t = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
            n_vectors, dimension = vectors_t.shape
            
            # Initialize centroids
            if init_centroids is not None:
                centroids = torch.from_numpy(init_centroids.astype(np.float32)).to(self.device)
            else:
                # Random initialization
                indices = torch.randperm(n_vectors)[:n_clusters]
                centroids = vectors_t[indices].clone()
            
            for iteration in range(max_iter):
                # Assignment
                distances = torch.cdist(vectors_t, centroids, p=2).pow(2)
                assignments = torch.argmin(distances, dim=1)
                
                # Update
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(n_clusters, dtype=torch.float32, device=self.device)
                
                assignments_expanded = assignments.unsqueeze(1).expand(-1, dimension)
                new_centroids.scatter_add_(0, assignments_expanded, vectors_t)
                counts.scatter_add_(0, assignments, 
                                   torch.ones(n_vectors, dtype=torch.float32, device=self.device))
                
                counts = torch.clamp(counts, min=1)
                new_centroids /= counts.unsqueeze(1)
                
                # Check convergence
                shift = torch.max(torch.norm(new_centroids - centroids, dim=1))
                if shift < tol:
                    break
                
                centroids = new_centroids
            
            return centroids.cpu().numpy()

    # Register PyTorch backends
    try:
        if CUDA_AVAILABLE:
            _pytorch_gpu_backend = PyTorchBackend('cuda')
            BackendManager.register(_pytorch_gpu_backend)
        
        _pytorch_cpu_backend = PyTorchBackend('cpu')
        BackendManager.register(_pytorch_cpu_backend)
    except Exception:
        pass  # Skip if any error during initialization

else:
    # Placeholder when PyTorch is not available
    PyTorchBackend = None  # type: ignore


def is_pytorch_available() -> bool:
    """Check if PyTorch is available"""
    return PYTORCH_AVAILABLE


def is_cuda_available() -> bool:
    """Check if CUDA is available"""
    return CUDA_AVAILABLE


def get_pytorch_device_info() -> dict:
    """Get PyTorch device information"""
    if not PYTORCH_AVAILABLE:
        return {'available': False}
    
    info = {
        'available': True,
        'cuda_available': CUDA_AVAILABLE,
        'pytorch_version': torch.__version__,
    }
    
    if CUDA_AVAILABLE:
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
        })
    
    return info

