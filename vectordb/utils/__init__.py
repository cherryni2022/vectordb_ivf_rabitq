"""
VectorDB utility functions
"""
from .numba_utils import (
    fast_hamming_distance,
    fast_l2_distances,
    is_numba_available,
    get_backend_info,
    NUMBA_AVAILABLE,
)

__all__ = [
    'fast_hamming_distance',
    'fast_l2_distances',
    'is_numba_available',
    'get_backend_info',
    'NUMBA_AVAILABLE',
]
