"""
Configuration classes for VectorDB
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class IVFIndexConfig:
    """
    IVF (Inverted File Index) configuration

    Parameters:
        nlist: Number of clusters (partitions) for k-means clustering
        nprobe: Number of clusters to search during query time
        metric: Distance metric ('l2' for Euclidean, 'ip' for inner product)
    """
    nlist: int = 100
    nprobe: int = 10
    metric: str = "l2"


@dataclass
class HNSWIndexConfig:
    """
    HNSW (Hierarchical Navigable Small World) configuration

    Parameters:
        M: Maximum number of connections per element in layer 0
        ef_construction: Size of the dynamic list for the nearest neighbors (used during construction)
        ef_search: Size of the dynamic list for the nearest neighbors (used during search)
        metric: Distance metric ('l2' for Euclidean, 'ip' for inner product)
    """
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    metric: str = "l2"


@dataclass
class RaBitQConfig:
    """
    RaBitQ (Randomized Binary Quantization) configuration

    This is a simplified product quantization approach:
    - Divides vectors into sub-vectors (subspaces)
    - Each subspace is quantized independently
    - Uses binary quantization for extreme compression

    Parameters:
        nbits: Number of bits per subspace (for binary quantization)
        nsubq: Number of sub-quantizers (splits vector into nsubq parts)
        use_rabitq: Enable RaBitQ quantization (vs standard PQ)
        random_seed: Random seed for reproducibility
    """
    nbits: int = 8
    nsubq: int = 8  # Number of sub-vectors for product quantization
    use_rabitq: bool = True
    random_seed: Optional[int] = 42


@dataclass
class VectorDBConfig:
    """
    Main VectorDB configuration combining IVF and RaBitQ settings
    """
    ivf: IVFIndexConfig = field(default_factory=IVFIndexConfig)
    hnsw: HNSWIndexConfig = field(default_factory=HNSWIndexConfig)
    raq: RaBitQConfig = field(default_factory=RaBitQConfig)
    dimension: Optional[int] = None  # Will be inferred from data
    index_type: Literal["ivf", "hnsw"] = "ivf"
    use_quantization: bool = True
