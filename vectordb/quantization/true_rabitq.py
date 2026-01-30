"""
True RaBitQ (Randomized Binary Quantization) Implementation

Based on the paper: "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound 
for Approximate Nearest Neighbor Search" (SIGMOD'24)

Key Ideas:
1. Random Orthogonal Transformation: Apply a random rotation to distribute information evenly
2. Sign Quantization: Quantize each dimension to 1 bit (sign), achieving 32x compression
3. Norm Preservation: Store original norms for distance reconstruction
4. Measurement Concentration: Leverage high-dimensional concentration for error bounds

The key insight is that for high-dimensional vectors, the angle between any two vectors
can be well-approximated by counting how many sign bits differ (Hamming distance).
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import pickle


@dataclass
class TrueRaBitQStats:
    """Statistics for True RaBitQ quantizer"""
    dimension: int = 0
    is_trained: bool = False
    compression_ratio: float = 0.0
    n_vectors_encoded: int = 0
    bits_per_vector: int = 0
    bytes_per_vector: int = 0


class TrueRaBitQ:
    """
    True RaBitQ - Randomized Binary Quantization
    
    This quantizer achieves 32x compression by:
    1. Normalizing vectors
    2. Applying random orthogonal transformation
    3. Taking the sign of each dimension (1 bit per dim)
    
    Distance approximation uses:
    - Hamming distance between binary codes
    - Original norms for scaling
    - Theoretical formula based on measurement concentration
    
    Parameters:
        dimension: Dimension of input vectors
        random_seed: Random seed for reproducibility
    """
    
    # Static popcount lookup table (class-level, created once)
    _POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    
    def __init__(self, dimension: int, random_seed: Optional[int] = 42):
        self.dimension = dimension
        self.random_seed = random_seed
        self.is_trained = False
        
        # Random orthogonal transformation matrix
        # Shape: (dimension, dimension)
        self.rotation_matrix: Optional[np.ndarray] = None
        
        # Encoded data storage
        # binary_codes: packed bits, shape (n_vectors, ceil(dimension/8))
        self.binary_codes: Optional[np.ndarray] = None
        # norms: original vector norms, shape (n_vectors,)
        self.norms: Optional[np.ndarray] = None
        
        # Statistics for distance computation
        self.mean_norm: float = 0.0
        self.std_norm: float = 0.0
        
        # Stats
        self.stats = TrueRaBitQStats()
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def train(self, vectors: np.ndarray) -> None:
        """
        Train the quantizer
        
        For RaBitQ, "training" means:
        1. Generate random orthogonal matrix (via QR decomposition)
        2. Compute norm statistics for distance estimation
        
        Args:
            vectors: Shape (n_vectors, dimension) - Training vectors
        """
        n_vectors, dimension = vectors.shape
        
        if dimension != self.dimension:
            raise ValueError(f"Vector dimension {dimension} doesn't match expected {self.dimension}")
        
        # Generate random orthogonal matrix using QR decomposition
        # This ensures a uniform random rotation
        random_matrix = np.random.randn(dimension, dimension).astype(np.float32)
        self.rotation_matrix, _ = np.linalg.qr(random_matrix)
        self.rotation_matrix = self.rotation_matrix.astype(np.float32)
        
        # Compute norm statistics
        norms = np.linalg.norm(vectors, axis=1)
        self.mean_norm = float(np.mean(norms))
        self.std_norm = float(np.std(norms))
        
        self.is_trained = True
        self._update_stats(0)
    
    def encode(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode vectors to binary codes
        
        Process:
        1. Compute and store norms
        2. Normalize vectors
        3. Apply random rotation
        4. Take sign (positive -> 1, negative -> 0)
        5. Pack bits into bytes
        
        Args:
            vectors: Shape (n_vectors, dimension)
            
        Returns:
            binary_codes: Shape (n_vectors, ceil(dimension/8)) uint8
            norms: Shape (n_vectors,) float32
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before encoding")
        
        if self.rotation_matrix is None:
            raise RuntimeError("Rotation matrix not initialized. Call train() first.")
        
        n_vectors = vectors.shape[0]
        
        # Step 1: Compute norms
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        
        # Step 2: Normalize (avoid division by zero)
        normalized = vectors / (norms[:, np.newaxis] + 1e-8)
        
        # Step 3: Apply random rotation
        rotated = np.dot(normalized, self.rotation_matrix)
        
        # Step 4: Sign quantization (positive -> 1, non-positive -> 0)
        signs = (rotated > 0).astype(np.uint8)
        
        # Step 5: Pack bits into bytes (8 bits per byte)
        # packbits packs along the last axis by default
        binary_codes = np.packbits(signs, axis=1)
        
        # Store for later use
        self.binary_codes = binary_codes
        self.norms = norms
        
        self._update_stats(n_vectors)
        
        return binary_codes, norms
    
    def encode_query(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Encode a single query vector
        
        Args:
            query: Shape (dimension,) or (1, dimension)
            
        Returns:
            binary_code: Shape (ceil(dimension/8),) uint8
            norm: float
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before encoding")
        
        if self.rotation_matrix is None:
            raise RuntimeError("Rotation matrix not initialized. Call train() first.")
        
        query = query.flatten()
        
        # Compute norm
        norm = float(np.linalg.norm(query))
        
        # Normalize
        normalized = query / (norm + 1e-8)
        
        # Rotate
        rotated = np.dot(normalized, self.rotation_matrix)
        
        # Sign quantization
        signs = (rotated > 0).astype(np.uint8)
        
        # Pack bits
        binary_code = np.packbits(signs)
        
        return binary_code, norm
    
    def compute_hamming_distance(
        self, 
        query_code: np.ndarray, 
        database_codes: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hamming distance between query and database codes
        
        Uses XOR + popcount for efficient computation.
        
        Args:
            query_code: Shape (n_bytes,) uint8
            database_codes: Shape (n_vectors, n_bytes) uint8
            
        Returns:
            hamming_distances: Shape (n_vectors,) int
        """
        # XOR: different bits become 1
        xor_result = np.bitwise_xor(database_codes, query_code)
        
        # Use class-level pre-computed popcount table (avoids recreation per call)
        # Sum popcount for each byte
        hamming_distances = np.sum(self._POPCOUNT_TABLE[xor_result], axis=1)
        
        return hamming_distances
    
    def compute_approximate_distances(
        self,
        query: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute approximate L2 distances using RaBitQ
        
        The approximation formula (based on measurement concentration):
        
        For normalized vectors u, v after random rotation:
        - cos(θ) ≈ 1 - 2 * hamming(sign(u), sign(v)) / d
        
        For original vectors x, y with norms ||x||, ||y||:
        - ||x - y||² = ||x||² + ||y||² - 2 * ||x|| * ||y|| * cos(θ)
        
        Args:
            query: Shape (dimension,) - Query vector
            indices: Optional indices to compute for (subset of encoded vectors)
            
        Returns:
            approximate_distances: Shape (n_vectors,) or (len(indices),)
        """
        if self.binary_codes is None or self.norms is None:
            raise RuntimeError("No vectors encoded. Call encode() first.")
        
        # Encode query
        query_code, query_norm = self.encode_query(query)
        
        # Get relevant codes and norms
        if indices is not None:
            codes = self.binary_codes[indices]
            norms = self.norms[indices]
        else:
            codes = self.binary_codes
            norms = self.norms
        
        # Compute Hamming distances
        hamming_dists = self.compute_hamming_distance(query_code, codes)
        
        # Convert Hamming distance to cosine approximation
        # cos(θ) ≈ 1 - 2 * H(u, v) / d
        # where H is Hamming distance and d is dimension
        cos_approx = 1.0 - 2.0 * hamming_dists.astype(np.float32) / self.dimension
        
        # Compute approximate L2 distance
        # ||x - y||² = ||x||² + ||y||² - 2 * ||x|| * ||y|| * cos(θ)
        query_norm_sq = query_norm ** 2
        norms_sq = norms ** 2
        
        approx_distances = query_norm_sq + norms_sq - 2.0 * query_norm * norms * cos_approx
        
        # Clip negative values (can occur due to approximation errors)
        approx_distances = np.maximum(approx_distances, 0.0)
        
        return approx_distances
    
    def compute_approximate_inner_product(
        self,
        query: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute approximate inner product using RaBitQ
        
        <x, y> ≈ ||x|| * ||y|| * cos(θ)
        
        Args:
            query: Shape (dimension,) - Query vector
            indices: Optional indices to compute for
            
        Returns:
            approximate_ip: Shape (n_vectors,) or (len(indices),)
        """
        if self.binary_codes is None or self.norms is None:
            raise RuntimeError("No vectors encoded. Call encode() first.")
        
        # Encode query
        query_code, query_norm = self.encode_query(query)
        
        # Get relevant codes and norms
        if indices is not None:
            codes = self.binary_codes[indices]
            norms = self.norms[indices]
        else:
            codes = self.binary_codes
            norms = self.norms
        
        # Compute Hamming distances
        hamming_dists = self.compute_hamming_distance(query_code, codes)
        
        # Convert to cosine
        cos_approx = 1.0 - 2.0 * hamming_dists.astype(np.float32) / self.dimension
        
        # Inner product
        approx_ip = query_norm * norms * cos_approx
        
        return approx_ip
    
    def _update_stats(self, n_vectors: int) -> None:
        """Update quantizer statistics"""
        self.stats.dimension = self.dimension
        self.stats.is_trained = self.is_trained
        self.stats.n_vectors_encoded = n_vectors
        
        # Compression calculation
        # Original: dimension * 4 bytes (float32)
        # Compressed: ceil(dimension/8) bytes + 4 bytes (norm)
        original_bytes = self.dimension * 4
        compressed_bytes = (self.dimension + 7) // 8 + 4  # binary code + norm
        
        self.stats.bits_per_vector = self.dimension
        self.stats.bytes_per_vector = compressed_bytes
        self.stats.compression_ratio = compressed_bytes / original_bytes
        
    def get_stats(self) -> TrueRaBitQStats:
        """Get quantizer statistics"""
        return self.stats
    
    def save(self, path: str) -> None:
        """Save the quantizer to disk"""
        save_data = {
            "rotation_matrix": self.rotation_matrix,
            "binary_codes": self.binary_codes,
            "norms": self.norms,
            "mean_norm": self.mean_norm,
            "std_norm": self.std_norm,
            "dimension": self.dimension,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, path: str) -> "TrueRaBitQ":
        """Load the quantizer from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        quantizer = cls(
            dimension=int(data["dimension"]),
            random_seed=None  # Don't reset seed when loading
        )
        quantizer.rotation_matrix = data["rotation_matrix"]
        quantizer.binary_codes = data["binary_codes"]
        quantizer.norms = data["norms"]
        quantizer.mean_norm = data["mean_norm"]
        quantizer.std_norm = data["std_norm"]
        quantizer.is_trained = bool(data["is_trained"])
        
        if quantizer.binary_codes is not None:
            quantizer._update_stats(len(quantizer.binary_codes))
        
        return quantizer


class RaBitQWithRerank:
    """
    RaBitQ with Reranking for high-quality ANN search
    
    Two-stage search:
    1. Coarse stage: Use RaBitQ approximate distances to find candidates
    2. Rerank stage: Compute exact distances for top candidates
    
    This provides a good trade-off between speed and accuracy.
    """
    
    def __init__(
        self, 
        dimension: int,
        rerank_factor: int = 10,
        random_seed: Optional[int] = 42
    ):
        """
        Args:
            dimension: Vector dimension
            rerank_factor: Multiply k by this factor for candidate selection
            random_seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.rerank_factor = rerank_factor
        self.rabitq = TrueRaBitQ(dimension, random_seed)
        
        # Store original vectors for reranking
        self.vectors: Optional[np.ndarray] = None
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the quantizer and store vectors for reranking"""
        self.rabitq.train(vectors)
        self.vectors = vectors.copy()
    
    def encode(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode vectors and update stored vectors"""
        self.vectors = vectors.copy()
        return self.rabitq.encode(vectors)
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        candidate_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with RaBitQ + Rerank
        
        Args:
            query: Query vector, shape (dimension,)
            k: Number of results to return
            candidate_indices: Optional indices to search within (for IVF integration)
            
        Returns:
            indices: Top-k indices
            distances: Top-k exact distances
        """
        if self.vectors is None:
            raise RuntimeError("No vectors stored. Call train() and encode() first.")
        
        query = query.flatten()
        n_candidates = k * self.rerank_factor
        
        # Stage 1: Coarse search with RaBitQ
        if candidate_indices is not None:
            # Search within specified candidates (IVF mode)
            approx_dists = self.rabitq.compute_approximate_distances(query, candidate_indices)
            n_candidates = min(n_candidates, len(candidate_indices))
            top_approx_indices = np.argsort(approx_dists)[:n_candidates]
            candidate_pool = candidate_indices[top_approx_indices]
        else:
            # Search all vectors
            approx_dists = self.rabitq.compute_approximate_distances(query)
            n_candidates = min(n_candidates, len(approx_dists))
            candidate_pool = np.argsort(approx_dists)[:n_candidates]
        
        # Stage 2: Rerank with exact distances
        candidate_vectors = self.vectors[candidate_pool]
        exact_dists = np.sum((candidate_vectors - query) ** 2, axis=1)
        
        # Get top-k
        k = min(k, len(exact_dists))
        top_k_local = np.argsort(exact_dists)[:k]
        
        final_indices = candidate_pool[top_k_local]
        final_distances = exact_dists[top_k_local]
        
        return final_indices, final_distances
    
    @property
    def is_trained(self) -> bool:
        return self.rabitq.is_trained
    
    def get_stats(self) -> TrueRaBitQStats:
        return self.rabitq.get_stats()
    
    def save(self, path: str) -> None:
        """Save to disk"""
        save_data = {
            "rabitq_path": path + ".rabitq",
            "vectors": self.vectors,
            "rerank_factor": self.rerank_factor,
            "dimension": self.dimension,
        }
        self.rabitq.save(path + ".rabitq")
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, path: str) -> "RaBitQWithRerank":
        """Load from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        searcher = cls(
            dimension=data["dimension"],
            rerank_factor=data["rerank_factor"],
            random_seed=None
        )
        searcher.rabitq = TrueRaBitQ.load(data["rabitq_path"])
        searcher.vectors = data["vectors"]
        
        return searcher
