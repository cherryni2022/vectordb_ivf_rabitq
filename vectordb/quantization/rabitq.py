"""
RaBitQ (Randomized Binary Quantization) Implementation

RaBitQ combines ideas from:
1. Product Quantization (PQ): Splits vectors into sub-vectors and quantizes each independently
2. Binary Quantization: Uses binary codes for extreme compression
3. Randomized Projections: Uses random projections for speed

This implementation provides:
- Product Quantization for memory-efficient storage
- Binary quantization option for ultra-compressed storage
- Distance computation in quantized space
"""
import numpy as np
import pickle
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RaBitQStats:
    """Statistics for RaBitQ quantizer"""
    dimension: int = 0
    nsubq: int = 0
    sub_dim: int = 0
    nbits: int = 0
    is_trained: bool = False
    compression_ratio: float = 0.0


class RaBitQ:
    """
    RaBitQ - Randomized Binary/Product Quantization

    This quantizer divides vectors into sub-vectors and quantizes each independently.
    Supports both standard Product Quantization (PQ) and binary quantization modes.

    Parameters:
        dimension: Dimension of input vectors
        nsubq: Number of sub-quantizers (splits vector into nsubq parts)
        nbits: Number of bits per sub-quantizer (for codebook size: 2^nbits)
        use_binary: Use binary quantization instead of PQ
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dimension: int,
        nsubq: int = 8,
        nbits: int = 8,
        use_binary: bool = False,
        random_seed: Optional[int] = 42
    ):
        self.dimension = dimension
        self.nsubq = nsubq
        self.nbits = nbits
        self.use_binary = use_binary
        self.random_seed = random_seed
        self.is_trained = False

        # Calculate sub-vector dimensions
        self.sub_dim = dimension // nsubq

        # Codebook for each sub-quantizer
        # Shape: (nsubq, 2^nbits, sub_dim)
        self.codebook: Optional[np.ndarray] = None

        # Binary quantization: random projection matrix
        # Shape: (dimension, sub_dim * nsubq) for binary encoding
        self.random_matrix: Optional[np.ndarray] = None

        # Binary codes: each vector encoded as binary
        # Shape: (n_vectors, nbits) for each sub-quantizer
        self.binary_codes: Optional[np.ndarray] = None

        # PQ codes: each sub-quantizer stores codebook index
        # Shape: (n_vectors, nsubq)
        self.pq_codes: Optional[np.ndarray] = None

        # Statistics
        self.stats = RaBitQStats()

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def train(self, vectors: np.ndarray, max_iter: int = 25) -> None:
        """
        Train the quantizer on a set of vectors

        For PQ: Runs k-means on each sub-vector independently
        For Binary: Initializes random projection matrix

        Args:
            vectors: Shape (n_vectors, dimension) - Training vectors
            max_iter: Maximum iterations for k-means clustering
        """
        n_vectors, dimension = vectors.shape

        if dimension != self.dimension:
            raise ValueError(f"Vector dimension {dimension} doesn't match expected {self.dimension}")

        if self.use_binary:
            self._train_binary(vectors)
        else:
            self._train_pq(vectors, max_iter)

        self.is_trained = True
        self._update_stats(n_vectors)

    def _train_pq(self, vectors: np.ndarray, max_iter: int) -> None:
        """
        Train using Product Quantization (PQ)

        For each sub-vector, we run k-means to create a codebook.
        """
        nsubq = self.nsubq
        sub_dim = self.sub_dim
        n_centroids = 2 ** self.nbits

        # Initialize codebook
        self.codebook = np.zeros((nsubq, n_centroids, sub_dim), dtype=np.float32)

        # For each sub-quantizer, train k-means on corresponding sub-vectors
        for i in range(nsubq):
            start_idx = i * sub_dim
            end_idx = start_idx + sub_dim
            sub_vectors = vectors[:, start_idx:end_idx]

            # Train k-means on this sub-vector
            centroids = self._kmeans_subspace(sub_vectors, n_centroids, max_iter)
            self.codebook[i] = centroids

    def _train_binary(self, vectors: np.ndarray) -> None:
        """
        Train using Binary Quantization with random projections

        Uses random projections followed by thresholding to create binary codes.
        """
        n_vectors = vectors.shape[0]

        # For binary quantization, we create random projection matrices
        # This is inspired by Random Binary Quantization papers
        self.random_matrix = np.random.randn(self.dimension, self.sub_dim * self.nsubq).astype(np.float32)
        self.random_matrix /= np.linalg.norm(self.random_matrix, axis=0, keepdims=True)

    def _kmeans_subspace(
        self,
        sub_vectors: np.ndarray,
        n_centroids: int,
        max_iter: int
    ) -> np.ndarray:
        """
        Run k-means on a subspace of vectors

        Args:
            sub_vectors: Shape (n_vectors, sub_dim)
            n_centroids: Number of centroids
            max_iter: Maximum iterations

        Returns:
            centroids: Shape (n_centroids, sub_dim)
        """
        n_vectors = sub_vectors.shape[0]

        # Initialize centroids using k-means++
        centroids = self._kmeans_plusplus_init(sub_vectors, n_centroids)

        for _ in range(max_iter):
            # Assign to nearest centroid
            distances = self._compute_distances_l2(sub_vectors, centroids)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_centroids):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = np.mean(sub_vectors[mask], axis=0)
                else:
                    # Reinitialize empty cluster
                    new_centroids[i] = sub_vectors[np.random.choice(n_vectors)]

            # Check convergence
            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < 1e-4:
                break

            centroids = new_centroids

        return centroids

    def _kmeans_plusplus_init(
        self,
        vectors: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """K-means++ initialization"""
        n_vectors = vectors.shape[0]

        centroids = [vectors[np.random.choice(n_vectors)]]

        for _ in range(n_clusters - 1):
            current_centroids = np.array(centroids)
            distances = self._compute_distances_l2(vectors, current_centroids)
            min_distances = np.min(distances, axis=1)

            # Ensure non-negative distances (clip small negative values from floating point errors)
            min_distances = np.maximum(min_distances, 0.0)

            # Handle case where all distances are zero
            total = np.sum(min_distances)
            if total == 0:
                # If all distances are zero (e.g., duplicate vectors), pick uniformly
                probabilities = np.ones(n_vectors) / n_vectors
            else:
                probabilities = min_distances / total

            next_idx = np.random.choice(n_vectors, p=probabilities)
            centroids.append(vectors[next_idx])

        return np.array(centroids)

    def _compute_distances_l2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute L2 distances between two sets of vectors"""
        a_norm = np.sum(a ** 2, axis=1, keepdims=True)
        b_norm = np.sum(b ** 2, axis=1)
        dot = np.dot(a, b.T)
        distances = a_norm + b_norm - 2 * dot
        # Clip small negative values from floating point errors
        return np.maximum(distances, 0.0)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors using the trained quantizer

        Args:
            vectors: Shape (n_vectors, dimension)

        Returns:
            codes: Shape (n_vectors, nsubq) for PQ or (n_vectors, nbits) for binary
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer must be trained before encoding")

        n_vectors = vectors.shape[0]

        if self.use_binary:
            return self._encode_binary(vectors)
        else:
            return self._encode_pq(vectors)

    def _encode_pq(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode using Product Quantization

        For each sub-vector, find the closest codebook entry.
        """
        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.nsubq), dtype=np.uint8)

        for i in range(self.nsubq):
            start_idx = i * self.sub_dim
            end_idx = start_idx + self.sub_dim
            sub_vectors = vectors[:, start_idx:end_idx]

            # Find nearest codebook entry
            centroids = self.codebook[i]
            distances = self._compute_distances_l2(sub_vectors, centroids)
            codes[:, i] = np.argmin(distances, axis=1)

        self.pq_codes = codes
        return codes

    def _encode_binary(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode using binary quantization

        Project vectors using random matrix and threshold to binary.
        """
        # Project vectors
        projected = np.dot(vectors, self.random_matrix)

        # Threshold to binary (sign-based)
        binary_codes = (projected > 0).astype(np.uint8)

        self.binary_codes = binary_codes
        return binary_codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode codes back to vectors (approximate reconstruction)

        Args:
            codes: Shape (n_vectors, nsubq) for PQ or binary codes

        Returns:
            reconstructed: Shape (n_vectors, dimension)
        """
        if self.use_binary:
            raise NotImplementedError("Binary decoding not fully implemented for this MVP")

        n_vectors = codes.shape[0]
        reconstructed = np.zeros((n_vectors, self.dimension), dtype=np.float32)

        for i in range(self.nsubq):
            start_idx = i * self.sub_dim
            end_idx = start_idx + self.sub_dim

            # Get centroids for this sub-quantizer
            sub_centroids = self.codebook[i, codes[:, i]]
            reconstructed[:, start_idx:end_idx] = sub_centroids

        return reconstructed

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance table for a query vector

        For PQ: Compute distances from query to all codebook entries
        For each sub-quantizer i: table[i, j] = distance(query_sub_i, codebook[i, j])

        Args:
            query: Shape (dimension,)

        Returns:
            distance_table: Shape (nsubq, 2^nbits)
        """
        if self.use_binary:
            # For binary, compute distances to random projections
            projected_query = np.dot(query, self.random_matrix)
            return projected_query.reshape(self.nsubq, self.sub_dim)

        n_centroids = 2 ** self.nbits
        distance_table = np.zeros((self.nsubq, n_centroids), dtype=np.float32)

        for i in range(self.nsubq):
            start_idx = i * self.sub_dim
            end_idx = start_idx + self.sub_dim
            query_sub = query[start_idx:end_idx].reshape(1, -1)

            # Compute distances to all centroids in this sub-quantizer
            centroids = self.codebook[i]
            distances = self._compute_distances_l2(query_sub, centroids)[0]
            distance_table[i] = distances

        return distance_table

    def compute_asymmetric_distance(
        self,
        codes: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        """
        Compute asymmetric distance between query (encoded as distance table)
        and database vectors (encoded as codes)

        This is the main operation for PQ-based ANN search:
        distance = sum over sub-quantizers of distance_table[i, codes[j, i]]

        Args:
            codes: Shape (n_vectors, nsubq) - PQ codes for database vectors
            distance_table: Shape (nsubq, 2^nbits) - Precomputed distances

        Returns:
            distances: Shape (n_vectors,) - Approximate distances
        """
        n_vectors = codes.shape[0]
        distances = np.zeros(n_vectors, dtype=np.float32)

        for i in range(self.nsubq):
            # Add distances for this sub-quantizer
            distances += distance_table[i, codes[:, i]]

        return distances

    def compute_hamming_distance(self, binary_codes_a: np.ndarray, binary_codes_b: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distance between binary codes

        Args:
            binary_codes_a: Shape (n1, n_bits)
            binary_codes_b: Shape (n2, n_bits)

        Returns:
            distances: Shape (n1, n2) - Hamming distances
        """
        # XOR + count bits
        xor_result = np.bitwise_xor(
            binary_codes_a[:, np.newaxis, :],
            binary_codes_b[np.newaxis, :, :]
        )
        return np.sum(xor_result, axis=2)

    def _update_stats(self, n_vectors: int) -> None:
        """Update quantizer statistics"""
        self.stats.dimension = self.dimension
        self.stats.nsubq = self.nsubq
        self.stats.sub_dim = self.sub_dim
        self.stats.nbits = self.nbits
        self.stats.is_trained = True

        if self.use_binary:
            # Binary: each vector compressed to nsubq * nbits bits
            original_bits = self.dimension * 32  # Assuming float32
            compressed_bits = self.nsubq * self.nbits
            self.stats.compression_ratio = compressed_bits / original_bits
        else:
            # PQ: each vector compressed to nsubq * nbits bits
            original_bits = self.dimension * 32
            compressed_bits = self.nsubq * self.nbits
            self.stats.compression_ratio = compressed_bits / original_bits

    def get_stats(self) -> RaBitQStats:
        """Get quantizer statistics"""
        return self.stats

    def save(self, path: str) -> None:
        """Save the quantizer to disk"""
        save_data = {
            "codebook": self.codebook,
            "random_matrix": self.random_matrix,
            "pq_codes": self.pq_codes,
            "binary_codes": self.binary_codes,
            "dimension": self.dimension,
            "nsubq": self.nsubq,
            "nbits": self.nbits,
            "use_binary": self.use_binary,
            "sub_dim": self.sub_dim,
            "is_trained": self.is_trained
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str) -> "RaBitQ":
        """Load the quantizer from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        quantizer = cls(
            dimension=int(data["dimension"]),
            nsubq=int(data["nsubq"]),
            nbits=int(data["nbits"]),
            use_binary=bool(data["use_binary"]),
            random_seed=None
        )
        quantizer.codebook = data["codebook"]
        quantizer.random_matrix = data["random_matrix"]
        quantizer.pq_codes = data["pq_codes"]
        quantizer.binary_codes = data["binary_codes"]
        quantizer.sub_dim = int(data["sub_dim"])
        quantizer.is_trained = bool(data["is_trained"])

        return quantizer


def compute_pq_distance(codebook: np.ndarray, query: np.ndarray, code: int) -> float:
    """
    Compute distance between query sub-vector and codebook entry

    Helper function for IVF-PQ search

    Args:
        codebook: Shape (n_centroids, sub_dim)
        query: Shape (sub_dim,)
        code: Index into codebook

    Returns:
        distance: Squared L2 distance
    """
    return np.sum((query - codebook[code]) ** 2)
