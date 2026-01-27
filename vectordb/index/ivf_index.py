"""
IVF (Inverted File) Index Implementation

IVF partitions vectors into clusters using k-means clustering.
During search, only a subset of closest clusters to the query are examined.

Reference: FAISS IVF implementation
"""
import numpy as np
import pickle
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass
import heapq

if TYPE_CHECKING:
    from vectordb.quantization.true_rabitq import TrueRaBitQ


@dataclass
class IVFIndexStats:
    """Statistics for the IVF index"""
    total_vectors: int = 0
    num_clusters: int = 0
    avg_vectors_per_cluster: float = 0.0


class IVFIndex:
    """
    Inverted File Index for approximate nearest neighbor search

    Parameters:
        nlist: Number of clusters (partitions)
        nprobe: Number of clusters to search during query
        metric: Distance metric ('l2' or 'ip')
    """

    def __init__(self, nlist: int = 100, nprobe: int = 10, metric: str = "l2"):
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.dimension: Optional[int] = None
        self.is_built = False

        # Clustering-related data
        self.centroids: Optional[np.ndarray] = None  # Shape: (nlist, dimension)
        self.inverted_lists: List[np.ndarray] = []  # Each element contains indices of vectors in that cluster
        self.vectors: Optional[np.ndarray] = None  # Original vectors (if stored)

        # Mapping from vector index to cluster index
        self.vector_to_cluster: np.ndarray = np.array([], dtype=np.int32)

        # Statistics
        self.stats = IVFIndexStats()

    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors based on metric"""
        if self.metric == "l2":
            return np.linalg.norm(a - b) ** 2
        elif self.metric == "ip":
            return -np.dot(a, b)  # Negative because we want min distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_distances_matrix(self, queries: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Compute distances between query vectors and point vectors

        Args:
            queries: Shape (n_queries, dimension)
            points: Shape (n_points, dimension)

        Returns:
            Shape (n_queries, n_points)
        """
        if self.metric == "l2":
            # Efficient L2 distance computation
            # ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q*p
            q_norm = np.sum(queries ** 2, axis=1, keepdims=True)
            p_norm = np.sum(points ** 2, axis=1, keepdims=True).T
            dot = np.dot(queries, points.T)
            return q_norm + p_norm - 2 * dot
        elif self.metric == "ip":
            # Negative inner product (for min distance)
            return -np.dot(queries, points.T)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def build(self, vectors: np.ndarray) -> None:
        """
        Build the IVF index from a collection of vectors

        Args:
            vectors: Shape (n_vectors, dimension) - The vectors to index
        """
        n_vectors, dimension = vectors.shape
        self.dimension = dimension
        self.vectors = vectors.copy()

        # Adjust nlist if it's larger than the number of vectors
        if self.nlist > n_vectors:
            self.nlist = min(n_vectors, 10)

        # Step 1: Run k-means clustering to get centroids
        centroids = self._kmeans_clustering(vectors, self.nlist)

        # Step 2: Assign each vector to its nearest centroid
        distances = self._compute_distances_matrix(vectors, centroids)
        cluster_assignments = np.argmin(distances, axis=1)

        # Step 3: Build inverted lists
        self.inverted_lists = []
        for i in range(self.nlist):
            mask = cluster_assignments == i
            indices = np.where(mask)[0]
            self.inverted_lists.append(indices)

        # Store metadata
        self.centroids = centroids
        self.vector_to_cluster = cluster_assignments
        self.is_built = True

        # Update statistics
        self.stats.total_vectors = n_vectors
        self.stats.num_clusters = self.nlist
        self.stats.avg_vectors_per_cluster = n_vectors / self.nlist

    def _kmeans_clustering(self, vectors: np.ndarray, n_clusters: int, max_iter: int = 100,
                           tol: float = 1e-4) -> np.ndarray:
        """
        Perform k-means clustering on vectors

        Args:
            vectors: Shape (n_vectors, dimension)
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            centroids: Shape (n_clusters, dimension)
        """
        n_vectors = vectors.shape[0]

        # Initialize centroids using k-means++ for better quality
        centroids = self._kmeans_plusplus_init(vectors, n_clusters)

        for iteration in range(max_iter):
            # Assign each vector to the nearest centroid
            distances = self._compute_distances_matrix(vectors, centroids)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(n_clusters)

            for i in range(n_clusters):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = np.mean(vectors[mask], axis=0)
                    counts[i] = np.sum(mask)
                else:
                    # Reinitialize empty clusters to a random point
                    new_centroids[i] = vectors[np.random.choice(n_vectors)]

            # Check convergence
            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
                break

            centroids = new_centroids

        return centroids

    def _kmeans_plusplus_init(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        K-means++ initialization for better clustering quality

        Args:
            vectors: Shape (n_vectors, dimension)
            n_clusters: Number of clusters

        Returns:
            Initial centroids: Shape (n_clusters, dimension)
        """
        n_vectors = vectors.shape[0]

        # Choose first centroid uniformly at random
        centroids = [vectors[np.random.choice(n_vectors)]]

        for _ in range(n_clusters - 1):
            # Compute distances to nearest existing centroid
            current_centroids = np.array(centroids)
            distances = self._compute_distances_matrix(vectors, current_centroids)
            min_distances = np.min(distances, axis=1) ** 2

            # Ensure non-negative distances (clip small negative values from floating point errors)
            min_distances = np.maximum(min_distances, 0.0)

            # Choose next centroid with probability proportional to distance^2
            total = np.sum(min_distances)
            if total == 0:
                # If all distances are zero (e.g., duplicate vectors), pick uniformly
                probabilities = np.ones(n_vectors) / n_vectors
            else:
                probabilities = min_distances / total

            next_idx = np.random.choice(n_vectors, p=probabilities)
            centroids.append(vectors[next_idx])

        return np.array(centroids)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors using IVF index

        Args:
            query: Shape (dimension,) - The query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (vector_index, distance) tuples
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before search")

        # Reshape query if needed
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Step 1: Find the nprobe closest centroids to the query
        centroid_distances = self._compute_distances_matrix(query, self.centroids)[0]
        closest_cluster_indices = np.argsort(centroid_distances)[:self.nprobe]

        # Step 2: Search only vectors in the selected clusters
        all_candidates = []
        for cluster_idx in closest_cluster_indices:
            indices = self.inverted_lists[cluster_idx]
            if len(indices) > 0:
                candidate_vectors = self.vectors[indices]
                distances = self._compute_distances_matrix(query, candidate_vectors)[0]

                for idx, dist in zip(indices, distances):
                    all_candidates.append((dist, idx))

        # Step 3: Return top-k results
        k = min(k, len(all_candidates))
        top_k = heapq.nsmallest(k, all_candidates)


        return [(idx, dist) for dist, idx in top_k]

    def search_with_rabitq(
        self, 
        query: np.ndarray, 
        k: int, 
        quantizer: "TrueRaBitQ",
        rerank_factor: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search using IVF + RaBitQ for accelerated approximate nearest neighbor search
        
        Two-stage search:
        1. Coarse stage: Use RaBitQ approximate distances to find candidates within nprobe clusters
        2. Rerank stage: Compute exact distances for top candidates
        
        This provides significant speedup while maintaining high recall.
        
        Args:
            query: Shape (dimension,) - The query vector
            k: Number of nearest neighbors to return
            quantizer: Trained TrueRaBitQ quantizer with encoded vectors
            rerank_factor: Multiply k by this for candidate selection (default: 10)
            
        Returns:
            List of (vector_index, distance) tuples sorted by distance
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before search")
        
        if not quantizer.is_trained or quantizer.binary_codes is None:
            raise RuntimeError("Quantizer must be trained and have encoded vectors")
        
        # Reshape query if needed
        query_flat = query.flatten() if query.ndim > 1 else query
        query_2d = query_flat.reshape(1, -1)
        
        # Step 1: Find the nprobe closest centroids to the query
        centroid_distances = self._compute_distances_matrix(query_2d, self.centroids)[0]
        closest_cluster_indices = np.argsort(centroid_distances)[:self.nprobe]
        
        # Step 2: Collect all candidate indices from selected clusters
        all_candidate_indices = []
        for cluster_idx in closest_cluster_indices:
            indices = self.inverted_lists[cluster_idx]
            if len(indices) > 0:
                all_candidate_indices.extend(indices.tolist())
        
        if len(all_candidate_indices) == 0:
            return []
        
        candidate_indices = np.array(all_candidate_indices, dtype=np.int64)
        
        # Step 3: Compute approximate distances using RaBitQ
        approx_distances = quantizer.compute_approximate_distances(
            query_flat, 
            indices=candidate_indices
        )
        
        # Step 4: Select top-k * rerank_factor candidates for reranking
        n_rerank = min(k * rerank_factor, len(candidate_indices))
        top_approx_indices = np.argsort(approx_distances)[:n_rerank]
        rerank_candidates = candidate_indices[top_approx_indices]
        
        # Step 5: Compute exact distances for reranking
        candidate_vectors = self.vectors[rerank_candidates]
        exact_distances = self._compute_distances_matrix(query_2d, candidate_vectors)[0]
        
        # Step 6: Return top-k by exact distance
        k = min(k, len(exact_distances))
        top_k_local = np.argsort(exact_distances)[:k]
        
        results = [
            (int(rerank_candidates[i]), float(exact_distances[i])) 
            for i in top_k_local
        ]
        
        return results

    def get_cluster_candidates(
        self, 
        query: np.ndarray, 
        nprobe: Optional[int] = None
    ) -> np.ndarray:
        """
        Get all candidate indices from the nprobe closest clusters
        
        Useful for external quantizer integration.
        
        Args:
            query: Shape (dimension,) - The query vector
            nprobe: Number of clusters to probe (uses self.nprobe if None)
            
        Returns:
            Array of candidate vector indices
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before search")
        
        actual_nprobe = nprobe if nprobe is not None else self.nprobe
        
        # Reshape query if needed
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Find closest centroids
        centroid_distances = self._compute_distances_matrix(query, self.centroids)[0]
        closest_cluster_indices = np.argsort(centroid_distances)[:actual_nprobe]
        
        # Collect all candidate indices
        all_candidates = []
        for cluster_idx in closest_cluster_indices:
            indices = self.inverted_lists[cluster_idx]
            if len(indices) > 0:
                all_candidates.extend(indices.tolist())
        
        return np.array(all_candidates, dtype=np.int64)

    def search_batch(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Batch search for multiple queries

        Args:
            queries: Shape (n_queries, dimension)
            k: Number of nearest neighbors per query

        Returns:
            List of lists, each containing (vector_index, distance) tuples
        """
        return [self.search(q, k) for q in queries]

    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add new vectors to the index

        Note: In production, you would typically rebuild or update clusters periodically.
        For MVP, we'll simply add to the nearest cluster without re-clustering.

        Args:
            vectors: Shape (n_new_vectors, dimension)
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before adding vectors")

        n_new = vectors.shape[0]
        new_indices = np.arange(len(self.vectors), len(self.vectors) + n_new)

        # Assign to nearest existing clusters
        # Step 1: Compute distances to centroids, shape: (n_new, n_clusters)
        distances = self._compute_distances_matrix(vectors, self.centroids)
        cluster_assignments = np.argmin(distances, axis=1)

        # Add vectors to storage
        self.vectors = np.vstack([self.vectors, vectors])

        # Update inverted lists and assignment mapping
        for idx, cluster_idx in zip(new_indices, cluster_assignments):
            self.inverted_lists[cluster_idx] = np.append(
                self.inverted_lists[cluster_idx], idx
            )

        self.vector_to_cluster = np.concatenate([self.vector_to_cluster, cluster_assignments])

        # Update statistics
        self.stats.total_vectors += n_new
        self.stats.avg_vectors_per_cluster = self.stats.total_vectors / self.stats.num_clusters

    def get_stats(self) -> IVFIndexStats:
        """Get index statistics"""
        return self.stats

    def save(self, path: str) -> None:
        """Save the index to disk"""
        save_data = {
            "centroids": self.centroids,
            "vectors": self.vectors,
            "inverted_lists": self.inverted_lists,
            "vector_to_cluster": self.vector_to_cluster,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "dimension": self.dimension,
            "is_built": self.is_built
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        """Load the index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls(
            nlist=int(data["nlist"]),
            nprobe=int(data["nprobe"]),
            metric=str(data["metric"])
        )
        index.centroids = data["centroids"]
        index.vectors = data["vectors"]
        index.inverted_lists = data["inverted_lists"]
        index.vector_to_cluster = data["vector_to_cluster"]
        index.dimension = int(data["dimension"])
        index.is_built = bool(data["is_built"])

        index.stats.total_vectors = len(index.vectors)
        index.stats.num_clusters = index.nlist
        index.stats.avg_vectors_per_cluster = index.stats.total_vectors / index.nlist

        return index
