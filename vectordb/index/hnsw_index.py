"""
HNSW (Hierarchical Navigable Small World) Index Implementation

Reference: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
by Yu. A. Malkov and D. A. Yashunin
"""
import numpy as np
import pickle
import heapq
import random
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from vectordb.core.config import HNSWIndexConfig

@dataclass
class HNSWIndexStats:
    """Statistics for the HNSW index"""
    total_vectors: int = 0
    max_level: int = 0
    avg_neighbors: float = 0.0

class HNSWIndex:
    """
    Hierarchical Navigable Small World Index
    """

    def __init__(self, config: HNSWIndexConfig):
        self.config = config
        self.M = config.M
        self.M0 = config.M * 2  # Max connections for layer 0
        self.ef_construction = config.ef_construction
        self.ef_search = config.ef_search
        self.metric = config.metric

        # Data storage
        self.vectors: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
        
        # HNSW Graph structure
        # layers[i][node_id] = [neighbor_ids]
        # layers[0] includes all nodes
        self.layers: List[Dict[int, List[int]]] = []
        
        # Entry point for the graph (highest level)
        self.entry_point: Optional[int] = None
        
        # Max level of the current graph
        self.max_level: int = -1
        
        # Level of each node
        self.node_levels: Dict[int, int] = {}
        
        # Probability for level generation
        self.level_mult = 1.0 / np.log(self.M)
        
        self.is_built = False
        self.stats = HNSWIndexStats()

    def _dist_func(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors"""
        if self.metric == "l2":
            return float(np.sum((a - b) ** 2))
        elif self.metric == "ip":
            return float(-np.dot(a, b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _batch_distances(self, query: np.ndarray, indices: List[int]) -> np.ndarray:
        """Compute distances to multiple vectors at once (optimized)"""
        if len(indices) == 0:
            return np.array([])
        vecs = self.vectors[indices]
        if self.metric == "l2":
            return np.sum((vecs - query) ** 2, axis=1)
        elif self.metric == "ip":
            return -np.dot(vecs, query)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _get_random_level(self) -> int:
        """Generate a random level for a new node"""
        level = int(-np.log(random.random()) * self.level_mult)
        return level

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        """
        Search nearest neighbors in a specific layer (optimized with batch distance)
        """
        candidates = []
        w = []
        visited = set()
        
        # Initialize with entry points
        for ep in entry_points:
            dist = self._dist_func(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w, (-dist, ep))
            visited.add(ep)
            
        while candidates:
            c_dist, c_idx = heapq.heappop(candidates)
            
            f_dist = -w[0][0]
            if c_dist > f_dist:
                break
                
            # Get unvisited neighbors
            neighbors = self.layers[level].get(c_idx, [])
            unvisited = [n for n in neighbors if n not in visited]
            
            if not unvisited:
                continue
                
            # Batch compute distances
            visited.update(unvisited)
            distances = self._batch_distances(query, unvisited)
            
            for neighbor, dist in zip(unvisited, distances):
                f_dist = -w[0][0]
                if dist < f_dist or len(w) < ef:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(w, (-dist, neighbor))
                    if len(w) > ef:
                        heapq.heappop(w)
                            
        results = []
        while w:
            neg_dist, idx = heapq.heappop(w)
            results.append((-neg_dist, idx))
        return results

    def _select_neighbors(self, query: np.ndarray, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """
        Select best neighbors from candidates using heuristic
        Simple heuristic: just pick M closest
        """
        # Candidates are from _search_layer, which returns them somewhat sorted but not strictly top-k
        # Sort by distance
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:M]]

    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the HNSW index
        """
        n_vectors, dim = vectors.shape
        
        if self.vectors is None:
            self.vectors = vectors.copy()
            self.dimension = dim
            start_idx = 0
        else:
            start_idx = len(self.vectors)
            self.vectors = np.vstack([self.vectors, vectors])
            
        # Process each vector
        for i in range(n_vectors):
            self._insert(start_idx + i, vectors[i])
            
        self.is_built = True
        self.stats.total_vectors = len(self.vectors)
        self.stats.max_level = self.max_level
        
        # Estimate average neighbors at layer 0
        if self.layers and len(self.layers[0]) > 0:
            total_edges = sum(len(neighbors) for neighbors in self.layers[0].values())
            self.stats.avg_neighbors = total_edges / len(self.layers[0])

    def _insert(self, node_id: int, vector: np.ndarray) -> None:
        """Insert a single node into the graph"""
        level = self._get_random_level()
        self.node_levels[node_id] = level
        
        # Ensure layers exist
        while len(self.layers) <= level:
            self.layers.append({})
            
        # If first node
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            # Add to all layers up to 'level'
            for l in range(level + 1):
                self.layers[l][node_id] = []
            return
            
        curr_obj = self.entry_point
        curr_dist = self._dist_func(vector, self.vectors[curr_obj])
        
        # Search from top layer down to level + 1
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[l].get(curr_obj, [])
                for neighbor in neighbors:
                    dist = self._dist_func(vector, self.vectors[neighbor])
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True
        
        # From level down to 0, search and link
        for l in range(min(level, self.max_level), -1, -1):
            # perform search to find candidates
            w = self._search_layer(vector, [curr_obj], self.ef_construction, l)
            neighbors = self._select_neighbors(vector, w, self.M if l > 0 else self.M0)
            
            # Bidirectional connection
            self.layers[l][node_id] = [] # Initialize neighbor list
            for neighbor in neighbors:
                # Only add connections to neighbors that exist at this layer
                if neighbor not in self.layers[l]:
                    continue
                    
                self.layers[l][node_id].append(neighbor)
                self.layers[l][neighbor].append(node_id)
                
                # Prune connections if too many
                max_m = self.M if l > 0 else self.M0
                if len(self.layers[l][neighbor]) > max_m:
                    # Re-evaluate neighbors of 'neighbor'
                    # Get all their vectors
                    n_neighbors = self.layers[l][neighbor]
                    n_vectors = self.vectors[n_neighbors]
                    # Compute distances
                    dists = [(self._dist_func(self.vectors[neighbor], v), idx) 
                             for v, idx in zip(n_vectors, n_neighbors)]
                    # Keep closest M
                    dists.sort(key=lambda x: x[0])
                    self.layers[l][neighbor] = [x[1] for x in dists[:max_m]]
            
            # Update entry point for next layer (closest found in this layer)
            if w:
                w.sort(key=lambda x: x[0])
                curr_obj = w[0][1]
                curr_dist = w[0][0]
                
        # Update entry point if new node is at higher level
        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def build(self, vectors: np.ndarray) -> None:
        """
        Build index from scratch
        """
        self.vectors = None
        self.layers = []
        self.entry_point = None
        self.max_level = -1
        self.node_levels = {}
        
        self.add_vectors(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors
        """
        if self.entry_point is None:
            return []
            
        if query.ndim > 1:
            query = query.flatten()
            
        curr_obj = self.entry_point
        curr_dist = self._dist_func(query, self.vectors[curr_obj])
        
        # 1. Greedy search from top to layer 1
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[l].get(curr_obj, [])
                for neighbor in neighbors:
                    dist = self._dist_func(query, self.vectors[neighbor])
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True
                        
        # 2. Search layer 0 with ef_search
        w = self._search_layer(query, [curr_obj], self.ef_search, 0)
        
        # Sort and return top-k
        w.sort(key=lambda x: x[0])
        
        # Result format: (idx, dist)
        return [(idx, dist) for dist, idx in w[:k]]

    def get_stats(self) -> HNSWIndexStats:
        return self.stats

    def save(self, path: str) -> None:
        """Save index to disk"""
        save_data = {
            "config": self.config,
            "vectors": self.vectors,
            "layers": self.layers,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "node_levels": self.node_levels,
            "stats": self.stats
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """Load index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        index = cls(data["config"])
        index.vectors = data["vectors"]
        index.layers = data["layers"]
        index.entry_point = data["entry_point"]
        index.max_level = data["max_level"]
        index.node_levels = data["node_levels"]
        index.stats = data["stats"]
        index.is_built = True
        
        return index
