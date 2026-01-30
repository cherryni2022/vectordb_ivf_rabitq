#!/usr/bin/env python3
"""
Vector Database Benchmark Utility

Run performance benchmarks for vector database operations:
- Index building
- Search latency
- Memory usage
- Accuracy/recall

Usage:
    python benchmark.py --dataset path/to/vectors.npy --queries path/to/queries.npy
"""
import argparse
import time
import numpy as np
import sys
from typing import Dict, Any, Tuple
from pathlib import Path


class VectorDBBenchmark:
    """Performance benchmarking for vector databases"""

    def __init__(self, index):
        self.index = index
        self.results = {}

    def build_time(self, vectors: np.ndarray) -> float:
        """Measure index build time"""
        start = time.perf_counter()
        self.index.build(vectors)
        return time.perf_counter() - start

    def search_latency(self,
                     queries: np.ndarray,
                     k: int = 10,
                     n_runs: int = 10) -> Dict[str, float]:
        """
        Measure search latency percentiles

        Returns: dict with qps, p50, p95, p99 latency in ms
        """
        latencies = []

        for _ in range(n_runs):
            start = time.perf_counter()
            for q in queries:
                self.index.search(q, k)
            end = time.perf_counter()
            latencies.append(end - start)

        latencies = np.array(latencies)
        per_query = latencies / len(queries)

        return {
            'qps': float(len(queries) / np.mean(latencies)),
            'p50_ms': float(np.percentile(per_query, 50) * 1000),
            'p95_ms': float(np.percentile(per_query, 95) * 1000),
            'p99_ms': float(np.percentile(per_query, 99) * 1000),
            'mean_ms': float(np.mean(per_query) * 1000),
        }

    def memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB"""
        import pickle
        import sys

        # Serialized size
        serialized = pickle.dumps(self.index)

        # Object memory
        obj_size = sys.getsizeof(self.index)

        # Vector storage (if available)
        vec_size = 0
        if hasattr(self.index, 'vectors') and self.index.vectors is not None:
            vec_size = self.index.vectors.nbytes / (1024 * 1024)

        # Centroids (if available)
        cent_size = 0
        if hasattr(self.index, 'centroids') and self.index.centroids is not None:
            cent_size = self.index.centroids.nbytes / (1024 * 1024)

        return {
            'serialized_mb': len(serialized) / (1024 * 1024),
            'object_mb': obj_size / (1024 * 1024),
            'vectors_mb': vec_size,
            'centroids_mb': cent_size,
        }

    def recall_at_k(self,
                   queries: np.ndarray,
                   ground_truth: np.ndarray,
                   k: int = 10) -> float:
        """
        Compute recall@k using ground truth labels

        Args:
            queries: query vectors
            ground_truth: true nearest neighbor indices for each query
        """
        correct = 0
        for i, query in enumerate(queries):
            results = self.index.search(query, k)
            predicted_ids = [idx for idx, _ in results]
            true_id = ground_truth[i]

            if true_id in predicted_ids:
                correct += 1

        return correct / len(queries)


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute recall@k between predicted and ground truth

    Args:
        predicted: shape (n_queries, k) - predicted indices
        ground_truth: shape (n_queries, k) - ground truth indices
    """
    correct = 0
    total = 0

    for i in range(len(predicted)):
        predicted_set = set(predicted[i])
        ground_set = set(ground_truth[i])
        correct += len(predicted_set & ground_set)
        total += len(ground_set)

    return correct / total if total > 0 else 0.0


def generate_test_vectors(n_vectors: int,
                         dimension: int = 128,
                         seed: int = 42) -> np.ndarray:
    """Generate random test vectors"""
    np.random.seed(seed)
    return np.random.randn(n_vectors, dimension).astype(np.float32)


def compute_ground_truth(queries: np.ndarray,
                       database: np.ndarray,
                       k: int = 10) -> np.ndarray:
    """
    Compute exact nearest neighbors (ground truth) using brute force

    Returns: shape (n_queries, k) indices of nearest neighbors
    """
    n_queries = queries.shape[0]
    results = np.zeros((n_queries, k), dtype=np.int64)

    for i, query in enumerate(queries):
        distances = np.sum((database - query) ** 2, axis=1)
        results[i] = np.argpartition(distances, k-1)[:k]
        results[i] = results[i][np.argsort(distances[results[i]])]

    return results


def run_benchmark(index_class,
                 index_kwargs: Dict[str, Any],
                 n_vectors: int = 10000,
                 n_queries: int = 100,
                 dimension: int = 128,
                 k: int = 10,
                 seed: int = 42) -> Dict[str, Any]:
    """
    Run complete benchmark suite

    Returns: dict with all benchmark results
    """
    print(f"Generating test data: {n_vectors} vectors, {n_queries} queries, {dimension}-dim")

    # Generate data
    vectors = generate_test_vectors(n_vectors, dimension, seed)
    queries = generate_test_vectors(n_queries, dimension, seed + 1)

    # Create index
    index = index_class(**index_kwargs)

    benchmark = VectorDBBenchmark(index)

    # Benchmark build
    print("Benchmarking build time...")
    build_time = benchmark.build_time(vectors)
    print(f"  Build time: {build_time:.3f}s")

    # Benchmark search
    print("Benchmarking search...")
    search_results = benchmark.search_latency(queries, k, n_runs=5)
    print(f"  QPS: {search_results['qps']:.1f}")
    print(f"  p50: {search_results['p50_ms']:.2f}ms")
    print(f"  p95: {search_results['p95_ms']:.2f}ms")

    # Benchmark memory
    print("Benchmarking memory...")
    memory_results = benchmark.memory_usage()
    print(f"  Vectors: {memory_results['vectors_mb']:.1f}MB")
    print(f"  Serialized: {memory_results['serialized_mb']:.1f}MB")

    # Compute recall (if ground truth is available)
    ground_truth = compute_ground_truth(queries, vectors, k)
    recall = benchmark.recall_at_k(queries, ground_truth, k)
    print(f"  Recall@{k}: {recall:.3f}")

    return {
        'build_time_s': build_time,
        'search': search_results,
        'memory': memory_results,
        'recall_at_k': recall,
    }


def print_comparison(results: Dict[str, Dict[str, Any]]):
    """Print comparison of multiple benchmark results"""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    print(f"{'Name':<20} {'Build (s)':<10} {'QPS':<10} {'p95 (ms)':<10} {'Recall@10':<10}")
    print("-" * 60)

    for name, res in results.items():
        print(f"{name:<20} {res['build_time_s']:<10.3f} "
              f"{res['search']['qps']:<10.1f} "
              f"{res['search']['p95_ms']:<10.2f} "
              f"{res['recall_at_k']:<10.3f}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Vector Database Benchmark")
    parser.add_argument('--vectors', type=str, help="Path to vectors .npy file")
    parser.add_argument('--queries', type=str, help="Path to queries .npy file")
    parser.add_argument('--n-vectors', type=int, default=10000,
                       help="Number of test vectors (if not loading from file)")
    parser.add_argument('--n-queries', type=int, default=100,
                       help="Number of test queries")
    parser.add_argument('--dimension', type=int, default=128,
                       help="Vector dimension")
    parser.add_argument('--k', type=int, default=10,
                       help="K for top-k search")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Import index classes
    try:
        from vectordb.index.ivf_index import IVFIndex
        has_ivf = True
    except ImportError:
        has_ivf = False

    # Load or generate data
    if args.vectors:
        vectors = np.load(args.vectors)
        queries = np.load(args.queries) if args.queries else vectors[:args.n_queries]
    else:
        vectors = generate_test_vectors(args.n_vectors, args.dimension, args.seed)
        queries = generate_test_vectors(args.n_queries, args.dimension, args.seed + 1)

    print(f"Vectors: {vectors.shape}")
    print(f"Queries: {queries.shape}")

    # Run benchmarks
    results = {}

    if has_ivf:
        print("\n--- IVF Index (nprobe=10) ---")
        results['IVF-10'] = run_benchmark(
            IVFIndex, {'nlist': 100, 'nprobe': 10},
            n_vectors=args.n_vectors, n_queries=args.n_queries,
            dimension=args.dimension, k=args.k, seed=args.seed
        )

        print("\n--- IVF Index (nprobe=20) ---")
        results['IVF-20'] = run_benchmark(
            IVFIndex, {'nlist': 100, 'nprobe': 20},
            n_vectors=args.n_vectors, n_queries=args.n_queries,
            dimension=args.dimension, k=args.k, seed=args.seed
        )

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
