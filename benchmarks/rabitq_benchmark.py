"""
Performance Benchmark: RaBitQ Accelerated Search vs Standard Search

This script demonstrates the speedup achieved by using True RaBitQ
for approximate nearest neighbor search.
"""
import numpy as np
import time
from typing import Tuple, List
from vectordb import VectorDB


def benchmark_search(
    db: VectorDB,
    queries: np.ndarray,
    k: int,
    accelerated: bool,
    warmup_runs: int = 10
) -> Tuple[float, float, List[float]]:
    """
    Benchmark search performance
    
    Returns:
        mean_latency_ms: Mean latency in milliseconds
        qps: Queries per second
        latencies: List of individual latencies
    """
    # Warmup
    for i in range(min(warmup_runs, len(queries))):
        db.search(queries[i], k=k, accelerated=accelerated)
    
    # Benchmark
    latencies = []
    start_time = time.time()
    
    for query in queries:
        query_start = time.perf_counter()
        db.search(query, k=k, accelerated=accelerated)
        latencies.append((time.perf_counter() - query_start) * 1000)  # ms
    
    total_time = time.time() - start_time
    
    mean_latency = np.mean(latencies)
    qps = len(queries) / total_time
    
    return mean_latency, qps, latencies


def compute_recall(
    db: VectorDB,
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    accelerated: bool
) -> float:
    """Compute recall@k against brute force"""
    recalls = []
    
    for i, query in enumerate(queries):
        # Get approximate results
        results = db.search(query, k=k, accelerated=accelerated)
        
        # Compute ground truth (brute force)
        distances = np.sum((vectors - query) ** 2, axis=1)
        ground_truth = set(np.argsort(distances)[:k].tolist())
        
        # Calculate recall
        retrieved = set(results.ids[:k])
        recall = len(retrieved & ground_truth) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def run_benchmark():
    """Run comprehensive benchmark"""
    print("=" * 70)
    print("VectorDB Performance Benchmark: RaBitQ Accelerated Search")
    print("=" * 70)
    
    # Configuration
    configs = [
        {"n_vectors": 10000, "dimension": 128, "n_queries": 100},
        {"n_vectors": 50000, "dimension": 128, "n_queries": 100},
        {"n_vectors": 100000, "dimension": 128, "n_queries": 100},
    ]
    
    for config in configs:
        n_vectors = config["n_vectors"]
        dimension = config["dimension"]
        n_queries = config["n_queries"]
        
        print(f"\n{'=' * 70}")
        print(f"Dataset: {n_vectors:,} vectors, {dimension}-dim, {n_queries} queries")
        print("=" * 70)
        
        # Generate data
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        
        # Sample queries from dataset (in-distribution)
        query_indices = np.random.choice(n_vectors, n_queries, replace=False)
        queries = vectors[query_indices].copy()
        
        # Create and build database
        print("\n[1] Building database...")
        db = VectorDB(use_memory_storage=True)
        db.add_vectors(vectors)
        
        build_start = time.time()
        db.build(use_true_rabitq=True)
        build_time = time.time() - build_start
        
        print(f"    Build time: {build_time:.2f}s")
        
        # Print stats
        stats = db.get_stats()
        print(f"\n[2] Database Stats:")
        print(f"    IVF clusters: {stats['ivf_index']['num_clusters']}")
        print(f"    True RaBitQ compression: {stats['true_rabitq']['compression_ratio']:.2%}")
        print(f"    Bits per vector: {stats['true_rabitq']['bits_per_vector']}")
        
        # Benchmark accelerated search
        print(f"\n[3] Benchmarking Accelerated Search (IVF + RaBitQ)...")
        k = 10
        
        mean_lat_accel, qps_accel, latencies_accel = benchmark_search(
            db, queries, k, accelerated=True
        )
        recall_accel = compute_recall(db, vectors, queries[:50], k, accelerated=True)
        
        print(f"    Mean latency: {mean_lat_accel:.3f} ms")
        print(f"    P50 latency: {np.percentile(latencies_accel, 50):.3f} ms")
        print(f"    P95 latency: {np.percentile(latencies_accel, 95):.3f} ms")
        print(f"    P99 latency: {np.percentile(latencies_accel, 99):.3f} ms")
        print(f"    Throughput: {qps_accel:.0f} QPS")
        print(f"    Recall@{k}: {recall_accel:.4f}")
        
        # Benchmark standard search
        print(f"\n[4] Benchmarking Standard Search (IVF only)...")
        
        mean_lat_std, qps_std, latencies_std = benchmark_search(
            db, queries, k, accelerated=False
        )
        recall_std = compute_recall(db, vectors, queries[:50], k, accelerated=False)
        
        print(f"    Mean latency: {mean_lat_std:.3f} ms")
        print(f"    P50 latency: {np.percentile(latencies_std, 50):.3f} ms")
        print(f"    P95 latency: {np.percentile(latencies_std, 95):.3f} ms")
        print(f"    P99 latency: {np.percentile(latencies_std, 99):.3f} ms")
        print(f"    Throughput: {qps_std:.0f} QPS")
        print(f"    Recall@{k}: {recall_std:.4f}")
        
        # Comparison
        print(f"\n[5] Comparison:")
        speedup = mean_lat_std / mean_lat_accel if mean_lat_accel > 0 else 0
        qps_improvement = qps_accel / qps_std if qps_std > 0 else 0
        recall_diff = recall_accel - recall_std
        
        print(f"    Latency speedup: {speedup:.2f}x")
        print(f"    QPS improvement: {qps_improvement:.2f}x")
        print(f"    Recall difference: {recall_diff:+.4f}")
        
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


def run_nprobe_analysis():
    """Analyze the effect of nprobe on accelerated search"""
    print("\n" + "=" * 70)
    print("nprobe Analysis: Accelerated Search")
    print("=" * 70)
    
    n_vectors = 50000
    dimension = 128
    n_queries = 50
    k = 10
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)
    queries = vectors[query_indices].copy()
    
    db = VectorDB(use_memory_storage=True)
    db.add_vectors(vectors)
    db.build(use_true_rabitq=True)
    
    nprobe_values = [1, 5, 10, 20, 50, 100]
    
    print(f"\n{'nprobe':>8} {'Latency(ms)':>12} {'QPS':>10} {'Recall@10':>12}")
    print("-" * 50)
    
    for nprobe in nprobe_values:
        latencies = []
        for query in queries:
            start = time.perf_counter()
            db.search(query, k=k, nprobe=nprobe, accelerated=True)
            latencies.append((time.perf_counter() - start) * 1000)
        
        recall = compute_recall(db, vectors, queries, k, accelerated=True)
        mean_lat = np.mean(latencies)
        qps = 1000 / mean_lat
        
        print(f"{nprobe:>8} {mean_lat:>12.3f} {qps:>10.0f} {recall:>12.4f}")


def run_rerank_factor_analysis():
    """Analyze the effect of rerank_factor on search quality"""
    print("\n" + "=" * 70)
    print("Rerank Factor Analysis")
    print("=" * 70)
    
    n_vectors = 50000
    dimension = 128
    n_queries = 50
    k = 10
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)
    queries = vectors[query_indices].copy()
    
    db = VectorDB(use_memory_storage=True)
    db.add_vectors(vectors)
    db.build(use_true_rabitq=True)
    
    rerank_factors = [1, 5, 10, 20, 50]
    
    print(f"\n{'rerank_factor':>14} {'Latency(ms)':>12} {'QPS':>10} {'Recall@10':>12}")
    print("-" * 55)
    
    for rerank_factor in rerank_factors:
        latencies = []
        recalls = []
        
        for i, query in enumerate(queries):
            start = time.perf_counter()
            results = db.search(query, k=k, accelerated=True, rerank_factor=rerank_factor)
            latencies.append((time.perf_counter() - start) * 1000)
            
            # Calculate recall
            distances = np.sum((vectors - query) ** 2, axis=1)
            ground_truth = set(np.argsort(distances)[:k].tolist())
            retrieved = set(results.ids[:k])
            recalls.append(len(retrieved & ground_truth) / k)
        
        mean_lat = np.mean(latencies)
        qps = 1000 / mean_lat
        recall = np.mean(recalls)
        
        print(f"{rerank_factor:>14} {mean_lat:>12.3f} {qps:>10.0f} {recall:>12.4f}")


if __name__ == "__main__":
    run_benchmark()
    run_nprobe_analysis()
    run_rerank_factor_analysis()
