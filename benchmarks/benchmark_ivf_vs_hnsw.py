"""
Benchmark: IVF vs HNSW Index Comparison

This benchmark compares:
1. Build time
2. Search latency
3. Recall @ k
4. Memory usage
"""
import numpy as np
import time
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

from vectordb.core.config import VectorDBConfig, IVFIndexConfig, HNSWIndexConfig
from vectordb.index.ivf_index import IVFIndex
from vectordb.index.hnsw_index import HNSWIndex

# Optional HNSWLib
try:
    from vectordb.index.hnswlib_index import HNSWLibIndex
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    index_type: str
    build_time: float
    avg_search_latency: float
    recall_at_k: float
    memory_bytes: int


def compute_recall(results: List[Tuple[int, float]], ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k"""
    result_ids = set(r[0] for r in results[:k])
    gt_ids = set(ground_truth[:k])
    return len(result_ids & gt_ids) / k


def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Brute force nearest neighbor search"""
    distances = np.sum((vectors - query) ** 2, axis=1)
    return np.argsort(distances)[:k]


def benchmark_ivf(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    ground_truth: List[np.ndarray],
    nlist: int = 100,
    nprobe: int = 10
) -> BenchmarkResult:
    """Benchmark IVF index"""
    print(f"\n{'='*60}")
    print(f"Benchmarking IVF Index (nlist={nlist}, nprobe={nprobe})")
    print(f"{'='*60}")
    
    # Build
    index = IVFIndex(nlist=nlist, nprobe=nprobe, metric="l2")
    start_time = time.time()
    index.build(vectors)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.3f}s")
    
    # Search
    search_times = []
    recalls = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        results = index.search(query, k=k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        recall = compute_recall(results, ground_truth[i], k)
        recalls.append(recall)
    
    avg_search_latency = np.mean(search_times) * 1000  # Convert to ms
    avg_recall = np.mean(recalls)
    
    print(f"Avg search latency: {avg_search_latency:.3f}ms")
    print(f"Recall@{k}: {avg_recall:.4f}")
    
    # Memory estimate
    memory_bytes = sys.getsizeof(index.vectors) if index.vectors is not None else 0
    memory_bytes += sys.getsizeof(index.centroids) if index.centroids is not None else 0
    
    return BenchmarkResult(
        index_type="IVF",
        build_time=build_time,
        avg_search_latency=avg_search_latency,
        recall_at_k=avg_recall,
        memory_bytes=memory_bytes
    )


def benchmark_hnsw(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    ground_truth: List[np.ndarray],
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50
) -> BenchmarkResult:
    """Benchmark HNSW index (pure Python)"""
    print(f"\n{'='*60}")
    print(f"Benchmarking HNSW Index (M={M}, ef_construction={ef_construction}, ef_search={ef_search})")
    print(f"{'='*60}")
    
    # Build
    config = HNSWIndexConfig(M=M, ef_construction=ef_construction, ef_search=ef_search)
    index = HNSWIndex(config)
    start_time = time.time()
    index.build(vectors)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.3f}s")
    
    # Search
    search_times = []
    recalls = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        results = index.search(query, k=k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        recall = compute_recall(results, ground_truth[i], k)
        recalls.append(recall)
    
    avg_search_latency = np.mean(search_times) * 1000  # Convert to ms
    avg_recall = np.mean(recalls)
    
    print(f"Avg search latency: {avg_search_latency:.3f}ms")
    print(f"Recall@{k}: {avg_recall:.4f}")
    
    # Memory estimate
    memory_bytes = sys.getsizeof(index.vectors) if index.vectors is not None else 0
    for layer in index.layers:
        memory_bytes += sys.getsizeof(layer)
    
    return BenchmarkResult(
        index_type="HNSW",
        build_time=build_time,
        avg_search_latency=avg_search_latency,
        recall_at_k=avg_recall,
        memory_bytes=memory_bytes
    )


def benchmark_hnswlib(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    ground_truth: List[np.ndarray],
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50
) -> BenchmarkResult:
    """Benchmark HNSWLib index (C++ implementation)"""
    if not HNSWLIB_AVAILABLE:
        raise ImportError("hnswlib not available")
    
    print(f"\n{'='*60}")
    print(f"Benchmarking HNSWLib (C++) (M={M}, ef_c={ef_construction}, ef_s={ef_search})")
    print(f"{'='*60}")
    
    # Build
    config = HNSWIndexConfig(M=M, ef_construction=ef_construction, ef_search=ef_search)
    index = HNSWLibIndex(config, max_elements=len(vectors))
    start_time = time.time()
    index.build(vectors)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.3f}s")
    
    # Search
    search_times = []
    recalls = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        results = index.search(query, k=k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        recall = compute_recall(results, ground_truth[i], k)
        recalls.append(recall)
    
    avg_search_latency = np.mean(search_times) * 1000  # Convert to ms
    avg_recall = np.mean(recalls)
    
    print(f"Avg search latency: {avg_search_latency:.3f}ms")
    print(f"Recall@{k}: {avg_recall:.4f}")
    
    memory_bytes = sys.getsizeof(index.vectors) if index.vectors is not None else 0
    
    return BenchmarkResult(
        index_type="HNSWLib",
        build_time=build_time,
        avg_search_latency=avg_search_latency,
        recall_at_k=avg_recall,
        memory_bytes=memory_bytes
    )


def run_benchmarks(
    n_vectors: int = 10000,
    dimension: int = 128,
    n_queries: int = 100,
    k: int = 10,
    skip_python_hnsw: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run all benchmarks"""
    print(f"\n{'#'*60}")
    print(f"# IVF vs HNSW vs HNSWLib Benchmark")
    print(f"# Vectors: {n_vectors}, Dimension: {dimension}")
    print(f"# Queries: {n_queries}, k: {k}")
    print(f"{'#'*60}")
    
    # Generate data
    np.random.seed(42)
    vectors = np.random.random((n_vectors, dimension)).astype(np.float32)
    queries = np.random.random((n_queries, dimension)).astype(np.float32)
    
    # Compute ground truth
    print("\nComputing ground truth (brute force)...")
    ground_truth = [brute_force_search(vectors, q, k) for q in queries]
    
    results = {}
    
    # IVF benchmarks
    for nlist, nprobe in [(100, 10), (100, 20)]:
        result = benchmark_ivf(vectors, queries, k, ground_truth, nlist=nlist, nprobe=nprobe)
        results[f"IVF(nlist={nlist},nprobe={nprobe})"] = result
    
    # Pure Python HNSW (skip if requested due to long build time)
    if not skip_python_hnsw:
        result = benchmark_hnsw(vectors, queries, k, ground_truth, M=16, ef_construction=100, ef_search=50)
        results["HNSW-Python(M=16,ef=50)"] = result
    
    # HNSWLib benchmarks (C++)
    if HNSWLIB_AVAILABLE:
        for M, ef_c, ef_s in [(16, 200, 50), (16, 200, 100)]:
            result = benchmark_hnswlib(vectors, queries, k, ground_truth, M=M, ef_construction=ef_c, ef_search=ef_s)
            results[f"HNSWLib(M={M},ef_c={ef_c},ef_s={ef_s})"] = result
    else:
        print("\n[!] HNSWLib not available. Install with: pip install hnswlib")
    
    return results


def print_summary(results: Dict[str, BenchmarkResult]) -> None:
    """Print summary table"""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Index Type':<35} {'Build(s)':<12} {'Search(ms)':<12} {'Recall@k':<12}")
    print(f"{'-'*80}")
    
    for name, result in results.items():
        print(f"{name:<35} {result.build_time:<12.3f} {result.avg_search_latency:<12.3f} {result.recall_at_k:<12.4f}")


def generate_markdown_report(results: Dict[str, BenchmarkResult], n_vectors: int, dimension: int, k: int) -> str:
    """Generate markdown report"""
    report = f"""# IVF vs HNSW Benchmark Results

## Test Configuration
- **Number of vectors**: {n_vectors}
- **Vector dimension**: {dimension}
- **k (neighbors to retrieve)**: {k}

## Results

| Index Type | Build Time (s) | Search Latency (ms) | Recall@{k} |
|------------|----------------|---------------------|------------|
"""
    for name, result in results.items():
        report += f"| {name} | {result.build_time:.3f} | {result.avg_search_latency:.3f} | {result.recall_at_k:.4f} |\n"
    
    report += """
## Analysis

### Build Time
- IVF typically builds faster for smaller datasets due to simpler k-means clustering
- HNSW build time increases with M and ef_construction parameters

### Search Latency
- HNSW generally provides faster search with higher recall
- IVF search time depends on nprobe; higher nprobe = better recall but slower

### Recall
- HNSW typically achieves higher recall at similar latency
- Both benefit from parameter tuning (nprobe for IVF, ef_search for HNSW)

## Recommendations
- **Small datasets (<10K vectors)**: IVF is simpler and faster to build
- **Large datasets with high recall requirements**: HNSW with tuned parameters
- **Memory-constrained environments**: IVF uses less memory
"""
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark IVF vs HNSW")
    parser.add_argument("--n-vectors", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dimension", type=int, default=128, help="Vector dimension")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--output", type=str, default=None, help="Output markdown file")
    parser.add_argument("--skip-python-hnsw", action="store_true", help="Skip slow Python HNSW")
    
    args = parser.parse_args()
    
    results = run_benchmarks(
        n_vectors=args.n_vectors,
        dimension=args.dimension,
        n_queries=args.n_queries,
        k=args.k,
        skip_python_hnsw=args.skip_python_hnsw
    )
    
    print_summary(results)
    
    if args.output:
        report = generate_markdown_report(results, args.n_vectors, args.dimension, args.k)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
