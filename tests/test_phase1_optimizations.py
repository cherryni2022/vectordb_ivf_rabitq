#!/usr/bin/env python3
"""
Test script to verify Phase 1 optimizations for IVF and RaBitQ
"""
import sys
sys.path.insert(0, '/Users/niwen/PycharmProjects/coleam00_proj/study_vectordb')

import numpy as np
import time
from vectordb.index.ivf_index import IVFIndex
from vectordb.quantization.true_rabitq import TrueRaBitQ

def test_ivf_vectorized_kmeans():
    """Test vectorized K-Means centroid update"""
    print("=" * 60)
    print("Testing IVF Vectorized K-Means")
    print("=" * 60)
    
    np.random.seed(42)
    n_vectors = 10000
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Test standard k-means
    index = IVFIndex(nlist=50, nprobe=5)
    
    start = time.time()
    index.build(vectors)
    build_time = time.time() - start
    
    print(f"✓ Index built successfully")
    print(f"  - Vectors: {n_vectors}")
    print(f"  - Clusters: {index.nlist}")
    print(f"  - Build time: {build_time:.3f}s")
    
    # Test search
    query = np.random.randn(dimension).astype(np.float32)
    results = index.search(query, k=10)
    
    print(f"✓ Search completed, returned {len(results)} results")
    print(f"  - Top result: idx={results[0][0]}, dist={results[0][1]:.4f}")
    
    return True

def test_ivf_minibatch_kmeans():
    """Test Mini-batch K-Means"""
    print("\n" + "=" * 60)
    print("Testing IVF Mini-batch K-Means")
    print("=" * 60)
    
    np.random.seed(42)
    n_vectors = 50000
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Test with mini-batch enabled
    index = IVFIndex(nlist=100, nprobe=10, use_minibatch=True, minibatch_size=1024)
    
    start = time.time()
    index.build(vectors)
    build_time = time.time() - start
    
    print(f"✓ Index built with mini-batch K-Means")
    print(f"  - Vectors: {n_vectors}")
    print(f"  - Clusters: {index.nlist}")
    print(f"  - Build time: {build_time:.3f}s")
    
    # Verify cluster distribution
    cluster_sizes = [len(inv_list) for inv_list in index.inverted_lists]
    print(f"  - Cluster size: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
    
    # Test search
    query = np.random.randn(dimension).astype(np.float32)
    results = index.search(query, k=10)
    
    print(f"✓ Search completed, returned {len(results)} results")
    
    return True

def test_ivf_chunked_distance():
    """Test chunked distance computation"""
    print("\n" + "=" * 60)
    print("Testing IVF Chunked Distance Computation")
    print("=" * 60)
    
    np.random.seed(42)
    n_vectors = 15000  # Larger than default chunk_size (10000)
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    index = IVFIndex(nlist=50, nprobe=5, chunk_size=5000)
    
    start = time.time()
    index.build(vectors)
    build_time = time.time() - start
    
    print(f"✓ Index built with chunked distance computation")
    print(f"  - Vectors: {n_vectors}")
    print(f"  - Chunk size: {index.chunk_size}")
    print(f"  - Build time: {build_time:.3f}s")
    
    return True

def test_rabitq_static_popcount():
    """Test static popcount table"""
    print("\n" + "=" * 60)
    print("Testing RaBitQ Static Popcount Table")
    print("=" * 60)
    
    np.random.seed(42)
    n_vectors = 10000
    dimension = 256
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Verify class-level popcount table exists
    assert hasattr(TrueRaBitQ, '_POPCOUNT_TABLE'), "Static popcount table not found!"
    print(f"✓ Static popcount table exists: shape={TrueRaBitQ._POPCOUNT_TABLE.shape}")
    
    quantizer = TrueRaBitQ(dimension)
    quantizer.train(vectors[:1000])
    
    start = time.time()
    binary_codes, norms = quantizer.encode(vectors)
    encode_time = time.time() - start
    
    print(f"✓ Encoding completed")
    print(f"  - Vectors: {n_vectors}")
    print(f"  - Binary codes shape: {binary_codes.shape}")
    print(f"  - Encode time: {encode_time:.3f}s")
    
    # Test Hamming distance computation
    query = np.random.randn(dimension).astype(np.float32)
    query_code, query_norm = quantizer.encode_query(query)
    
    start = time.time()
    for _ in range(100):
        distances = quantizer.compute_hamming_distance(query_code, binary_codes)
    hamming_time = (time.time() - start) / 100
    
    print(f"✓ Hamming distance computed")
    print(f"  - Avg time per query: {hamming_time*1000:.3f}ms")
    
    return True

def test_ivf_rabitq_integration():
    """Test IVF + RaBitQ integration"""
    print("\n" + "=" * 60)
    print("Testing IVF + RaBitQ Integration")
    print("=" * 60)
    
    np.random.seed(42)
    n_vectors = 10000
    dimension = 256
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Build IVF index
    index = IVFIndex(nlist=50, nprobe=10)
    index.build(vectors)
    
    # Build RaBitQ quantizer
    quantizer = TrueRaBitQ(dimension)
    quantizer.train(vectors[:1000])
    quantizer.encode(vectors)
    
    # Test search with RaBitQ
    query = np.random.randn(dimension).astype(np.float32)
    
    # Regular search
    start = time.time()
    results_regular = index.search(query, k=10)
    regular_time = time.time() - start
    
    # RaBitQ search
    start = time.time()
    results_rabitq = index.search_with_rabitq(query, k=10, quantizer=quantizer)
    rabitq_time = time.time() - start
    
    print(f"✓ Search comparison:")
    print(f"  - Regular search: {regular_time*1000:.3f}ms")
    print(f"  - RaBitQ search: {rabitq_time*1000:.3f}ms")
    print(f"  - Regular top result: idx={results_regular[0][0]}")
    print(f"  - RaBitQ top result: idx={results_rabitq[0][0]}")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("Phase 1 Optimization Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_ivf_vectorized_kmeans()
    except Exception as e:
        print(f"✗ IVF Vectorized K-Means test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_ivf_minibatch_kmeans()
    except Exception as e:
        print(f"✗ IVF Mini-batch K-Means test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_ivf_chunked_distance()
    except Exception as e:
        print(f"✗ IVF Chunked Distance test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_rabitq_static_popcount()
    except Exception as e:
        print(f"✗ RaBitQ Static Popcount test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_ivf_rabitq_integration()
    except Exception as e:
        print(f"✗ IVF + RaBitQ Integration test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All Phase 1 optimization tests PASSED!")
    else:
        print("✗ Some tests FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
