#!/usr/bin/env python3
"""
Comprehensive test script for all optimization phases
"""
import sys
sys.path.insert(0, '/Users/niwen/PycharmProjects/coleam00_proj/study_vectordb')

import numpy as np
import time
import tempfile
import os

def test_numba_utils():
    """Test Phase 2: Numba utilities"""
    print("\n" + "=" * 60)
    print("Testing Numba Utilities (Phase 2)")
    print("=" * 60)
    
    from vectordb.utils.numba_utils import (
        fast_hamming_distance,
        fast_l2_distances,
        is_numba_available,
        get_backend_info,
    )
    
    info = get_backend_info()
    print(f"Numba available: {info['numba_available']}")
    if info['numba_available']:
        print(f"Numba version: {info['numba_version']}")
    
    # Test L2 distances
    np.random.seed(42)
    queries = np.random.randn(10, 128).astype(np.float32)
    points = np.random.randn(1000, 128).astype(np.float32)
    
    start = time.time()
    for _ in range(100):
        distances = fast_l2_distances(queries, points)
    l2_time = (time.time() - start) / 100
    
    print(f"✓ L2 distances computed")
    print(f"  - Query shape: {queries.shape}, Points shape: {points.shape}")
    print(f"  - Avg time: {l2_time*1000:.3f}ms")
    
    # Test Hamming distance
    query_code = np.random.randint(0, 256, size=16, dtype=np.uint8)
    db_codes = np.random.randint(0, 256, size=(10000, 16), dtype=np.uint8)
    
    start = time.time()
    for _ in range(100):
        hamming = fast_hamming_distance(query_code, db_codes)
    hamming_time = (time.time() - start) / 100
    
    print(f"✓ Hamming distances computed")
    print(f"  - DB size: {len(db_codes)}")
    print(f"  - Avg time: {hamming_time*1000:.3f}ms")
    
    return True

def test_backends():
    """Test Phase 3: Compute backends"""
    print("\n" + "=" * 60)
    print("Testing Compute Backends (Phase 3)")
    print("=" * 60)
    
    from vectordb.backends import (
        list_backends,
        get_backend,
        get_best_backend,
        is_pytorch_available,
        is_cuda_available,
        get_pytorch_device_info,
    )
    
    print(f"Available backends: {list_backends()}")
    print(f"PyTorch available: {is_pytorch_available()}")
    print(f"CUDA available: {is_cuda_available()}")
    
    if is_pytorch_available():
        info = get_pytorch_device_info()
        print(f"PyTorch info: {info}")
    
    # Test NumPy backend
    backend = get_backend('numpy')
    print(f"\n✓ Using backend: {backend.name} (GPU: {backend.is_gpu})")
    
    np.random.seed(42)
    vectors = np.random.randn(1000, 128).astype(np.float32)
    centroids = np.random.randn(10, 128).astype(np.float32)
    
    # Test distance computation
    distances = backend.compute_l2_distances(vectors[:10], centroids)
    print(f"  - L2 distances shape: {distances.shape}")
    
    # Test kmeans iteration
    new_centroids, assignments = backend.kmeans_iteration(vectors, centroids)
    print(f"  - K-Means iteration: centroids {new_centroids.shape}, assignments {assignments.shape}")
    
    return True

def test_mmap_storage():
    """Test Phase 4: Memory-mapped storage"""
    print("\n" + "=" * 60)
    print("Testing Memory-Mapped Storage (Phase 4)")
    print("=" * 60)
    
    from vectordb.storage import MemoryMappedVectorStore, ChunkedVectorLoader
    
    np.random.seed(42)
    n_vectors = 10000
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.mmap', delete=False) as f:
        mmap_path = f.name
    
    try:
        # Test create
        store = MemoryMappedVectorStore(mmap_path, dimension)
        mmap = store.create(n_vectors)
        mmap[:] = vectors
        store.flush()
        
        print(f"✓ Memory-mapped file created")
        print(f"  - Shape: {store.shape}")
        print(f"  - File size: {os.path.getsize(mmap_path) / 1024 / 1024:.2f} MB")
        
        store.close()
        
        # Test load
        store2 = MemoryMappedVectorStore(mmap_path, dimension)
        loaded = store2.load()
        
        print(f"✓ Memory-mapped file loaded")
        print(f"  - Vectors loaded: {store2.n_vectors}")
        
        # Test batch access
        batch = store2.get_batch([0, 1, 2, 100, 200])
        assert batch.shape == (5, dimension)
        print(f"✓ Batch access works: shape={batch.shape}")
        
        # Test slice access
        slice_data = store2.get_slice(0, 100)
        assert slice_data.shape == (100, dimension)
        print(f"✓ Slice access works: shape={slice_data.shape}")
        
        store2.close()
        
        # Test chunked loader
        loader = ChunkedVectorLoader(vectors, chunk_size=2000)
        chunks_processed = 0
        for chunk, start_idx in loader:
            chunks_processed += 1
        
        print(f"✓ Chunked loader works: {chunks_processed} chunks processed")
        
    finally:
        os.unlink(mmap_path)
    
    return True

def test_parallel_search():
    """Test Phase 4: Parallel search and caching"""
    print("\n" + "=" * 60)
    print("Testing Parallel Search & Caching (Phase 4)")
    print("=" * 60)
    
    from vectordb.search import ParallelSearcher, SearchCache, AdaptiveNProbe
    from vectordb.index.ivf_index import IVFIndex
    
    np.random.seed(42)
    n_vectors = 10000
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Build index
    index = IVFIndex(nlist=50, nprobe=5)
    index.build(vectors)
    
    # Test search cache
    cache = SearchCache(max_size=100)
    query = np.random.randn(dimension).astype(np.float32)
    
    # First search - cache miss
    result = cache.get(query, k=10, nprobe=5)
    assert result is None
    
    # Store in cache
    search_result = index.search(query, k=10)
    cache.put(query, 10, 5, search_result)
    
    # Second search - cache hit
    result = cache.get(query, k=10, nprobe=5)
    assert result is not None
    
    stats = cache.get_stats()
    print(f"✓ Search cache works")
    print(f"  - Total queries: {stats.total_queries}")
    print(f"  - Cache hits: {stats.cache_hits}")
    print(f"  - Hit rate: {stats.cache_hit_rate:.2%}")
    
    # Test parallel searcher
    searcher = ParallelSearcher(
        search_fn=lambda q, k: index.search(q, k),
        n_threads=4,
        cache=cache
    )
    
    queries = np.random.randn(20, dimension).astype(np.float32)
    
    start = time.time()
    results = searcher.search_batch(queries, k=10, nprobe=5)
    batch_time = time.time() - start
    
    print(f"✓ Parallel batch search works")
    print(f"  - Queries: {len(queries)}")
    print(f"  - Total time: {batch_time*1000:.1f}ms")
    print(f"  - Avg per query: {batch_time*1000/len(queries):.1f}ms")
    
    searcher.shutdown()
    
    # Test adaptive nprobe
    adaptive = AdaptiveNProbe(base_nprobe=10, min_nprobe=1, max_nprobe=50)
    centroid_distances = np.random.rand(100).astype(np.float32)
    centroid_distances.sort()
    
    recommended_nprobe = adaptive.estimate_nprobe(centroid_distances)
    print(f"✓ Adaptive nprobe works")
    print(f"  - Recommended nprobe: {recommended_nprobe}")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("VectorDB Optimization Tests - All Phases")
    print("=" * 60)
    
    all_passed = True
    
    # Phase 2: Numba
    try:
        all_passed &= test_numba_utils()
    except Exception as e:
        print(f"✗ Numba utils test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Phase 3: Backends
    try:
        all_passed &= test_backends()
    except Exception as e:
        print(f"✗ Backends test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Phase 4: Storage
    try:
        all_passed &= test_mmap_storage()
    except Exception as e:
        print(f"✗ Memory-mapped storage test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Phase 4: Parallel Search
    try:
        all_passed &= test_parallel_search()
    except Exception as e:
        print(f"✗ Parallel search test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All optimization tests PASSED!")
    else:
        print("✗ Some tests FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
