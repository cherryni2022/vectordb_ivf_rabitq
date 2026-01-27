"""
Example Usage of VectorDB with IVF + RaBitQ

This script demonstrates the basic usage of the vector database MVP:
1. Creating a database
2. Adding vectors
3. Building IVF index
4. Querying for top-k similar vectors
"""
import numpy as np
from vectordb import VectorDB


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create an in-memory database
    db = VectorDB(use_memory_storage=True)

    # Generate some random vectors (simulating embeddings)
    n_vectors = 1000
    dimension = 128
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

    print(f"Adding {n_vectors} vectors of dimension {dimension}...")
    db.add_vectors(vectors)

    # Build the IVF index
    print("Building IVF index...")
    db.build()

    # Print statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  IVF clusters: {stats['ivf_index']['num_clusters']}")
    print(f"  Avg vectors per cluster: {stats['ivf_index']['avg_vectors_per_cluster']:.2f}")

    # Search for similar vectors
    query = vectors[0]
    print(f"\nSearching for top-5 vectors similar to query vector 0...")
    results = db.search(query, k=5)

    print(f"\nTop-5 Results:")
    for i, (id_, dist) in enumerate(zip(results.ids, results.distances)):
        print(f"  {i+1}. ID: {id_}, Distance: {dist:.4f}")

    print()


def example_with_metadata():
    """Example with metadata"""
    print("=" * 60)
    print("Example 2: Vector Database with Metadata")
    print("=" * 60)

    db = VectorDB(use_memory_storage=True)

    # Create vectors for documents
    n_docs = 100
    dimension = 64
    vectors = np.random.randn(n_docs, dimension).astype(np.float32)

    # Generate document metadata
    ids = [f"doc_{i}" for i in range(n_docs)]
    metadata = []
    for i in range(n_docs):
        category = "tech" if i < 50 else "science"
        metadata.append({
            "category": category,
            "title": f"Document {i}",
            "year": 2020 + (i % 5)
        })

    print(f"Adding {n_docs} documents with metadata...")
    db.add_vectors(vectors, ids=ids, metadata=metadata)
    db.build()

    # Search and show metadata
    query = vectors[0]
    results = db.search(query, k=5)

    print(f"\nSearch results for 'doc_0':")
    for i, (id_, dist, meta) in enumerate(zip(results.ids, results.distances, results.metadata)):
        print(f"  {i+1}. ID: {id_}")
        print(f"     Distance: {dist:.4f}")
        print(f"     Category: {meta.get('category')}, Year: {meta.get('year')}")

    print()


def example_batch_search():
    """Example with batch search"""
    print("=" * 60)
    print("Example 3: Batch Search")
    print("=" * 60)

    db = VectorDB(use_memory_storage=True)

    # Add vectors
    vectors = np.random.randn(500, 128).astype(np.float32)
    db.add_vectors(vectors)
    db.build()

    # Batch search multiple queries
    queries = vectors[:10]  # First 10 vectors as queries
    print(f"Searching for top-3 results for {len(queries)} queries...")

    results_list = db.search_batch(queries, k=3)

    print("\nResults:")
    for i, results in enumerate(results_list):
        print(f"  Query {i}: Top result ID={results.ids[0]}, Distance={results.distances[0]:.4f}")

    print()


def example_custom_config():
    """Example with custom configuration"""
    print("=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)

    from vectordb import IVFIndexConfig, RaBitQConfig, VectorDBConfig

    # Create custom configuration
    config = VectorDBConfig(
        ivf=IVFIndexConfig(
            nlist=50,      # Number of clusters
            nprobe=5,      # Number of clusters to search
            metric="l2"    # Distance metric (l2 or ip)
        ),
        raq=RaBitQConfig(
            nsubq=8,       # Number of sub-quantizers
            nbits=8,       # Bits per sub-quantizer
            use_rabitq=True  # Enable RaBitQ quantization
        ),
        dimension=128,
        use_quantization=True
    )

    db = VectorDB(config=config, use_memory_storage=True)

    vectors = np.random.randn(200, 128).astype(np.float32)
    db.add_vectors(vectors)
    db.build()

    stats = db.get_stats()
    print(f"\nConfiguration:")
    print(f"  IVF nlist: {stats['ivf_index']['nlist']}")
    print(f"  IVF nprobe: {stats['ivf_index']['nprobe']}")
    print(f"  Quantization nsubq: {stats['quantization']['nsubq']}")
    print(f"  Compression ratio: {stats['quantization']['compression_ratio']:.2%}")

    query = vectors[0]
    results = db.search(query, k=5)
    print(f"\nSearch results: {len(results.ids)} results found")

    print()


def example_nprobe_tuning():
    """Example showing effect of nprobe on search quality"""
    print("=" * 60)
    print("Example 5: Nprobe Tuning")
    print("=" * 60)

    db = VectorDB(use_memory_storage=True)

    vectors = np.random.randn(1000, 64).astype(np.float32)
    db.add_vectors(vectors)
    db.build()

    query = vectors[0]

    print(f"Searching with different nprobe values:")
    for nprobe in [1, 5, 10, 20, 50]:
        results = db.search(query, k=5, nprobe=nprobe)
        print(f"  nprobe={nprobe:2d}: Top distance = {results.distances[0]:.6f}")

    print()


def example_with_quantization():
    """Example showing quantization effects"""
    print("=" * 60)
    print("Example 6: With and Without Quantization")
    print("=" * 60)

    # Create two databases: one with quantization, one without
    vectors = np.random.randn(200, 128).astype(np.float32)

    # Without quantization
    db_no_q = VectorDB(use_memory_storage=True)
    from vectordb import VectorDBConfig, IVFIndexConfig, RaBitQConfig
    config_no_q = VectorDBConfig(
        ivf=IVFIndexConfig(nlist=20, nprobe=5),
        raq=RaBitQConfig(),
        use_quantization=False
    )
    db_no_q.config = config_no_q
    db_no_q.add_vectors(vectors)
    db_no_q.build()

    # With quantization
    db_with_q = VectorDB(use_memory_storage=True)
    config_with_q = VectorDBConfig(
        ivf=IVFIndexConfig(nlist=20, nprobe=5),
        raq=RaBitQConfig(nsubq=8, nbits=8),
        use_quantization=True
    )
    db_with_q.config = config_with_q
    db_with_q.add_vectors(vectors)
    db_with_q.build()

    # Compare
    query = vectors[0]
    results_no_q = db_no_q.search(query, k=10)
    results_with_q = db_with_q.search(query, k=10)

    print(f"Without quantization: Top distance = {results_no_q.distances[0]:.6f}")
    print(f"With quantization: Top distance = {results_with_q.distances[0]:.6f}")

    stats_with_q = db_with_q.get_stats()
    print(f"\nQuantization compression ratio: {stats_with_q['quantization']['compression_ratio']:.2%}")

    print()


def example_persistence():
    """Example with save and load"""
    print("=" * 60)
    print("Example 7: Persistence (Save/Load)")
    print("=" * 60)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        # Create and populate database
        db = VectorDB(storage_path=temp_dir)
        vectors = np.random.randn(100, 64).astype(np.float32)
        db.add_vectors(vectors)
        db.build()

        # Save
        print(f"Saving database to {temp_dir}...")
        db.save()

        # Create new instance and load
        db2 = VectorDB(storage_path=temp_dir)
        print("Loading database...")
        db2.load()

        # Verify
        stats = db2.get_stats()
        print(f"\nLoaded database:")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Is built: {stats['is_built']}")

        # Search should work
        results = db2.search(vectors[0], k=5)
        print(f"  Search results: {len(results.ids)} found")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nTemporary directory cleaned up.")

    print()


def example_convenience_function():
    """Example using the convenience create_vector_db function"""
    print("=" * 60)
    print("Example 8: Convenience Function")
    print("=" * 60)

    from vectordb.core.vector_db import create_vector_db

    # Quick setup with common parameters
    db = create_vector_db(
        dimension=128,
        nlist=50,
        nprobe=10,
        use_quantization=True,
        storage_path="./my_vector_db"
    )

    vectors = np.random.randn(500, 128).astype(np.float32)
    db.add_vectors(vectors)
    db.build()

    results = db.search(vectors[0], k=10)
    print(f"Created database with 500 vectors")
    print(f"Search returned {len(results.ids)} results")

    # Clean up
    import shutil
    if "./my_vector_db" in db.storage.base_path.as_posix():
        shutil.rmtree("./my_vector_db", ignore_errors=True)

    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_with_metadata()
    example_batch_search()
    example_custom_config()
    example_nprobe_tuning()
    example_with_quantization()
    example_persistence()
    example_convenience_function()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
