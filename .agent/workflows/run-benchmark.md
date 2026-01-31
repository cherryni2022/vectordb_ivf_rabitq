---
description: Run benchmark comparing IVF, HNSW, and HNSWLib indexes
---

# Run Benchmark

This workflow runs the index comparison benchmark to evaluate performance of different index implementations.

## Quick Run (Default Parameters)

// turbo
1. Activate the virtual environment and run benchmark:
```bash
source .venv/bin/activate && python benchmarks/benchmark_ivf_vs_hnsw.py
```

## Custom Parameters

2. Run with custom vector count and dimensions:
```bash
source .venv/bin/activate && python benchmarks/benchmark_ivf_vs_hnsw.py --n-vectors 10000 --dimension 128 --n-queries 100 --k 10
```

3. Skip slow Python HNSW (recommended for large datasets):
```bash
source .venv/bin/activate && python benchmarks/benchmark_ivf_vs_hnsw.py --n-vectors 10000 --skip-python-hnsw
```

4. Save results to markdown:
```bash
source .venv/bin/activate && python benchmarks/benchmark_ivf_vs_hnsw.py --output docs/benchmark_results.md
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-vectors` | Number of vectors to index | 10000 |
| `--dimension` | Vector dimension | 128 |
| `--n-queries` | Number of search queries | 100 |
| `--k` | Neighbors to retrieve | 10 |
| `--skip-python-hnsw` | Skip slow Python HNSW | False |
| `--output` | Save markdown report | None |
