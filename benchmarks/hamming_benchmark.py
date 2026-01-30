#!/usr/bin/env python3
"""
Benchmark: Hamming 距离计算性能对比

对比四种实现方式的性能:
1. 纯 Python 循环
2. NumPy 向量化
3. Numba JIT
4. Numba JIT + 并行
"""
import sys
sys.path.insert(0, '/Users/niwen/PycharmProjects/coleam00_proj/study_vectordb')

import numpy as np
import time

# ============================================================================
# 实现方式 1: 纯 Python 循环
# ============================================================================
def hamming_pure_python(query_code, database_codes):
    """最慢: 纯 Python 实现"""
    popcount_table = [bin(i).count('1') for i in range(256)]
    n_vectors = len(database_codes)
    n_bytes = len(query_code)
    
    result = []
    for i in range(n_vectors):
        count = 0
        for j in range(n_bytes):
            xor_val = database_codes[i][j] ^ query_code[j]
            count += popcount_table[xor_val]
        result.append(count)
    
    return result


# ============================================================================
# 实现方式 2: NumPy 向量化
# ============================================================================
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def hamming_numpy(query_code, database_codes):
    """中等: NumPy 向量化"""
    xor_result = np.bitwise_xor(database_codes, query_code)
    return np.sum(_POPCOUNT_TABLE[xor_result], axis=1)


# ============================================================================
# 实现方式 3 & 4: Numba JIT (串行 & 并行)
# ============================================================================
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    
    @njit(cache=True)
    def hamming_numba_serial(query_code, database_codes, popcount_table):
        """快: Numba JIT 串行"""
        n_vectors = database_codes.shape[0]
        n_bytes = query_code.shape[0]
        result = np.zeros(n_vectors, dtype=np.int32)
        
        for i in range(n_vectors):
            count = 0
            for j in range(n_bytes):
                xor_val = database_codes[i, j] ^ query_code[j]
                count += popcount_table[xor_val]
            result[i] = count
        
        return result
    
    @njit(parallel=True, cache=True)
    def hamming_numba_parallel(query_code, database_codes, popcount_table):
        """最快: Numba JIT 并行"""
        n_vectors = database_codes.shape[0]
        n_bytes = query_code.shape[0]
        result = np.zeros(n_vectors, dtype=np.int32)
        
        for i in prange(n_vectors):
            count = 0
            for j in range(n_bytes):
                xor_val = database_codes[i, j] ^ query_code[j]
                count += popcount_table[xor_val]
            result[i] = count
        
        return result

except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba 未安装，跳过 Numba 测试")


# ============================================================================
# 基准测试
# ============================================================================
def benchmark(func, *args, n_runs=10, warmup=2, name=""):
    """运行基准测试"""
    # 预热
    for _ in range(warmup):
        func(*args)
    
    # 正式测试
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, result


def main():
    print("=" * 70)
    print("Hamming 距离计算性能基准测试")
    print("=" * 70)
    
    # 测试参数
    n_vectors = 10000
    n_bytes = 16  # 128-bit codes
    
    print(f"\n测试参数:")
    print(f"  - 向量数量: {n_vectors}")
    print(f"  - 编码长度: {n_bytes} bytes ({n_bytes * 8} bits)")
    
    # 生成测试数据
    np.random.seed(42)
    query_code = np.random.randint(0, 256, size=n_bytes, dtype=np.uint8)
    database_codes = np.random.randint(0, 256, size=(n_vectors, n_bytes), dtype=np.uint8)
    
    results = {}
    
    print("\n" + "-" * 70)
    print("运行测试...")
    print("-" * 70)
    
    # 测试 1: 纯 Python (只测试少量数据，太慢了)
    print("\n1. 纯 Python 循环 (测试 1000 个向量)...")
    small_db = database_codes[:1000]
    avg, std, res = benchmark(hamming_pure_python, query_code.tolist(), 
                               small_db.tolist(), n_runs=3, name="Pure Python")
    # 推算 10000 个向量的时间
    estimated_time = avg * 10
    results["Pure Python"] = estimated_time
    print(f"   ⏱️  {avg:.2f} ± {std:.2f} ms (1000 vectors)")
    print(f"   📊 推算 {n_vectors} 向量: ~{estimated_time:.0f} ms")
    
    # 测试 2: NumPy 向量化
    print("\n2. NumPy 向量化...")
    avg, std, res_numpy = benchmark(hamming_numpy, query_code, database_codes, 
                                     n_runs=100, name="NumPy")
    results["NumPy"] = avg
    print(f"   ⏱️  {avg:.3f} ± {std:.3f} ms")
    
    # 测试 3 & 4: Numba (如果可用)
    if NUMBA_AVAILABLE:
        # 预编译 (第一次调用会触发 JIT 编译)
        print("\n3. Numba JIT 串行 (预编译中...)...")
        _ = hamming_numba_serial(query_code, database_codes[:100], _POPCOUNT_TABLE)
        
        avg, std, res_numba = benchmark(
            hamming_numba_serial, query_code, database_codes, _POPCOUNT_TABLE,
            n_runs=100, name="Numba Serial"
        )
        results["Numba JIT"] = avg
        print(f"   ⏱️  {avg:.3f} ± {std:.3f} ms")
        
        # 验证结果正确性
        if not np.allclose(res_numpy, res_numba):
            print("   ⚠️  结果不匹配!")
        
        print("\n4. Numba JIT 并行 (预编译中...)...")
        _ = hamming_numba_parallel(query_code, database_codes[:100], _POPCOUNT_TABLE)
        
        avg, std, res_parallel = benchmark(
            hamming_numba_parallel, query_code, database_codes, _POPCOUNT_TABLE,
            n_runs=100, name="Numba Parallel"
        )
        results["Numba 并行"] = avg
        print(f"   ⏱️  {avg:.3f} ± {std:.3f} ms")
        
        # 验证结果正确性
        if not np.allclose(res_numpy, res_parallel):
            print("   ⚠️  结果不匹配!")
    else:
        print("\n3-4. Numba 不可用，跳过")
    
    # ============================================================================
    # 结果汇总
    # ============================================================================
    print("\n" + "=" * 70)
    print("性能对比汇总")
    print("=" * 70)
    
    baseline = results.get("Pure Python", 1)
    
    print(f"\n{'方法':<20} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 50)
    
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        bar = "█" * min(int(speedup / 5), 20)
        print(f"{name:<20} {time_ms:>10.3f} ms   {speedup:>6.1f}x  {bar}")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if NUMBA_AVAILABLE and "Numba 并行" in results:
        numpy_speedup = baseline / results["NumPy"]
        numba_speedup = baseline / results["Numba JIT"]
        parallel_speedup = baseline / results["Numba 并行"]
        
        print(f"""
场景: {n_vectors} 个向量的 Hamming 距离计算

纯 Python 循环:  ~{results['Pure Python']:.0f} ms
NumPy 向量化:    ~{results['NumPy']:.1f} ms  ({numpy_speedup:.0f}x 加速)
Numba JIT:       ~{results['Numba JIT']:.2f} ms  ({numba_speedup:.0f}x 加速)
Numba + 并行:    ~{results['Numba 并行']:.2f} ms  ({parallel_speedup:.0f}x 加速)
""")
    else:
        print(f"""
场景: {n_vectors} 个向量的 Hamming 距离计算

纯 Python 循环:  ~{results.get('Pure Python', 'N/A')} ms
NumPy 向量化:    ~{results.get('NumPy', 'N/A'):.1f} ms

提示: 安装 Numba 可以获得更好的性能
  pip install numba
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
