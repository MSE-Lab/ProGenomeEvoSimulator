#!/usr/bin/env python3
"""
并行化进化引擎测试脚本
用于验证并行实现的正确性和性能
"""

import time
import numpy as np
import multiprocessing as mp
from core.genome import create_initial_genome
from core.parallel_evolution_engine import ParallelEvolutionEngine
from core.evolution_engine_optimized import OptimizedEvolutionEngine


def quick_performance_test():
    """快速性能测试"""
    
    print("🧪 Quick Parallel Performance Test")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(42)
    test_genome = create_initial_genome(
        gene_count=500,
        avg_gene_length=800,
        min_gene_length=200
    )
    
    print(f"📊 Test genome: {test_genome.gene_count} genes, {test_genome.size:,} bp")
    print(f"🖥️  Available CPU cores: {mp.cpu_count()}")
    
    # 测试参数
    generations = 20
    mutation_rate = 1e-4
    hgt_rate = 0.01
    recombination_rate = 1e-3
    
    results = {}
    
    # 1. 串行测试
    print("\n🔄 Testing serial evolution...")
    serial_genome = test_genome.copy()
    serial_engine = OptimizedEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate
    )
    
    serial_start = time.time()
    serial_history = serial_engine.evolve_multiple_generations(
        serial_genome, generations, show_progress=False
    )
    serial_time = time.time() - serial_start
    
    results['serial'] = {
        'time': serial_time,
        'speed': generations / serial_time,
        'final_mutations': serial_genome.total_mutations,
        'final_genes': serial_genome.gene_count
    }
    
    print(f"   ✓ Serial: {serial_time:.2f}s ({generations/serial_time:.2f} gen/s)")
    
    # 2. 并行测试
    print("\n⚡ Testing parallel evolution...")
    parallel_genome = test_genome.copy()
    parallel_engine = ParallelEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate,
        num_processes=None,  # 使用所有核心
        enable_progress_sharing=False
    )
    
    parallel_start = time.time()
    parallel_history = parallel_engine.evolve_multiple_generations_parallel(
        parallel_genome, generations, show_progress=False
    )
    parallel_time = time.time() - parallel_start
    
    results['parallel'] = {
        'time': parallel_time,
        'speed': generations / parallel_time,
        'final_mutations': parallel_genome.total_mutations,
        'final_genes': parallel_genome.gene_count,
        'processes': parallel_engine.num_processes
    }
    
    print(f"   ✓ Parallel: {parallel_time:.2f}s ({generations/parallel_time:.2f} gen/s)")
    
    # 3. 结果分析
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / mp.cpu_count() * 100
    
    print(f"\n📈 Performance Results:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Efficiency: {efficiency:.1f}%")
    print(f"   Time saved: {serial_time - parallel_time:.2f}s")
    
    # 4. 结果一致性检查
    mutation_diff = abs(serial_genome.total_mutations - parallel_genome.total_mutations)
    gene_diff = abs(serial_genome.gene_count - parallel_genome.gene_count)
    
    print(f"\n✅ Consistency Check:")
    print(f"   Mutation difference: {mutation_diff}")
    print(f"   Gene count difference: {gene_diff}")
    
    if mutation_diff < 50 and gene_diff < 5:
        print("   ✓ Results are consistent!")
    else:
        print("   ⚠️  Significant differences detected")
    
    return results


def test_different_process_counts():
    """测试不同进程数的性能"""
    
    print("\n🔬 Testing Different Process Counts")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(42)
    test_genome = create_initial_genome(
        gene_count=300,
        avg_gene_length=600,
        min_gene_length=150
    )
    
    generations = 10
    max_processes = min(8, mp.cpu_count())  # 最多测试8个进程
    
    results = []
    
    for num_proc in range(1, max_processes + 1):
        print(f"\n🧮 Testing with {num_proc} process(es)...")
        
        genome_copy = test_genome.copy()
        engine = ParallelEvolutionEngine(
            mutation_rate=1e-4,
            hgt_rate=0.01,
            recombination_rate=1e-3,
            num_processes=num_proc,
            enable_progress_sharing=False
        )
        
        start_time = time.time()
        engine.evolve_multiple_generations_parallel(
            genome_copy, generations, show_progress=False
        )
        elapsed_time = time.time() - start_time
        
        speed = generations / elapsed_time
        results.append({
            'processes': num_proc,
            'time': elapsed_time,
            'speed': speed
        })
        
        print(f"   {num_proc} processes: {elapsed_time:.2f}s ({speed:.2f} gen/s)")
    
    # 分析最优进程数
    best_result = max(results, key=lambda x: x['speed'])
    print(f"\n🏆 Best performance: {best_result['processes']} processes")
    print(f"   Speed: {best_result['speed']:.2f} gen/s")
    
    return results


def test_chunk_size_optimization():
    """测试不同分块大小的影响"""
    
    print("\n📦 Testing Chunk Size Optimization")
    print("=" * 50)
    
    # 创建较大的测试基因组
    np.random.seed(42)
    test_genome = create_initial_genome(
        gene_count=1000,
        avg_gene_length=500,
        min_gene_length=100
    )
    
    generations = 5
    chunk_sizes = [None, 10, 25, 50, 100, 200]  # None表示自动计算
    
    results = []
    
    for chunk_size in chunk_sizes:
        chunk_label = "Auto" if chunk_size is None else str(chunk_size)
        print(f"\n📊 Testing chunk size: {chunk_label}")
        
        genome_copy = test_genome.copy()
        engine = ParallelEvolutionEngine(
            mutation_rate=1e-4,
            hgt_rate=0.005,
            recombination_rate=5e-4,
            num_processes=None,  # 使用所有核心
            chunk_size=chunk_size,
            enable_progress_sharing=False
        )
        
        start_time = time.time()
        history = engine.evolve_multiple_generations_parallel(
            genome_copy, generations, show_progress=False
        )
        elapsed_time = time.time() - start_time
        
        # 计算实际使用的分块大小
        if history:
            actual_chunks = history[0].get('chunks_processed', 0)
            actual_chunk_size = len(genome_copy.genes) // actual_chunks if actual_chunks > 0 else 0
        else:
            actual_chunk_size = 0
        
        speed = generations / elapsed_time
        results.append({
            'chunk_size_setting': chunk_size,
            'actual_chunk_size': actual_chunk_size,
            'chunks_used': actual_chunks,
            'time': elapsed_time,
            'speed': speed
        })
        
        print(f"   Actual chunk size: {actual_chunk_size}, Chunks: {actual_chunks}")
        print(f"   Time: {elapsed_time:.2f}s, Speed: {speed:.2f} gen/s")
    
    # 找到最优分块大小
    best_result = max(results, key=lambda x: x['speed'])
    print(f"\n🎯 Optimal chunk size: {best_result['chunk_size_setting']} "
          f"(actual: {best_result['actual_chunk_size']})")
    
    return results


def main():
    """主测试函数"""
    
    print("🧪 Parallel Evolution Engine Test Suite")
    print("=" * 60)
    print(f"🖥️  System: {mp.cpu_count()} CPU cores available")
    print()
    
    try:
        # 1. 快速性能测试
        quick_results = quick_performance_test()
        
        # 2. 不同进程数测试
        process_results = test_different_process_counts()
        
        # 3. 分块大小优化测试
        chunk_results = test_chunk_size_optimization()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        
        # 总结最佳配置
        print("\n📋 Recommended Configuration:")
        
        if quick_results['parallel']['time'] > 0:
            speedup = quick_results['serial']['time'] / quick_results['parallel']['time']
            print(f"   Expected speedup: {speedup:.2f}x with {mp.cpu_count()} cores")
        
        best_proc = max(process_results, key=lambda x: x['speed'])
        print(f"   Optimal processes: {best_proc['processes']}")
        
        best_chunk = max(chunk_results, key=lambda x: x['speed'])
        chunk_setting = best_chunk['chunk_size_setting']
        chunk_label = "Auto-calculated" if chunk_setting is None else f"{chunk_setting} genes"
        print(f"   Optimal chunk size: {chunk_label}")
        
        print(f"\n💡 For production runs, use:")
        print(f"   ParallelEvolutionEngine(num_processes={best_proc['processes']}, "
              f"chunk_size={chunk_setting})")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()