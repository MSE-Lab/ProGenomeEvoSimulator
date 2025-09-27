#!/usr/bin/env python3
"""
测试并行优化效果
对比原始并行实现和优化后的并行实现

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import time
import numpy as np
from core.genome import Genome, Gene
from core.unified_evolution_engine import UnifiedEvolutionEngine
from core.optimized_parallel_engine import OptimizedParallelEvolutionEngine


def create_test_genome(size: int = 2000) -> Genome:
    """创建测试基因组"""
    genes = []
    for i in range(size):
        # 创建随机基因序列
        sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=1000))
        gene = Gene(
            id=f"gene_{i:04d}",
            sequence=sequence,
            start_pos=i * 1000,
            length=1000,
            is_core=i < size * 0.8,  # 80%为核心基因
            origin='ancestral'
        )
        genes.append(gene)
    
    return Genome(genes)


def benchmark_engines(genome_size: int = 2000, generations: int = 50):
    """对比不同引擎的性能"""
    
    print(f"🧪 PARALLEL OPTIMIZATION BENCHMARK")
    print(f"   Genome size: {genome_size:,} genes")
    print(f"   Generations: {generations:,}")
    print("=" * 80)
    
    # 创建测试基因组
    test_genome = create_test_genome(genome_size)
    
    results = {}
    
    # 1. 测试串行处理（基准）
    print("\n🔄 Testing SERIAL processing...")
    serial_genome = test_genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,
        hgt_rate=0.01,
        enable_parallel=False,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    serial_start = time.time()
    serial_history = serial_engine.evolve_multiple_generations(
        serial_genome, generations, show_progress=False
    )
    serial_time = time.time() - serial_start
    
    results['serial'] = {
        'time': serial_time,
        'speed': generations / serial_time,
        'final_genes': serial_genome.gene_count,
        'mutations': serial_genome.total_mutations
    }
    
    print(f"   Time: {serial_time:.2f}s")
    print(f"   Speed: {generations/serial_time:.2f} gen/s")
    
    # 2. 测试原始并行处理
    print("\n⚡ Testing ORIGINAL PARALLEL processing...")
    original_genome = test_genome.copy()
    original_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,
        hgt_rate=0.01,
        enable_parallel=True,
        enable_gene_loss=False,
        enable_optimization=True,
        num_processes=4
    )
    
    original_start = time.time()
    original_history = original_engine.evolve_multiple_generations(
        original_genome, generations, show_progress=False
    )
    original_time = time.time() - original_start
    
    results['original_parallel'] = {
        'time': original_time,
        'speed': generations / original_time,
        'speedup': serial_time / original_time,
        'final_genes': original_genome.gene_count,
        'mutations': original_genome.total_mutations
    }
    
    print(f"   Time: {original_time:.2f}s")
    print(f"   Speed: {generations/original_time:.2f} gen/s")
    print(f"   Speedup: {serial_time/original_time:.2f}x")
    
    # 3. 测试优化并行处理
    print("\n🚀 Testing OPTIMIZED PARALLEL processing...")
    optimized_genome = test_genome.copy()
    optimized_engine = OptimizedParallelEvolutionEngine(
        mutation_rate=1e-5,
        hgt_rate=0.01,
        enable_parallel=True,
        enable_gene_loss=False,
        enable_optimization=True,
        num_processes=4,
        min_chunk_size=200,
        max_chunk_size=800
    )
    
    optimized_start = time.time()
    optimized_history = optimized_engine.evolve_multiple_generations(
        optimized_genome, generations, show_progress=False
    )
    optimized_time = time.time() - optimized_start
    
    results['optimized_parallel'] = {
        'time': optimized_time,
        'speed': generations / optimized_time,
        'speedup': serial_time / optimized_time,
        'final_genes': optimized_genome.gene_count,
        'mutations': optimized_genome.total_mutations
    }
    
    print(f"   Time: {optimized_time:.2f}s")
    print(f"   Speed: {generations/optimized_time:.2f} gen/s")
    print(f"   Speedup: {serial_time/optimized_time:.2f}x")
    
    # 清理资源
    optimized_engine.cleanup()
    
    # 4. 性能对比总结
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"{'Method':<20} {'Time(s)':<10} {'Speed(gen/s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    # 串行
    print(f"{'Serial':<20} {results['serial']['time']:<10.2f} "
          f"{results['serial']['speed']:<12.2f} {'1.00x':<10} {'100.0%':<12}")
    
    # 原始并行
    orig_efficiency = (results['original_parallel']['speedup'] / 4) * 100
    print(f"{'Original Parallel':<20} {results['original_parallel']['time']:<10.2f} "
          f"{results['original_parallel']['speed']:<12.2f} "
          f"{results['original_parallel']['speedup']:<10.2f}x "
          f"{orig_efficiency:<12.1f}%")
    
    # 优化并行
    opt_efficiency = (results['optimized_parallel']['speedup'] / 4) * 100
    print(f"{'Optimized Parallel':<20} {results['optimized_parallel']['time']:<10.2f} "
          f"{results['optimized_parallel']['speed']:<12.2f} "
          f"{results['optimized_parallel']['speedup']:<10.2f}x "
          f"{opt_efficiency:<12.1f}%")
    
    # 改进幅度
    improvement = results['optimized_parallel']['speedup'] / results['original_parallel']['speedup']
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"   Optimized vs Original: {improvement:.2f}x improvement")
    print(f"   Efficiency gain: {opt_efficiency - orig_efficiency:+.1f}%")
    
    if improvement > 1.2:
        print("   ✅ Significant performance improvement achieved!")
    elif improvement > 1.05:
        print("   ✅ Moderate performance improvement achieved!")
    else:
        print("   ⚠️  Limited improvement - may need further optimization")
    
    return results


def test_different_genome_sizes():
    """测试不同基因组大小的性能"""
    
    print(f"\n🔬 TESTING DIFFERENT GENOME SIZES")
    print("=" * 80)
    
    sizes = [500, 1000, 2000, 4000]
    generations = 20
    
    for size in sizes:
        print(f"\n📏 Testing genome size: {size:,} genes")
        print("-" * 40)
        
        try:
            results = benchmark_engines(size, generations)
            
            # 简要总结
            orig_speedup = results['original_parallel']['speedup']
            opt_speedup = results['optimized_parallel']['speedup']
            improvement = opt_speedup / orig_speedup
            
            print(f"   Original speedup: {orig_speedup:.2f}x")
            print(f"   Optimized speedup: {opt_speedup:.2f}x")
            print(f"   Improvement: {improvement:.2f}x")
            
        except Exception as e:
            print(f"   ❌ Error testing size {size}: {e}")


if __name__ == "__main__":
    print("🧬 PARALLEL EVOLUTION ENGINE OPTIMIZATION TEST")
    print("=" * 80)
    
    # 主要性能测试
    benchmark_engines(genome_size=2000, generations=50)
    
    # 不同大小测试
    test_different_genome_sizes()
    
    print(f"\n🎉 All tests completed!")