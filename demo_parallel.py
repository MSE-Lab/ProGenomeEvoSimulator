#!/usr/bin/env python3
"""
并行化进化引擎演示脚本
展示如何使用并行化功能提升计算效率
"""

import time
import numpy as np
import multiprocessing as mp
from core.genome import create_initial_genome
from core.parallel_evolution_engine import ParallelEvolutionEngine
from core.evolution_engine_optimized import OptimizedEvolutionEngine


def demo_basic_parallel():
    """基础并行化演示"""
    
    print("🚀 Basic Parallel Evolution Demo")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(42)
    genome = create_initial_genome(
        gene_count=1000,
        avg_gene_length=800,
        min_gene_length=200
    )
    
    print(f"\n📊 Demo genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    print(f"🖥️  Available CPU cores: {mp.cpu_count()}")
    
    # 创建并行进化引擎
    parallel_engine = ParallelEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.01,
        recombination_rate=1e-3,
        num_processes=None,  # 使用所有可用核心
        chunk_size=None,     # 自动计算最优分块大小
        enable_progress_sharing=True
    )
    
    print(f"\n⚡ Starting parallel evolution with {parallel_engine.num_processes} processes...")
    
    # 运行并行进化
    start_time = time.time()
    evolved_genome, snapshots = parallel_engine.simulate_evolution_parallel(
        initial_genome=genome,
        generations=50,
        save_snapshots=True,
        snapshot_interval=10
    )
    total_time = time.time() - start_time
    
    # 显示结果
    print(f"\n📈 Evolution Results:")
    print(f"   Initial: {genome.gene_count:,} genes, {genome.size:,} bp")
    print(f"   Final: {evolved_genome.gene_count:,} genes, {evolved_genome.size:,} bp")
    print(f"   Changes: {evolved_genome.gene_count - genome.gene_count:+,} genes, "
          f"{evolved_genome.size - genome.size:+,} bp")
    print(f"   Total mutations: {evolved_genome.total_mutations:,}")
    print(f"   Total HGT events: {evolved_genome.total_hgt_events:,}")
    print(f"   Total recombinations: {evolved_genome.total_recombination_events:,}")
    
    # 性能分析
    performance = parallel_engine.get_parallel_performance_analysis()
    print(f"\n⚡ Performance Analysis:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Evolution speed: {50/total_time:.2f} generations/second")
    print(f"   Parallel efficiency: {performance['avg_parallel_efficiency']:.1f}%")
    print(f"   Actual speedup: {performance['actual_speedup']:.2f}x")
    
    return evolved_genome, performance


def demo_performance_comparison():
    """性能对比演示"""
    
    print("\n🏁 Performance Comparison Demo")
    print("=" * 50)
    
    # 创建相同的测试基因组
    np.random.seed(123)
    test_genome = create_initial_genome(
        gene_count=800,
        avg_gene_length=600,
        min_gene_length=150
    )
    
    generations = 30
    
    print(f"\n📊 Comparison setup:")
    print(f"   Test genome: {test_genome.gene_count:,} genes")
    print(f"   Generations: {generations}")
    print(f"   CPU cores: {mp.cpu_count()}")
    
    # 1. 串行版本
    print(f"\n🔄 Running serial evolution...")
    serial_genome = test_genome.copy()
    serial_engine = OptimizedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.01,
        recombination_rate=1e-3
    )
    
    serial_start = time.time()
    serial_engine.evolve_multiple_generations(
        serial_genome, generations, show_progress=False
    )
    serial_time = time.time() - serial_start
    
    print(f"   ✓ Serial completed: {serial_time:.2f}s ({generations/serial_time:.2f} gen/s)")
    
    # 2. 并行版本
    print(f"\n⚡ Running parallel evolution...")
    parallel_genome = test_genome.copy()
    parallel_engine = ParallelEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.01,
        recombination_rate=1e-3,
        num_processes=None,
        enable_progress_sharing=False  # 关闭进度共享以获得最佳性能
    )
    
    parallel_start = time.time()
    parallel_engine.evolve_multiple_generations_parallel(
        parallel_genome, generations, show_progress=False
    )
    parallel_time = time.time() - parallel_start
    
    print(f"   ✓ Parallel completed: {parallel_time:.2f}s ({generations/parallel_time:.2f} gen/s)")
    
    # 3. 性能对比
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / mp.cpu_count() * 100
    time_saved = serial_time - parallel_time
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Efficiency: {efficiency:.1f}%")
    print(f"   Time saved: {time_saved:.2f}s ({time_saved/serial_time*100:.1f}%)")
    
    # 4. 结果一致性
    mutation_diff = abs(serial_genome.total_mutations - parallel_genome.total_mutations)
    gene_diff = abs(serial_genome.gene_count - parallel_genome.gene_count)
    
    print(f"\n✅ Result Consistency:")
    print(f"   Mutation difference: {mutation_diff}")
    print(f"   Gene count difference: {gene_diff}")
    
    if mutation_diff < 100 and gene_diff < 10:
        print("   ✓ Results are consistent between serial and parallel!")
    else:
        print("   ⚠️  Some differences detected (expected due to randomness)")
    
    return {
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'efficiency': efficiency
    }


def demo_scalability_test():
    """可扩展性测试演示"""
    
    print("\n📈 Scalability Test Demo")
    print("=" * 50)
    
    # 测试不同基因组大小的性能
    genome_sizes = [200, 500, 1000, 2000]
    generations = 20
    
    results = []
    
    for size in genome_sizes:
        print(f"\n🧬 Testing genome size: {size:,} genes")
        
        # 创建测试基因组
        np.random.seed(42)
        test_genome = create_initial_genome(
            gene_count=size,
            avg_gene_length=500,
            min_gene_length=100
        )
        
        # 并行进化测试
        parallel_engine = ParallelEvolutionEngine(
            mutation_rate=1e-4,
            hgt_rate=0.005,
            recombination_rate=5e-4,
            num_processes=None,
            enable_progress_sharing=False
        )
        
        start_time = time.time()
        parallel_engine.evolve_multiple_generations_parallel(
            test_genome, generations, show_progress=False
        )
        elapsed_time = time.time() - start_time
        
        speed = generations / elapsed_time
        throughput = size * generations / elapsed_time  # genes * generations per second
        
        results.append({
            'genome_size': size,
            'time': elapsed_time,
            'speed': speed,
            'throughput': throughput
        })
        
        print(f"   Time: {elapsed_time:.2f}s, Speed: {speed:.2f} gen/s")
        print(f"   Throughput: {throughput:.0f} gene-generations/s")
    
    # 分析可扩展性
    print(f"\n📊 Scalability Analysis:")
    for i, result in enumerate(results):
        if i == 0:
            baseline_throughput = result['throughput']
        
        relative_throughput = result['throughput'] / baseline_throughput
        print(f"   {result['genome_size']:,} genes: {relative_throughput:.2f}x baseline throughput")
    
    return results


def main():
    """主演示函数"""
    
    print("🧪 Parallel Evolution Engine Demo Suite")
    print("=" * 60)
    print(f"🖥️  System: {mp.cpu_count()} CPU cores available")
    print()
    
    try:
        # 1. 基础并行化演示
        evolved_genome, performance = demo_basic_parallel()
        
        # 2. 性能对比演示
        comparison_results = demo_performance_comparison()
        
        # 3. 可扩展性测试演示
        scalability_results = demo_scalability_test()
        
        # 总结
        print("\n" + "=" * 60)
        print("🎉 Demo Suite Completed!")
        print("=" * 60)
        
        print(f"\n📋 Key Findings:")
        print(f"   Best speedup achieved: {comparison_results['speedup']:.2f}x")
        print(f"   Parallel efficiency: {comparison_results['efficiency']:.1f}%")
        print(f"   Recommended for genomes: >500 genes")
        
        if comparison_results['speedup'] > 1.5:
            print(f"   ✅ Parallel processing provides significant benefits!")
        else:
            print(f"   ⚠️  Limited speedup - consider larger genomes or more generations")
        
        print(f"\n💡 Usage Tips:")
        print(f"   - Use all CPU cores for best performance")
        print(f"   - Disable progress sharing for maximum speed")
        print(f"   - Larger genomes benefit more from parallelization")
        print(f"   - Optimal for multi-generation simulations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()