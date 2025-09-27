#!/usr/bin/env python3
"""
快速并行测试 - 诊断并行性能问题
"""

import time
import numpy as np
from core.genome import Genome, Gene
from core.unified_evolution_engine import UnifiedEvolutionEngine


def create_small_test_genome(size: int = 500) -> Genome:
    """创建小型测试基因组"""
    genes = []
    for i in range(size):
        sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=500))
        gene = Gene(
            id=f"gene_{i:04d}",
            sequence=sequence,
            start_pos=i * 500,
            length=500,
            is_core=True,
            origin='ancestral'
        )
        genes.append(gene)
    
    return Genome(genes)


def quick_performance_test():
    """快速性能测试"""
    
    print("🔬 QUICK PARALLEL PERFORMANCE DIAGNOSIS")
    print("=" * 60)
    
    # 创建测试基因组
    test_genome = create_small_test_genome(1000)  # 1000个基因
    generations = 10
    
    print(f"Test setup: {test_genome.gene_count:,} genes, {generations} generations")
    print()
    
    # 1. 串行测试
    print("🔄 Serial processing...")
    serial_genome = test_genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.005,
        enable_parallel=False,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    serial_start = time.time()
    serial_engine.evolve_multiple_generations(serial_genome, generations, show_progress=False)
    serial_time = time.time() - serial_start
    
    print(f"   Time: {serial_time:.3f}s")
    print(f"   Speed: {generations/serial_time:.2f} gen/s")
    
    # 2. 并行测试 - 2进程
    print("\n⚡ Parallel processing (2 processes)...")
    parallel_genome_2 = test_genome.copy()
    parallel_engine_2 = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.005,
        enable_parallel=True,
        num_processes=2,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    parallel_start_2 = time.time()
    parallel_engine_2.evolve_multiple_generations(parallel_genome_2, generations, show_progress=False)
    parallel_time_2 = time.time() - parallel_start_2
    
    speedup_2 = serial_time / parallel_time_2 if parallel_time_2 > 0 else 0
    efficiency_2 = (speedup_2 / 2) * 100
    
    print(f"   Time: {parallel_time_2:.3f}s")
    print(f"   Speed: {generations/parallel_time_2:.2f} gen/s")
    print(f"   Speedup: {speedup_2:.2f}x")
    print(f"   Efficiency: {efficiency_2:.1f}%")
    
    # 3. 并行测试 - 4进程
    print("\n⚡ Parallel processing (4 processes)...")
    parallel_genome_4 = test_genome.copy()
    parallel_engine_4 = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.005,
        enable_parallel=True,
        num_processes=4,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    parallel_start_4 = time.time()
    parallel_engine_4.evolve_multiple_generations(parallel_genome_4, generations, show_progress=False)
    parallel_time_4 = time.time() - parallel_start_4
    
    speedup_4 = serial_time / parallel_time_4 if parallel_time_4 > 0 else 0
    efficiency_4 = (speedup_4 / 4) * 100
    
    print(f"   Time: {parallel_time_4:.3f}s")
    print(f"   Speed: {generations/parallel_time_4:.2f} gen/s")
    print(f"   Speedup: {speedup_4:.2f}x")
    print(f"   Efficiency: {efficiency_4:.1f}%")
    
    # 4. 分析结果
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"{'Processes':<12} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 60)
    print(f"{'1 (Serial)':<12} {serial_time:<10.3f} {'1.00x':<10} {'100.0%':<12}")
    print(f"{'2 (Parallel)':<12} {parallel_time_2:<10.3f} {speedup_2:<10.2f}x {efficiency_2:<12.1f}%")
    print(f"{'4 (Parallel)':<12} {parallel_time_4:<10.3f} {speedup_4:<10.2f}x {efficiency_4:<12.1f}%")
    
    # 5. 诊断问题
    print(f"\n🔍 DIAGNOSIS:")
    
    if speedup_2 < 1.2:
        print("   ❌ 2进程并行几乎没有加速 - 存在严重的并行开销")
        print("      可能原因：")
        print("      - 进程创建/销毁开销过大")
        print("      - 数据序列化/反序列化开销")
        print("      - 任务分块过小，通信开销超过计算收益")
        print("      - GIL或其他同步问题")
    elif speedup_2 < 1.5:
        print("   ⚠️  2进程并行加速有限 - 存在明显的并行开销")
    else:
        print("   ✅ 2进程并行效果良好")
    
    if speedup_4 < speedup_2:
        print("   ❌ 4进程性能反而下降 - 并行开销随进程数增加")
        print("      建议：减少进程数或优化分块策略")
    elif speedup_4 < 2.0:
        print("   ⚠️  4进程加速不理想 - 需要优化并行策略")
    else:
        print("   ✅ 4进程并行效果良好")
    
    # 6. 建议
    print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
    if efficiency_2 < 50 or efficiency_4 < 50:
        print("   1. 增加每个分块的大小，减少进程间通信")
        print("   2. 重用进程池，避免重复创建进程")
        print("   3. 预初始化工作进程中的对象")
        print("   4. 考虑使用线程池而非进程池（如果GIL不是瓶颈）")
        print("   5. 批量处理多个代数以摊销并行开销")
    
    return {
        'serial_time': serial_time,
        'parallel_time_2': parallel_time_2,
        'parallel_time_4': parallel_time_4,
        'speedup_2': speedup_2,
        'speedup_4': speedup_4,
        'efficiency_2': efficiency_2,
        'efficiency_4': efficiency_4
    }


if __name__ == "__main__":
    results = quick_performance_test()