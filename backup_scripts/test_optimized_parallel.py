#!/usr/bin/env python3
"""
测试优化后的并行性能
简单快速的性能验证

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import time
import numpy as np
from core.genome import Genome, Gene
from core.unified_evolution_engine import UnifiedEvolutionEngine


def create_test_genome(size: int = 1000) -> Genome:
    """创建测试基因组"""
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


def test_performance():
    """测试性能改进"""
    
    print("🧪 PARALLEL PERFORMANCE TEST")
    print("=" * 50)
    
    # 创建测试基因组
    test_genome = create_test_genome(1000)
    generations = 5  # 少量代数快速测试
    
    print(f"Test setup: {test_genome.gene_count:,} genes, {generations} generations")
    print()
    
    # 1. 串行测试
    print("🔄 Serial processing...")
    serial_genome = test_genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.01,
        enable_parallel=False,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    serial_start = time.time()
    try:
        serial_engine.evolve_multiple_generations(serial_genome, generations, show_progress=False)
        serial_time = time.time() - serial_start
        print(f"   ✅ Time: {serial_time:.3f}s")
        print(f"   Speed: {generations/serial_time:.2f} gen/s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 2. 并行测试
    print("\n⚡ Parallel processing (optimized)...")
    parallel_genome = test_genome.copy()
    parallel_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.01,
        enable_parallel=True,
        num_processes=4,
        enable_gene_loss=False,
        enable_optimization=True
    )
    
    parallel_start = time.time()
    try:
        parallel_engine.evolve_multiple_generations(parallel_genome, generations, show_progress=False)
        parallel_time = time.time() - parallel_start
        
        # 清理资源
        if hasattr(parallel_engine, 'cleanup_parallel_resources'):
            parallel_engine.cleanup_parallel_resources()
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        efficiency = (speedup / 4) * 100
        
        print(f"   ✅ Time: {parallel_time:.3f}s")
        print(f"   Speed: {generations/parallel_time:.2f} gen/s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 3. 结果分析
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    if speedup > 2.0:
        print("   ✅ 优秀的并行性能！")
        print("   🎯 并行优化成功，CPU资源得到有效利用")
    elif speedup > 1.5:
        print("   ✅ 良好的并行性能")
        print("   🎯 并行优化有效，显著提升了处理速度")
    elif speedup > 1.2:
        print("   ⚠️  中等的并行性能")
        print("   💡 还有进一步优化的空间")
    else:
        print("   ❌ 并行性能有限")
        print("   🔧 需要进一步分析和优化")
    
    # 4. 问题诊断
    print(f"\n🔍 DIAGNOSIS:")
    if efficiency < 50:
        print("   ⚠️  并行效率较低，可能的原因：")
        print("      - 任务分块仍然过小")
        print("      - 进程间通信开销较大")
        print("      - 系统资源竞争")
    elif efficiency < 75:
        print("   ✅ 并行效率中等，已有明显改善")
    else:
        print("   ✅ 并行效率良好，优化效果显著")
    
    return {
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'efficiency': efficiency
    }


if __name__ == "__main__":
    try:
        results = test_performance()
        
        if results:
            print(f"\n🎉 Test completed successfully!")
            print(f"Final speedup: {results['speedup']:.2f}x")
        else:
            print(f"\n❌ Test failed")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()