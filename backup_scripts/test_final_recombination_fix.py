#!/usr/bin/env python3
"""最终重组修复验证测试"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.genome import create_initial_genome
from core.unified_evolution_engine import UnifiedEvolutionEngine

def test_final_recombination_fix():
    print("🧪 FINAL RECOMBINATION FIX TEST")
    print("=" * 60)
    
    # 创建测试基因组
    genome = create_initial_genome(gene_count=100, avg_gene_length=400)
    print(f"Created genome: {genome.gene_count} genes, {genome.size} bp")
    
    # 测试串行模式
    print("\n1️⃣ Testing SERIAL mode...")
    serial_engine = UnifiedEvolutionEngine(
        recombination_rate=0.1,
        min_similarity_for_recombination=0.3,
        enable_parallel=False
    )
    
    genome_serial = genome.copy()
    initial_recomb = genome_serial.total_recombination_events
    
    for i in range(3):
        stats = serial_engine.evolve_one_generation(genome_serial)
        recomb_events = stats.get('recombination_events', 0)
        print(f"  Gen {i+1}: {recomb_events} recombination events")
    
    serial_total = genome_serial.total_recombination_events - initial_recomb
    print(f"  Serial total: {serial_total} recombination events")
    
    # 测试并行模式
    print("\n2️⃣ Testing PARALLEL mode...")
    parallel_engine = UnifiedEvolutionEngine(
        recombination_rate=0.1,
        min_similarity_for_recombination=0.3,
        enable_parallel=True,
        parallel_threshold=50  # 确保启用并行
    )
    
    genome_parallel = genome.copy()
    initial_recomb = genome_parallel.total_recombination_events
    
    for i in range(3):
        stats = parallel_engine.evolve_one_generation(genome_parallel)
        recomb_events = stats.get('recombination_events', 0)
        total_recomb_events = stats.get('total_recombination_events', 0)
        print(f"  Gen {i+1}: {recomb_events} recombination events (total: {total_recomb_events})")
    
    parallel_total = genome_parallel.total_recombination_events - initial_recomb
    print(f"  Parallel total: {parallel_total} recombination events")
    
    # 结果评估
    print(f"\n📊 RESULTS:")
    print(f"  Serial mode: {serial_total} events")
    print(f"  Parallel mode: {parallel_total} events")
    
    serial_success = serial_total > 0
    parallel_success = parallel_total > 0
    
    print(f"\n{'✅' if serial_success else '❌'} Serial mode: {'SUCCESS' if serial_success else 'FAILED'}")
    print(f"{'✅' if parallel_success else '❌'} Parallel mode: {'SUCCESS' if parallel_success else 'FAILED'}")
    
    overall_success = serial_success and parallel_success
    print(f"\n{'🎉 OVERALL SUCCESS!' if overall_success else '❌ OVERALL FAILED'}")
    
    # 清理资源
    serial_engine.cleanup_parallel_resources()
    parallel_engine.cleanup_parallel_resources()
    
    return overall_success

if __name__ == "__main__":
    success = test_final_recombination_fix()
    sys.exit(0 if success else 1)