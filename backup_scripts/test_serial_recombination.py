#!/usr/bin/env python3
"""测试串行模式下的重组功能"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.genome import create_initial_genome
from core.unified_evolution_engine import UnifiedEvolutionEngine

def test_serial_recombination():
    print("🧪 Testing Serial Mode Recombination")
    print("=" * 60)
    
    # 创建小型测试基因组
    genome = create_initial_genome(gene_count=50, avg_gene_length=300)
    print(f"Created test genome: {genome.gene_count} genes, {genome.size} bp")
    
    # 创建串行进化引擎
    engine = UnifiedEvolutionEngine(
        recombination_rate=0.2,  # 高重组率
        min_similarity_for_recombination=0.3,  # 低相似性阈值
        enable_parallel=False,  # 强制串行模式
        enable_gene_loss=False  # 关闭基因丢失简化测试
    )
    
    print(f"\nInitial recombination events: {genome.total_recombination_events}")
    
    # 进化5代
    print("\nEvolving 5 generations...")
    for gen in range(5):
        stats = engine.evolve_one_generation(genome)
        recomb_events = stats.get('recombination_events', 0)
        print(f"Generation {gen+1}: {recomb_events} recombination events")
    
    print(f"\nFinal recombination events: {genome.total_recombination_events}")
    
    # 获取重组统计
    recomb_stats = engine.recombination.get_recombination_statistics(genome)
    print(f"\nRecombination Statistics:")
    for key, value in recomb_stats.items():
        print(f"  {key}: {value}")
    
    success = genome.total_recombination_events > 0
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: {genome.total_recombination_events} total recombination events")
    
    return success

if __name__ == "__main__":
    success = test_serial_recombination()
    sys.exit(0 if success else 1)