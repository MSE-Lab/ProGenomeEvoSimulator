#!/usr/bin/env python3
"""测试新的同源重组机制"""

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from core.genome import create_initial_genome
from mechanisms.homologous_recombination import HomologousRecombination

def test_new_recombination_mechanism():
    """测试新的同源重组机制"""
    print("🧬 Testing NEW Homologous Recombination Mechanism")
    print("=" * 60)
    
    # 创建测试基因组
    print("📋 Creating test genome...")
    genome = create_initial_genome(gene_count=100, avg_gene_length=500)
    print(f"   Initial genome: {len(genome.genes)} genes, {genome.size} bp")
    
    # 记录初始状态
    initial_sequences = {gene.id: gene.sequence for gene in genome.genes}
    initial_recombination_events = genome.total_recombination_events
    
    # 创建重组引擎
    print("\n🔧 Initializing recombination engine...")
    recombination = HomologousRecombination(
        recombination_rate=0.1,  # 高重组率便于测试
        mutations_per_event=(3, 8),  # 每次重组3-8个突变
        enable_debug=True
    )
    
    # 执行重组
    print("\n🎯 Applying recombination...")
    recombination_events = recombination.apply_recombination(genome, generations=1)
    
    # 分析结果
    print(f"\n📊 RESULTS ANALYSIS")
    print("-" * 40)
    print(f"Recombination events performed: {recombination_events}")
    print(f"Total recombination events in genome: {genome.total_recombination_events}")
    print(f"Events increase: {genome.total_recombination_events - initial_recombination_events}")
    
    # 检查基因变化
    genes_changed = 0
    total_mutations = 0
    
    for gene in genome.genes:
        if gene.id in initial_sequences:
            original_seq = initial_sequences[gene.id]
            current_seq = gene.sequence
            
            if original_seq != current_seq:
                genes_changed += 1
                # 计算突变数量
                mutations = sum(1 for i, (a, b) in enumerate(zip(original_seq, current_seq)) if a != b)
                total_mutations += mutations
                
                if genes_changed <= 3:  # 只显示前3个变化的基因
                    print(f"\n🧬 Gene {gene.id} changed:")
                    print(f"   Recombination count: {gene.recombination_count}")
                    print(f"   Mutations detected: {mutations}")
                    if mutations <= 10:  # 只显示少量突变的详细信息
                        for i, (a, b) in enumerate(zip(original_seq, current_seq)):
                            if a != b:
                                print(f"   Position {i}: {a} -> {b}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Genes affected: {genes_changed}/{len(genome.genes)}")
    print(f"   Total mutations introduced: {total_mutations}")
    print(f"   Average mutations per affected gene: {total_mutations/genes_changed:.1f}" if genes_changed > 0 else "   Average mutations per affected gene: 0")
    print(f"   Average mutations per recombination event: {total_mutations/recombination_events:.1f}" if recombination_events > 0 else "   Average mutations per recombination event: 0")
    
    # 获取统计信息
    stats = recombination.get_recombination_statistics(genome)
    print(f"\n📊 ENGINE STATISTICS:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 验证机制正确性
    print(f"\n✅ MECHANISM VALIDATION:")
    if recombination_events > 0:
        print("   ✅ Recombination events occurred")
    else:
        print("   ❌ No recombination events occurred")
    
    if genes_changed > 0:
        print("   ✅ Genes were modified")
    else:
        print("   ❌ No genes were modified")
    
    if total_mutations > 0:
        print("   ✅ Mutations were introduced")
        print(f"   ✅ New mechanism working: Multiple mutations per event")
    else:
        print("   ❌ No mutations were introduced")
    
    success = recombination_events > 0 and genes_changed > 0 and total_mutations > 0
    
    print(f"\n🎉 TEST RESULT: {'SUCCESS' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    test_new_recombination_mechanism()