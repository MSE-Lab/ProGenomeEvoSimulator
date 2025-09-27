#!/usr/bin/env python3
"""
测试索引越界修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_parallel_processing():
    """测试并行处理是否修复了索引越界问题"""
    print("🧪 测试并行处理索引越界修复")
    print("=" * 50)
    
    try:
        from core.genome import create_initial_genome
        from core.unified_evolution_engine import UnifiedEvolutionEngine
        
        # 创建中等大小的基因组
        print("1. 创建测试基因组 (500基因)...")
        genome = create_initial_genome(
            gene_count=500, 
            avg_gene_length=600, 
            use_biological_sequences=True
        )
        print(f"✅ 基因组: {genome.gene_count} 基因, {genome.size:,} bp")
        
        # 创建进化引擎
        print("\n2. 创建进化引擎 (并行模式)...")
        engine = UnifiedEvolutionEngine(
            mutation_rate=5e-3,  # 较高的突变率来测试
            hgt_rate=0.02,
            enable_parallel=True,
            enable_gene_loss=False,
            num_processes=2  # 使用较少进程避免过度并行
        )
        print("✅ 进化引擎创建成功")
        
        # 运行进化
        print("\n3. 运行并行进化 (2代)...")
        final_genome, snapshots = engine.simulate_evolution(
            initial_genome=genome,
            generations=2,
            save_snapshots=False
        )
        
        print(f"\n🎉 测试成功完成!")
        print(f"   初始基因组: {genome.gene_count} 基因")
        print(f"   最终基因组: {final_genome.gene_count} 基因")
        print(f"   总突变数: {final_genome.total_mutations:,}")
        print(f"   HGT事件: {final_genome.total_hgt_events}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hotspot_positions():
    """测试热点位置计算"""
    print("\n🔥 测试热点位置计算")
    print("=" * 30)
    
    try:
        from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
        from core.genome import Gene
        
        # 创建测试基因
        test_sequence = "ATGCGATCGATCGATCGATAG"  # 21 bp
        gene = Gene(
            id="test_gene",
            sequence=test_sequence,
            start_pos=0,
            length=len(test_sequence)
        )
        
        # 创建突变引擎
        engine = OptimizedPointMutationEngine(
            mutation_rate=1e-3,
            enable_hotspots=True
        )
        
        # 测试热点位置
        hotspots = engine._find_hotspot_positions_cached(gene)
        print(f"✅ 基因长度: {gene.length}")
        print(f"✅ 序列长度: {len(gene.sequence)}")
        print(f"✅ 热点位置: {sorted(hotspots) if hotspots else '无'}")
        
        # 测试突变率计算
        rates = engine._calculate_batch_mutation_rates(gene)
        print(f"✅ 突变率数组长度: {len(rates)}")
        print(f"✅ 最大索引: {len(rates) - 1}")
        
        return True
        
    except Exception as e:
        print(f"❌ 热点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧬 索引越界修复验证测试")
    print("=" * 60)
    
    # 测试热点位置计算
    hotspot_ok = test_hotspot_positions()
    
    # 测试并行处理
    parallel_ok = test_parallel_processing()
    
    print("\n" + "=" * 60)
    if hotspot_ok and parallel_ok:
        print("🎉 所有测试通过! 索引越界问题已修复!")
    else:
        print("❌ 部分测试失败，需要进一步调试")
    
    return hotspot_ok and parallel_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)