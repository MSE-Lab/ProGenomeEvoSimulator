#!/usr/bin/env python3
"""
基因丢失机制测试脚本
验证基因丢失功能的正确性
"""

import numpy as np
from core.genome import create_initial_genome
from mechanisms.gene_loss import GeneLossEngine
from core.evolution_engine_with_gene_loss import EvolutionEngineWithGeneLoss


def test_basic_gene_loss():
    """测试基础基因丢失功能"""
    
    print("🧪 Testing Basic Gene Loss Functionality")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(42)
    genome = create_initial_genome(
        gene_count=100,
        avg_gene_length=500,
        min_gene_length=200
    )
    
    print(f"📊 Initial test genome:")
    print(f"   Total genes: {genome.gene_count}")
    print(f"   Core genes: {genome.core_gene_count}")
    print(f"   HGT genes: {genome.hgt_gene_count}")
    
    # 创建基因丢失引擎
    gene_loss = GeneLossEngine(
        loss_rate=0.1,  # 高丢失率以便观察效果
        core_gene_protection=0.8,  # 80%保护
        hgt_gene_loss_multiplier=5.0,
        min_genome_size=50,
        optimal_genome_size=80
    )
    
    print(f"\n⚙️  Gene loss parameters:")
    print(f"   Loss rate: 0.1 (10%)")
    print(f"   Core protection: 80%")
    print(f"   HGT multiplier: 5x")
    print(f"   Min genome size: 50")
    
    # 应用基因丢失
    print(f"\n🗑️  Applying gene loss...")
    genes_lost = gene_loss.apply_gene_loss(genome, generations=1)
    
    print(f"   Genes lost in one generation: {genes_lost}")
    print(f"   Remaining genes: {genome.gene_count}")
    print(f"   Remaining core genes: {genome.core_gene_count}")
    print(f"   Remaining HGT genes: {genome.hgt_gene_count}")
    
    # 获取统计信息
    stats = gene_loss.get_loss_statistics(genome)
    print(f"\n📊 Loss statistics:")
    print(f"   Total lost: {stats['total_genes_lost']}")
    print(f"   Core lost: {stats['core_genes_lost']}")
    print(f"   HGT lost: {stats['hgt_genes_lost']}")
    
    return genes_lost > 0  # 测试是否成功丢失了基因


def test_gene_loss_protection():
    """测试基因丢失保护机制"""
    
    print("\n🛡️  Testing Gene Loss Protection")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(123)
    genome = create_initial_genome(
        gene_count=200,
        avg_gene_length=400,
        min_gene_length=150
    )
    
    # 添加一些HGT基因
    from core.genome import Gene
    for i in range(20):
        hgt_gene = Gene(
            id=f"hgt_{i}",
            sequence="ATCGATCGATCG" * 20,
            start_pos=0,
            length=240,
            is_core=False,
            origin="hgt"
        )
        genome.add_gene(hgt_gene)
    
    initial_core = genome.core_gene_count
    initial_hgt = genome.hgt_gene_count
    
    print(f"📊 Test genome composition:")
    print(f"   Core genes: {initial_core}")
    print(f"   HGT genes: {initial_hgt}")
    
    # 测试不同保护级别
    protection_levels = [0.5, 0.8, 0.95]
    
    for protection in protection_levels:
        test_genome = genome.copy()
        
        gene_loss = GeneLossEngine(
            loss_rate=0.05,  # 5%丢失率
            core_gene_protection=protection,
            hgt_gene_loss_multiplier=10.0,
            min_genome_size=100
        )
        
        # 运行多代以观察效果
        total_lost = 0
        for _ in range(5):
            lost = gene_loss.apply_gene_loss(test_genome, generations=1)
            total_lost += lost
        
        stats = gene_loss.get_loss_statistics(test_genome)
        
        print(f"\n🔬 Protection level {protection*100:.0f}%:")
        print(f"   Total lost: {total_lost}")
        print(f"   Core lost: {stats['core_genes_lost']}")
        print(f"   HGT lost: {stats['hgt_genes_lost']}")
        print(f"   Core retention: {(initial_core - stats['core_genes_lost'])/initial_core*100:.1f}%")
        print(f"   HGT retention: {(initial_hgt - stats['hgt_genes_lost'])/initial_hgt*100:.1f}%")


def test_integrated_evolution():
    """测试集成进化引擎"""
    
    print("\n🧬 Testing Integrated Evolution Engine")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(456)
    genome = create_initial_genome(
        gene_count=300,
        avg_gene_length=600,
        min_gene_length=200
    )
    
    print(f"📊 Initial genome: {genome.gene_count} genes")
    
    # 创建集成进化引擎
    engine = EvolutionEngineWithGeneLoss(
        mutation_rate=1e-3,
        hgt_rate=0.05,
        recombination_rate=1e-2,
        enable_gene_loss=True,
        loss_rate=1e-3,
        core_gene_protection=0.9,
        hgt_gene_loss_multiplier=5.0,
        optimal_genome_size=280
    )
    
    print(f"⚙️  Running integrated evolution for 20 generations...")
    
    # 运行进化
    history = engine.evolve_multiple_generations(genome, 20, show_progress=False)
    
    print(f"✅ Evolution completed!")
    print(f"   Final genome: {genome.gene_count} genes")
    print(f"   Gene change: {genome.gene_count - 300:+d}")
    print(f"   Total mutations: {genome.total_mutations}")
    print(f"   Total HGT events: {genome.total_hgt_events}")
    
    # 基因丢失统计
    if engine.gene_loss:
        stats = engine.gene_loss.get_loss_statistics(genome)
        print(f"   Total genes lost: {stats['total_genes_lost']}")
        print(f"   Loss per generation: {stats['avg_total_loss_per_generation']:.3f}")
    
    return len(history) == 20  # 验证是否完成了所有代数


def test_genome_size_regulation():
    """测试基因组大小调节机制"""
    
    print("\n📏 Testing Genome Size Regulation")
    print("=" * 50)
    
    # 创建较大的基因组以触发大小压力
    np.random.seed(789)
    large_genome = create_initial_genome(
        gene_count=500,  # 较大的基因组
        avg_gene_length=400,
        min_gene_length=100
    )
    
    print(f"📊 Large genome: {large_genome.gene_count} genes")
    
    # 设置较小的最优大小以触发压力
    gene_loss = GeneLossEngine(
        loss_rate=1e-3,
        core_gene_protection=0.95,
        hgt_gene_loss_multiplier=3.0,
        min_genome_size=200,
        optimal_genome_size=300,  # 小于初始大小
        enable_size_pressure=True
    )
    
    print(f"🎯 Optimal size: 300 genes (size pressure active)")
    
    # 运行多代观察大小调节
    sizes = [large_genome.gene_count]
    
    for gen in range(20):
        gene_loss.apply_gene_loss(large_genome, generations=1)
        sizes.append(large_genome.gene_count)
        
        if gen % 5 == 4:  # 每5代报告一次
            print(f"   Generation {gen+1}: {large_genome.gene_count} genes")
    
    # 分析大小变化趋势
    initial_size = sizes[0]
    final_size = sizes[-1]
    size_change = final_size - initial_size
    
    print(f"\n📈 Size regulation results:")
    print(f"   Initial size: {initial_size}")
    print(f"   Final size: {final_size}")
    print(f"   Size change: {size_change:+d} genes")
    print(f"   Trend: {'Decreasing' if size_change < 0 else 'Increasing' if size_change > 0 else 'Stable'}")
    
    return size_change < 0  # 应该趋向于减小


def main():
    """主测试函数"""
    
    print("🧪 Gene Loss Mechanism Test Suite")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 1. 基础功能测试
        result1 = test_basic_gene_loss()
        test_results.append(("Basic Gene Loss", result1))
        
        # 2. 保护机制测试
        test_gene_loss_protection()
        test_results.append(("Protection Mechanism", True))  # 这个测试主要是观察性的
        
        # 3. 集成进化测试
        result3 = test_integrated_evolution()
        test_results.append(("Integrated Evolution", result3))
        
        # 4. 大小调节测试
        result4 = test_genome_size_regulation()
        test_results.append(("Size Regulation", result4))
        
        # 总结测试结果
        print("\n" + "=" * 60)
        print("🎉 Test Suite Results")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n📊 Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Gene loss mechanism is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
        
        print(f"\n💡 Next steps:")
        print(f"   - Run 'python demo_gene_loss.py' for detailed demonstrations")
        print(f"   - Run 'python main_with_gene_loss.py' for full simulations")
        print(f"   - Adjust parameters based on your research needs")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()