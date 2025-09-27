#!/usr/bin/env python3
"""
基因丢失机制演示脚本
展示基因丢失对基因组进化的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from core.genome import create_initial_genome
from core.evolution_engine_with_gene_loss import EvolutionEngineWithGeneLoss
from core.evolution_engine_optimized import OptimizedEvolutionEngine


def demo_basic_gene_loss():
    """基础基因丢失演示"""
    
    print("🗑️  Basic Gene Loss Mechanism Demo")
    print("=" * 60)
    
    # 创建测试基因组
    np.random.seed(42)
    genome = create_initial_genome(
        gene_count=2000,
        avg_gene_length=800,
        min_gene_length=200
    )
    
    print(f"\n📊 Initial genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    print(f"   Core genes: {genome.core_gene_count:,}")
    print(f"   HGT genes: {genome.hgt_gene_count:,}")
    
    # 创建带基因丢失的进化引擎
    engine_with_loss = EvolutionEngineWithGeneLoss(
        mutation_rate=1e-4,
        hgt_rate=0.02,
        recombination_rate=1e-3,
        # 基因丢失参数
        enable_gene_loss=True,
        loss_rate=1e-4,  # 较高的丢失率以便观察效果
        core_gene_protection=0.9,  # 90%的核心基因保护
        hgt_gene_loss_multiplier=5.0,  # HGT基因丢失率是核心基因的5倍
        min_genome_size=1000,
        optimal_genome_size=1800  # 设置较小的最优大小以触发大小压力
    )
    
    print(f"\n⚙️  Evolution parameters:")
    print(f"   Gene loss rate: 1e-4")
    print(f"   Core gene protection: 90%")
    print(f"   HGT loss multiplier: 5x")
    print(f"   Optimal genome size: 1800 genes")
    
    # 运行进化模拟
    print(f"\n🧬 Starting evolution with gene loss...")
    evolved_genome, snapshots = engine_with_loss.simulate_evolution(
        initial_genome=genome,
        generations=100,
        save_snapshots=True,
        snapshot_interval=20
    )
    
    # 显示结果对比
    print(f"\n📈 Evolution Results:")
    print(f"   Initial: {genome.gene_count:,} genes ({genome.core_gene_count:,} core, {genome.hgt_gene_count:,} HGT)")
    print(f"   Final: {evolved_genome.gene_count:,} genes ({evolved_genome.core_gene_count:,} core, {evolved_genome.hgt_gene_count:,} HGT)")
    print(f"   Net change: {evolved_genome.gene_count - genome.gene_count:+,} genes")
    print(f"   Size change: {evolved_genome.size - genome.size:+,} bp")
    
    # 基因丢失详细分析
    loss_stats = engine_with_loss.gene_loss.get_loss_statistics(evolved_genome)
    print(f"\n🗑️  Gene Loss Analysis:")
    print(f"   Total genes lost: {loss_stats['total_genes_lost']}")
    print(f"   Core genes lost: {loss_stats['core_genes_lost']} ({loss_stats['core_loss_percentage']:.1f}%)")
    print(f"   HGT genes lost: {loss_stats['hgt_genes_lost']} ({loss_stats['hgt_loss_percentage']:.1f}%)")
    print(f"   Loss rate: {loss_stats['avg_total_loss_per_generation']:.3f} genes/generation")
    
    return evolved_genome, snapshots, engine_with_loss


def demo_gene_loss_comparison():
    """基因丢失机制对比演示"""
    
    print("\n🔬 Gene Loss vs No Gene Loss Comparison")
    print("=" * 60)
    
    # 创建相同的初始基因组
    np.random.seed(123)
    initial_genome = create_initial_genome(
        gene_count=1500,
        avg_gene_length=600,
        min_gene_length=150
    )
    
    generations = 80
    
    print(f"\n📊 Comparison setup:")
    print(f"   Initial genome: {initial_genome.gene_count:,} genes")
    print(f"   Generations: {generations}")
    
    # 1. 无基因丢失的进化
    print(f"\n🔄 Running evolution WITHOUT gene loss...")
    genome_no_loss = initial_genome.copy()
    engine_no_loss = OptimizedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.02,
        recombination_rate=1e-3
    )
    
    engine_no_loss.evolve_multiple_generations(
        genome_no_loss, generations, show_progress=False
    )
    
    print(f"   ✓ No loss: {initial_genome.gene_count:,} → {genome_no_loss.gene_count:,} genes")
    
    # 2. 有基因丢失的进化
    print(f"\n🗑️  Running evolution WITH gene loss...")
    genome_with_loss = initial_genome.copy()
    engine_with_loss = EvolutionEngineWithGeneLoss(
        mutation_rate=1e-4,
        hgt_rate=0.02,
        recombination_rate=1e-3,
        enable_gene_loss=True,
        loss_rate=5e-5,
        core_gene_protection=0.95,
        hgt_gene_loss_multiplier=8.0,
        optimal_genome_size=1400
    )
    
    engine_with_loss.evolve_multiple_generations(
        genome_with_loss, generations, show_progress=False
    )
    
    print(f"   ✓ With loss: {initial_genome.gene_count:,} → {genome_with_loss.gene_count:,} genes")
    
    # 3. 结果对比分析
    print(f"\n📊 Comparison Results:")
    
    # 基因数量变化
    change_no_loss = genome_no_loss.gene_count - initial_genome.gene_count
    change_with_loss = genome_with_loss.gene_count - initial_genome.gene_count
    
    print(f"   Gene count changes:")
    print(f"     Without loss: {change_no_loss:+,} genes")
    print(f"     With loss: {change_with_loss:+,} genes")
    print(f"     Difference: {change_with_loss - change_no_loss:+,} genes")
    
    # 基因组大小变化
    size_change_no_loss = genome_no_loss.size - initial_genome.size
    size_change_with_loss = genome_with_loss.size - initial_genome.size
    
    print(f"   Genome size changes:")
    print(f"     Without loss: {size_change_no_loss:+,} bp")
    print(f"     With loss: {size_change_with_loss:+,} bp")
    print(f"     Difference: {size_change_with_loss - size_change_no_loss:+,} bp")
    
    # 基因组成分析
    print(f"   Final composition:")
    print(f"     Without loss: {genome_no_loss.core_gene_count:,} core, {genome_no_loss.hgt_gene_count:,} HGT")
    print(f"     With loss: {genome_with_loss.core_gene_count:,} core, {genome_with_loss.hgt_gene_count:,} HGT")
    
    # 基因丢失统计
    if engine_with_loss.gene_loss:
        loss_stats = engine_with_loss.gene_loss.get_loss_statistics(genome_with_loss)
        print(f"   Gene loss statistics:")
        print(f"     Total lost: {loss_stats['total_genes_lost']}")
        print(f"     Core lost: {loss_stats['core_genes_lost']}")
        print(f"     HGT lost: {loss_stats['hgt_genes_lost']}")
    
    return {
        'no_loss': genome_no_loss,
        'with_loss': genome_with_loss,
        'initial': initial_genome
    }


def demo_parameter_sensitivity():
    """基因丢失参数敏感性测试"""
    
    print("\n🎛️  Gene Loss Parameter Sensitivity Test")
    print("=" * 60)
    
    # 创建测试基因组
    np.random.seed(456)
    test_genome = create_initial_genome(
        gene_count=1000,
        avg_gene_length=500,
        min_gene_length=100
    )
    
    generations = 50
    
    # 测试不同的丢失率
    loss_rates = [1e-5, 5e-5, 1e-4, 2e-4]
    results = []
    
    print(f"\n🧪 Testing different loss rates:")
    
    for loss_rate in loss_rates:
        print(f"\n   Testing loss rate: {loss_rate}")
        
        genome_copy = test_genome.copy()
        engine = EvolutionEngineWithGeneLoss(
            mutation_rate=1e-4,
            hgt_rate=0.01,
            recombination_rate=5e-4,
            enable_gene_loss=True,
            loss_rate=loss_rate,
            core_gene_protection=0.9,
            hgt_gene_loss_multiplier=5.0,
            optimal_genome_size=900
        )
        
        engine.evolve_multiple_generations(
            genome_copy, generations, show_progress=False
        )
        
        loss_stats = engine.gene_loss.get_loss_statistics(genome_copy)
        
        result = {
            'loss_rate': loss_rate,
            'final_gene_count': genome_copy.gene_count,
            'genes_lost': loss_stats['total_genes_lost'],
            'core_genes_lost': loss_stats['core_genes_lost'],
            'hgt_genes_lost': loss_stats['hgt_genes_lost'],
            'loss_per_generation': loss_stats['avg_total_loss_per_generation']
        }
        
        results.append(result)
        
        print(f"     Final genes: {genome_copy.gene_count:,}")
        print(f"     Genes lost: {loss_stats['total_genes_lost']}")
        print(f"     Loss rate: {loss_stats['avg_total_loss_per_generation']:.3f}/gen")
    
    # 分析结果
    print(f"\n📊 Parameter Sensitivity Analysis:")
    for result in results:
        efficiency = result['genes_lost'] / (result['loss_rate'] * 1e6)  # 标准化效率
        print(f"   Loss rate {result['loss_rate']}: "
              f"{result['genes_lost']} lost, "
              f"efficiency {efficiency:.1f}")
    
    return results


def visualize_gene_loss_results(comparison_results, snapshots=None):
    """可视化基因丢失结果"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gene Loss Mechanism Analysis', fontsize=16)
        
        # 1. 基因数量对比
        categories = ['Initial', 'Without Loss', 'With Loss']
        gene_counts = [
            comparison_results['initial'].gene_count,
            comparison_results['no_loss'].gene_count,
            comparison_results['with_loss'].gene_count
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = axes[0, 0].bar(categories, gene_counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Gene Count')
        axes[0, 0].set_title('Gene Count Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, gene_counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count:,}', ha='center', va='bottom')
        
        # 2. 基因组成对比
        genomes = ['Without Loss', 'With Loss']
        core_genes = [
            comparison_results['no_loss'].core_gene_count,
            comparison_results['with_loss'].core_gene_count
        ]
        hgt_genes = [
            comparison_results['no_loss'].hgt_gene_count,
            comparison_results['with_loss'].hgt_gene_count
        ]
        
        x = np.arange(len(genomes))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, core_genes, width, label='Core genes', color='skyblue', alpha=0.7)
        axes[0, 1].bar(x + width/2, hgt_genes, width, label='HGT genes', color='orange', alpha=0.7)
        
        axes[0, 1].set_xlabel('Evolution Type')
        axes[0, 1].set_ylabel('Gene Count')
        axes[0, 1].set_title('Gene Composition Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(genomes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 基因组大小变化
        if snapshots and len(snapshots) > 1:
            generations = [s.get('snapshot_generation', 0) for s in snapshots]
            sizes = [s['genome_stats']['total_size'] for s in snapshots]
            gene_counts_over_time = [s['genome_stats']['gene_count'] for s in snapshots]
            
            axes[1, 0].plot(generations, gene_counts_over_time, 'b-o', markersize=4, label='Gene count')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Gene Count')
            axes[1, 0].set_title('Gene Count Over Time (with Loss)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 基因丢失统计（如果有快照数据）
        if snapshots and len(snapshots) > 1:
            # 尝试从快照中提取基因丢失数据
            loss_data = []
            for snapshot in snapshots:
                if 'gene_loss_stats' in snapshot:
                    loss_data.append(snapshot['gene_loss_stats']['total_genes_lost'])
            
            if loss_data:
                axes[1, 1].plot(generations[:len(loss_data)], loss_data, 'r-o', markersize=4)
                axes[1, 1].set_xlabel('Generation')
                axes[1, 1].set_ylabel('Cumulative Genes Lost')
                axes[1, 1].set_title('Gene Loss Accumulation')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # 如果没有丢失数据，显示基因组大小变化
                axes[1, 1].plot(generations, sizes, 'g-o', markersize=4)
                axes[1, 1].set_xlabel('Generation')
                axes[1, 1].set_ylabel('Genome Size (bp)')
                axes[1, 1].set_title('Genome Size Over Time')
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # 显示基因类型分布饼图
            with_loss_genome = comparison_results['with_loss']
            labels = ['Core genes', 'HGT genes']
            sizes = [with_loss_genome.core_gene_count, with_loss_genome.hgt_gene_count]
            colors = ['lightblue', 'orange']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Final Gene Composition\n(With Gene Loss)')
        
        plt.tight_layout()
        plt.savefig('gene_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gene loss visualization saved as 'gene_loss_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        print("Skipping visualization step...")


def main():
    """主演示函数"""
    
    print("🧪 Gene Loss Mechanism Demo Suite")
    print("=" * 70)
    
    try:
        # 1. 基础基因丢失演示
        evolved_genome, snapshots, engine = demo_basic_gene_loss()
        
        # 2. 基因丢失对比演示
        comparison_results = demo_gene_loss_comparison()
        
        # 3. 参数敏感性测试
        sensitivity_results = demo_parameter_sensitivity()
        
        # 4. 可视化结果
        visualize_gene_loss_results(comparison_results, snapshots)
        
        # 总结
        print("\n" + "=" * 70)
        print("🎉 Gene Loss Demo Suite Completed!")
        print("=" * 70)
        
        print(f"\n📋 Key Findings:")
        initial_count = comparison_results['initial'].gene_count
        no_loss_count = comparison_results['no_loss'].gene_count
        with_loss_count = comparison_results['with_loss'].gene_count
        
        print(f"   Without gene loss: {initial_count:,} → {no_loss_count:,} genes ({no_loss_count - initial_count:+,})")
        print(f"   With gene loss: {initial_count:,} → {with_loss_count:,} genes ({with_loss_count - initial_count:+,})")
        print(f"   Net effect of gene loss: {with_loss_count - no_loss_count:+,} genes")
        
        if engine.gene_loss:
            loss_stats = engine.gene_loss.get_loss_statistics(evolved_genome)
            print(f"   Gene loss efficiency: {loss_stats['avg_total_loss_per_generation']:.3f} genes/generation")
        
        print(f"\n💡 Biological Insights:")
        print(f"   - Gene loss provides genome size regulation")
        print(f"   - HGT genes are preferentially lost")
        print(f"   - Core genes are protected from loss")
        print(f"   - Genome reaches dynamic equilibrium")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()