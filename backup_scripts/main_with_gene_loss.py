#!/usr/bin/env python3
"""
集成基因丢失机制的原核生物基因组进化模拟器
主程序入口 - 包含基因获得和基因丢失的完整进化模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from core.genome import create_initial_genome
from core.evolution_engine_with_gene_loss import EvolutionEngineWithGeneLoss
from analysis.ani_calculator import ANICalculator


def run_complete_evolution_simulation(generations: int = 1000,
                                    initial_gene_count: int = 3000,
                                    mutation_rate: float = 1e-8,
                                    hgt_rate: float = 0.001,
                                    recombination_rate: float = 1e-6,
                                    # 基因丢失参数
                                    enable_gene_loss: bool = True,
                                    loss_rate: float = 1e-6,
                                    core_gene_protection: float = 0.95,
                                    hgt_gene_loss_multiplier: float = 10.0,
                                    optimal_genome_size: int = 3000):
    """运行完整的进化模拟（包含基因丢失）"""
    
    print("=" * 80)
    print("Complete Prokaryotic Genome Evolution Simulator")
    print("(Including Gene Gain and Gene Loss)")
    print("=" * 80)
    
    # 1. 创建初始基因组
    print("1. Creating initial genome with realistic gene length distribution...")
    initial_genome = create_initial_genome(
        gene_count=initial_gene_count,
        avg_gene_length=1000,
        min_gene_length=100
    )
    
    print(f"   ✓ Genome created: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    print(f"   ✓ Core genes: {initial_genome.core_gene_count:,}")
    print(f"   ✓ HGT genes: {initial_genome.hgt_gene_count:,}")
    print()
    
    # 2. 初始化完整进化引擎
    print("2. Initializing complete evolution engine...")
    evolution_engine = EvolutionEngineWithGeneLoss(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate,
        # 基因丢失参数
        enable_gene_loss=enable_gene_loss,
        loss_rate=loss_rate,
        core_gene_protection=core_gene_protection,
        hgt_gene_loss_multiplier=hgt_gene_loss_multiplier,
        optimal_genome_size=optimal_genome_size
    )
    
    print(f"   Evolution mechanisms:")
    print(f"     Point mutation rate: {mutation_rate}")
    print(f"     Horizontal gene transfer rate: {hgt_rate}")
    print(f"     Homologous recombination rate: {recombination_rate}")
    if enable_gene_loss:
        print(f"     Gene loss rate: {loss_rate}")
        print(f"     Core gene protection: {core_gene_protection*100:.1f}%")
        print(f"     HGT loss multiplier: {hgt_gene_loss_multiplier}x")
    else:
        print(f"     Gene loss: Disabled")
    print()
    
    # 3. 运行进化模拟
    print("3. Starting complete evolution simulation...")
    evolved_genome, snapshots = evolution_engine.simulate_evolution(
        initial_genome=initial_genome,
        generations=generations,
        save_snapshots=True,
        snapshot_interval=max(10, generations // 20)
    )
    print()
    
    # 4. 计算ANI和直系同源基因分析
    print("4. Analyzing evolution results...")
    ani_calculator = ANICalculator()
    
    comprehensive_analysis = ani_calculator.compare_genomes_comprehensive(
        ancestral_genome=initial_genome,
        evolved_genome=evolved_genome
    )
    
    # 5. 输出结果
    print_complete_analysis_results(comprehensive_analysis, evolution_engine, evolved_genome, generations)
    
    # 6. 可视化结果
    visualize_complete_results(comprehensive_analysis, snapshots, evolution_engine, evolved_genome, generations)
    
    return initial_genome, evolved_genome, comprehensive_analysis, snapshots, evolution_engine


def print_complete_analysis_results(analysis: dict, engine, evolved_genome, generations: int):
    """打印完整分析结果"""
    
    print("=" * 80)
    print("Complete Evolution Analysis Results")
    print("=" * 80)
    
    # ANI结果
    ani_data = analysis['ani_analysis']
    print(f"🧬 Evolution Results:")
    print(f"   Average Nucleotide Identity (ANI): {ani_data['ani']:.4f}")
    print(f"   Weighted ANI: {ani_data['weighted_ani']:.4f}")
    print(f"   Orthologous gene pairs: {ani_data['ortholog_count']}")
    print(f"   Ortholog ratio: {ani_data['ortholog_ratio']:.4f}")
    print()
    
    # 基因组变化
    composition = analysis['genome_composition']
    print(f"📊 Genome Composition Changes:")
    print(f"   Ancestral core genes: {composition['ancestral_core_genes']}")
    print(f"   Evolved core genes: {composition['evolved_core_genes']}")
    print(f"   HGT acquired genes: {composition['evolved_hgt_genes']}")
    print(f"   Core gene retention rate: {composition['core_gene_retention']:.4f}")
    print()
    
    # 基因组大小变化
    size_changes = analysis['size_changes']
    print(f"📏 Genome Size Changes:")
    print(f"   Initial size: {size_changes['ancestral_size']:,} bp")
    print(f"   Final size: {size_changes['evolved_size']:,} bp")
    print(f"   Size change: {size_changes['size_change']:+,} bp ({size_changes['size_change_ratio']:+.2%})")
    print()
    
    # 基因数量变化
    gene_changes = analysis['gene_count_changes']
    print(f"🔢 Gene Count Changes:")
    print(f"   Initial gene count: {gene_changes['ancestral_gene_count']}")
    print(f"   Final gene count: {gene_changes['evolved_gene_count']}")
    print(f"   Gene count change: {gene_changes['gene_count_change']:+d} ({gene_changes['gene_count_change_ratio']:+.2%})")
    print()
    
    # 基因丢失分析（如果启用）
    if engine.enable_gene_loss and engine.gene_loss:
        loss_stats = engine.gene_loss.get_loss_statistics(evolved_genome)
        print(f"🗑️  Gene Loss Analysis:")
        print(f"   Total genes lost: {loss_stats['total_genes_lost']}")
        print(f"   Core genes lost: {loss_stats['core_genes_lost']} ({loss_stats['core_loss_percentage']:.1f}%)")
        print(f"   HGT genes lost: {loss_stats['hgt_genes_lost']} ({loss_stats['hgt_loss_percentage']:.1f}%)")
        print(f"   Average loss per generation: {loss_stats['avg_total_loss_per_generation']:.3f}")
        
        # 基因丢失模式
        loss_patterns = engine.gene_loss.analyze_loss_patterns(evolved_genome)
        if 'error' not in loss_patterns:
            print(f"   HGT loss enrichment: {loss_patterns['hgt_loss_enrichment']:.2f}x")
            print(f"   Protection effectiveness: {loss_patterns['protection_effectiveness']:.3f}")
        print()
    
    # 进化事件总结
    print(f"⚡ Evolution Events Summary:")
    print(f"   Total mutations: {evolved_genome.total_mutations:,}")
    print(f"   Total HGT events: {evolved_genome.total_hgt_events:,}")
    print(f"   Total recombination events: {evolved_genome.total_recombination_events:,}")
    
    if engine.enable_gene_loss and engine.gene_loss:
        loss_stats = engine.gene_loss.get_loss_statistics(evolved_genome)
        print(f"   Total gene loss events: {loss_stats['total_genes_lost']:,}")
    
    print()


def visualize_complete_results(analysis: dict, snapshots: list, engine, evolved_genome, generations: int):
    """可视化完整结果"""
    
    try:
        # 根据是否有基因丢失调整布局
        if engine.enable_gene_loss:
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        fig.suptitle(f'Complete Genome Evolution Results ({generations} generations)', fontsize=16)
        
        # 1. 直系同源基因身份分布
        identity_dist = analysis['identity_distribution']
        if identity_dist['counts']:
            axes[0, 0].hist(analysis['ani_analysis']['identity_distribution'], 
                           bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(identity_dist['mean'], color='red', linestyle='--', 
                              label=f'Mean: {identity_dist["mean"]:.3f}')
            axes[0, 0].set_xlabel('Sequence Identity')
            axes[0, 0].set_ylabel('Number of Gene Pairs')
            axes[0, 0].set_title('Orthologous Gene Identity Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 基因组大小变化
        if len(snapshots) > 1:
            generations_list = [s.get('snapshot_generation', 0) for s in snapshots]
            sizes = [s['genome_stats']['total_size'] for s in snapshots]
            axes[0, 1].plot(generations_list, sizes, 'b-o', markersize=4)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Genome Size (bp)')
            axes[0, 1].set_title('Genome Size Changes')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 基因数量变化
        if len(snapshots) > 1:
            gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
            core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
            hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
            
            axes[1, 0].plot(generations_list, gene_counts, 'g-o', label='Total genes', markersize=4)
            axes[1, 0].plot(generations_list, core_genes, 'b-s', label='Core genes', markersize=4)
            axes[1, 0].plot(generations_list, hgt_genes, 'r-^', label='HGT genes', markersize=4)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Gene Count')
            axes[1, 0].set_title('Gene Count Changes')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 进化事件累积
        if len(snapshots) > 1:
            mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
            hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
            recombinations = [s['genome_stats']['total_recombination_events'] for s in snapshots]
            
            axes[1, 1].plot(generations_list, mutations, 'purple', label='Point mutations', linewidth=2)
            axes[1, 1].plot(generations_list, hgt_events, 'orange', label='HGT events', linewidth=2)
            axes[1, 1].plot(generations_list, recombinations, 'green', label='Recombination events', linewidth=2)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Cumulative Events')
            axes[1, 1].set_title('Evolution Event Accumulation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 基因丢失分析（如果启用）
        if engine.enable_gene_loss and len(axes) > 2:
            # 基因丢失累积
            if engine.gene_loss:
                loss_stats = engine.gene_loss.get_loss_statistics(evolved_genome)
                
                # 基因丢失类型分布
                loss_types = ['Core genes lost', 'HGT genes lost']
                loss_counts = [loss_stats['core_genes_lost'], loss_stats['hgt_genes_lost']]
                colors = ['lightblue', 'orange']
                
                axes[2, 0].bar(loss_types, loss_counts, color=colors, alpha=0.7, edgecolor='black')
                axes[2, 0].set_ylabel('Genes Lost')
                axes[2, 0].set_title('Gene Loss by Type')
                axes[2, 0].grid(True, alpha=0.3, axis='y')
                
                # 添加数值标签
                for i, count in enumerate(loss_counts):
                    axes[2, 0].text(i, count + count*0.01, f'{count}', ha='center', va='bottom')
            
            # 基因组平衡分析
            if len(snapshots) > 1:
                # 计算净基因变化（获得 - 丢失）
                net_changes = []
                for i, snapshot in enumerate(snapshots):
                    if i == 0:
                        initial_count = snapshot['genome_stats']['gene_count']
                        net_changes.append(0)
                    else:
                        current_count = snapshot['genome_stats']['gene_count']
                        net_change = current_count - initial_count
                        net_changes.append(net_change)
                
                axes[2, 1].plot(generations_list, net_changes, 'darkgreen', linewidth=2, marker='o', markersize=4)
                axes[2, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[2, 1].set_xlabel('Generation')
                axes[2, 1].set_ylabel('Net Gene Change')
                axes[2, 1].set_title('Genome Size Balance (Gain - Loss)')
                axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('complete_evolution_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Complete evolution visualization saved as 'complete_evolution_results.png'")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        print("Skipping visualization step...")


def main():
    """主函数"""
    
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    print("🧬 Welcome to Complete Prokaryotic Genome Evolution Simulator!")
    print("Choose simulation mode:")
    print("1. Run complete evolution (with gene loss)")
    print("2. Run traditional evolution (without gene loss)")
    print("3. Compare both approaches")
    
    try:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            # 运行完整模拟（包含基因丢失）
            print("\n" + "="*60)
            print("COMPLETE EVOLUTION SIMULATION")
            print("="*60)
            
            initial_genome, evolved_genome, analysis, snapshots, engine = run_complete_evolution_simulation(
                generations=200,           # 进化代数
                initial_gene_count=3000,   # 初始基因数
                mutation_rate=1e-5,        # 点突变率
                hgt_rate=0.02,            # HGT率
                recombination_rate=1e-3,   # 重组率
                # 基因丢失参数
                enable_gene_loss=True,
                loss_rate=1e-5,           # 基因丢失率
                core_gene_protection=0.95, # 核心基因保护
                hgt_gene_loss_multiplier=8.0, # HGT基因丢失倍数
                optimal_genome_size=2800   # 最优基因组大小
            )
            
        elif choice == "2":
            # 运行传统模拟（无基因丢失）
            print("\n" + "="*60)
            print("TRADITIONAL EVOLUTION SIMULATION")
            print("="*60)
            
            initial_genome, evolved_genome, analysis, snapshots, engine = run_complete_evolution_simulation(
                generations=200,
                initial_gene_count=3000,
                mutation_rate=1e-5,
                hgt_rate=0.02,
                recombination_rate=1e-3,
                enable_gene_loss=False    # 关闭基因丢失
            )
            
        elif choice == "3":
            # 对比两种方法
            print("\n" + "="*60)
            print("COMPARATIVE ANALYSIS")
            print("="*60)
            
            # 创建相同的初始基因组
            np.random.seed(42)
            base_genome = create_initial_genome(
                gene_count=2000,
                avg_gene_length=800,
                min_gene_length=200
            )
            
            print("Running evolution WITHOUT gene loss...")
            genome_no_loss = base_genome.copy()
            _, _, analysis_no_loss, _, engine_no_loss = run_complete_evolution_simulation(
                generations=100,
                initial_gene_count=2000,
                mutation_rate=1e-4,
                hgt_rate=0.03,
                recombination_rate=1e-3,
                enable_gene_loss=False
            )
            
            print("\n" + "="*60)
            print("Running evolution WITH gene loss...")
            genome_with_loss = base_genome.copy()
            _, _, analysis_with_loss, _, engine_with_loss = run_complete_evolution_simulation(
                generations=100,
                initial_gene_count=2000,
                mutation_rate=1e-4,
                hgt_rate=0.03,
                recombination_rate=1e-3,
                enable_gene_loss=True,
                loss_rate=5e-5,
                core_gene_protection=0.9,
                hgt_gene_loss_multiplier=6.0
            )
            
            # 对比分析
            print("\n" + "="*60)
            print("COMPARATIVE RESULTS")
            print("="*60)
            
            no_loss_final = analysis_no_loss['gene_count_changes']['evolved_gene_count']
            with_loss_final = analysis_with_loss['gene_count_changes']['evolved_gene_count']
            initial_count = analysis_no_loss['gene_count_changes']['ancestral_gene_count']
            
            print(f"📊 Gene Count Comparison:")
            print(f"   Initial: {initial_count:,} genes")
            print(f"   Without loss: {no_loss_final:,} genes ({no_loss_final - initial_count:+,})")
            print(f"   With loss: {with_loss_final:,} genes ({with_loss_final - initial_count:+,})")
            print(f"   Difference: {with_loss_final - no_loss_final:+,} genes")
            
            if engine_with_loss.gene_loss:
                loss_stats = engine_with_loss.gene_loss.get_loss_statistics(genome_with_loss)
                print(f"\n🗑️  Gene Loss Impact:")
                print(f"   Total genes lost: {loss_stats['total_genes_lost']}")
                print(f"   Loss rate: {loss_stats['avg_total_loss_per_generation']:.3f} genes/generation")
            
        else:
            print("❌ Invalid choice. Please run the program again.")
            return
        
        print("\n🎉 Simulation completed successfully!")
        print("📊 Check the generated visualization files for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()