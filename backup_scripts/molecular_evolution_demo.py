#!/usr/bin/env python3
"""
Molecular Evolution Demo - 分子进化模拟器演示程序

展示基于分子进化理论优化后的进化模拟器功能：
1. 选择压力和基因功能重要性
2. 密码子使用偏好性
3. 增强的HGT机制
4. 改进的同源重组
5. 保守性分析
"""

import time
import numpy as np
from core.genome import create_initial_genome
from core.molecular_evolution_engine import MolecularEvolutionEngine
from mechanisms.enhanced_horizontal_transfer import EnhancedHorizontalGeneTransfer
from mechanisms.enhanced_homologous_recombination import EnhancedHomologousRecombination
from analysis.ani_calculator import ANICalculator
from analysis.conservation_analyzer import ConservationAnalyzer

def run_molecular_evolution_demo():
    """运行分子进化演示"""
    
    print("🧬" + "=" * 79)
    print("🧬 MOLECULAR EVOLUTION SIMULATOR - ENHANCED DEMO")
    print("🧬" + "=" * 79)
    print("🔬 Based on modern molecular evolution theory")
    print("⚙️  Features: Selection pressure, Codon bias, Enhanced HGT, Improved recombination")
    print("📊 Analysis: ANI calculation, Conservation analysis, Molecular statistics")
    print("=" * 80)
    
    # 1. 创建初始基因组
    print("\n🧬 STEP 1: Creating initial genome...")
    initial_genome = create_initial_genome(
        gene_count=2000,        # 中等规模基因组
        avg_gene_length=1000,   # 平均基因长度
        min_gene_length=150     # 最小基因长度
    )
    
    # 2. 设置分子进化引擎
    print("\n⚙️  STEP 2: Setting up molecular evolution engine...")
    
    # 创建增强的进化机制
    enhanced_hgt = EnhancedHorizontalGeneTransfer(
        hgt_rate=0.003,                    # 稍高的HGT率
        enable_transfer_barriers=True,      # 启用转移屏障
        enable_metabolic_integration=True,  # 启用代谢整合
        gc_content_tolerance=0.12          # GC含量容忍度
    )
    
    enhanced_recombination = EnhancedHomologousRecombination(
        recombination_rate=2e-6,           # 稍高的重组率
        min_similarity=0.75,               # 提高相似性要求
        enable_recombination_hotspots=True, # 启用重组热点
        enable_gene_conversion=True,        # 启用基因转换
        enable_functional_protection=True   # 启用功能保护
    )
    
    # 创建分子进化引擎
    evolution_engine = MolecularEvolutionEngine(
        mutation_rate=2e-9,                # 真实的突变率
        hgt_rate=0.003,                    # HGT率
        recombination_rate=2e-6,           # 重组率
        enable_selection=True,             # 启用选择压力
        enable_codon_bias=True,            # 启用密码子偏好性
        enable_functional_constraints=True  # 启用功能约束
    )
    
    # 替换为增强的机制
    evolution_engine.hgt = enhanced_hgt
    evolution_engine.recombination = enhanced_recombination
    
    print("✓ Molecular evolution engine configured")
    print(f"  - Selection pressure: {evolution_engine.enable_selection}")
    print(f"  - Codon bias: {evolution_engine.enable_codon_bias}")
    print(f"  - Functional constraints: {evolution_engine.enable_functional_constraints}")
    print(f"  - Enhanced HGT: Transfer barriers + Metabolic integration")
    print(f"  - Enhanced recombination: Hotspots + Gene conversion + Protection")
    
    # 3. 运行分子进化模拟
    print("\n🚀 STEP 3: Running molecular evolution simulation...")
    
    generations = 500  # 中等代数用于演示
    
    start_time = time.time()
    evolved_genome, snapshots = evolution_engine.simulate_molecular_evolution(
        initial_genome=initial_genome,
        generations=generations,
        save_snapshots=True,
        snapshot_interval=100
    )
    simulation_time = time.time() - start_time
    
    print(f"\n✅ Simulation completed in {simulation_time/60:.2f} minutes")
    
    # 4. 分析结果
    print("\n📊 STEP 4: Analyzing evolution results...")
    
    # ANI分析
    print("\n🔍 Performing ANI analysis...")
    ani_calculator = ANICalculator(
        ortholog_identity_threshold=0.5,
        min_alignment_length=100
    )
    
    ani_analysis = ani_calculator.compare_genomes_comprehensive(
        initial_genome, evolved_genome
    )
    
    # 保守性分析
    print("🔬 Performing conservation analysis...")
    conservation_analyzer = ConservationAnalyzer(
        conservation_threshold=0.3,
        high_conservation_threshold=0.8,
        moderate_conservation_threshold=0.6
    )
    
    conservation_analysis = conservation_analyzer.analyze_genome_conservation(
        evolved_genome, initial_genome
    )
    
    # 5. 显示详细结果
    print("\n" + "=" * 80)
    print("📈 MOLECULAR EVOLUTION RESULTS SUMMARY")
    print("=" * 80)
    
    # 基本统计
    print(f"\n📊 Genome Evolution Summary:")
    print(f"   Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    print(f"   Final genome: {evolved_genome.gene_count:,} genes, {evolved_genome.size:,} bp")
    print(f"   Gene count change: {evolved_genome.gene_count - initial_genome.gene_count:+,}")
    print(f"   Size change: {evolved_genome.size - initial_genome.size:+,} bp")
    print(f"   Generations: {generations:,}")
    
    # 进化事件统计
    print(f"\n⚙️  Evolution Events:")
    print(f"   Total mutations: {evolved_genome.total_mutations:,}")
    print(f"   HGT events: {evolved_genome.total_hgt_events:,}")
    print(f"   Recombination events: {evolved_genome.total_recombination_events:,}")
    print(f"   Mutations per generation: {evolved_genome.total_mutations / generations:.2f}")
    
    # ANI结果
    ani_result = ani_analysis['ani_analysis']
    print(f"\n🧬 ANI Analysis:")
    print(f"   Average Nucleotide Identity: {ani_result['ani']:.4f}")
    print(f"   Weighted ANI: {ani_result['weighted_ani']:.4f}")
    print(f"   Orthologous genes: {ani_result['ortholog_count']:,}")
    print(f"   Ortholog ratio: {ani_result['ortholog_ratio']:.3f}")
    
    # 保守性结果
    print(f"\n🛡️  Conservation Analysis:")
    print(f"   Conservative genes: {conservation_analysis['conservative_genes']:,}")
    print(f"   Conservative ratio: {conservation_analysis['conservative_ratio']:.3f}")
    
    categories = conservation_analysis['conservation_categories']
    print(f"   Highly conserved: {categories['highly_conserved']:,}")
    print(f"   Moderately conserved: {categories['moderately_conserved']:,}")
    print(f"   Poorly conserved: {categories['poorly_conserved']:,}")
    print(f"   Non-conserved: {categories['non_conserved']:,}")
    
    # 分子进化特有统计
    if 'molecular_evolution_stats' in snapshots[-1]:
        mol_stats = snapshots[-1]['molecular_evolution_stats']['evolution_stats']
        total_tracked = sum(mol_stats.values())
        
        if total_tracked > 0:
            print(f"\n🔬 Molecular Evolution Details:")
            print(f"   Synonymous mutations: {mol_stats['synonymous_mutations']:,} "
                  f"({mol_stats['synonymous_mutations']/total_tracked*100:.1f}%)")
            print(f"   Non-synonymous mutations: {mol_stats['nonsynonymous_mutations']:,} "
                  f"({mol_stats['nonsynonymous_mutations']/total_tracked*100:.1f}%)")
            print(f"   Neutral mutations: {mol_stats['neutral_mutations']:,}")
            print(f"   Selected against: {mol_stats['selected_against_mutations']:,}")
            print(f"   Beneficial mutations: {mol_stats['beneficial_mutations']:,}")
            
            # dN/dS比率
            if mol_stats['synonymous_mutations'] > 0:
                dn_ds = mol_stats['nonsynonymous_mutations'] / mol_stats['synonymous_mutations']
                print(f"   Approximate dN/dS ratio: {dn_ds:.3f}")
    
    # 6. 显示增强机制的详细分析
    print("\n" + "=" * 80)
    print("🔬 ENHANCED MECHANISMS ANALYSIS")
    print("=" * 80)
    
    # HGT分析
    enhanced_hgt.print_hgt_analysis(evolved_genome)
    
    # 重组分析
    enhanced_recombination.print_recombination_analysis(evolved_genome)
    
    # 保守性详细分析
    conservation_analyzer.print_conservation_summary(conservation_analysis)
    
    # 7. 性能统计
    print(f"\n⏱️  Performance Statistics:")
    print(f"   Total simulation time: {simulation_time:.2f} seconds")
    print(f"   Average time per generation: {simulation_time/generations:.4f} seconds")
    print(f"   Generations per second: {generations/simulation_time:.2f}")
    print(f"   Final genome processing rate: {evolved_genome.size/simulation_time:.0f} bp/second")
    
    print("\n" + "=" * 80)
    print("🎉 MOLECULAR EVOLUTION DEMO COMPLETED!")
    print("=" * 80)
    
    return {
        'initial_genome': initial_genome,
        'evolved_genome': evolved_genome,
        'ani_analysis': ani_analysis,
        'conservation_analysis': conservation_analysis,
        'simulation_time': simulation_time,
        'snapshots': snapshots
    }

def run_comparison_demo():
    """运行对比演示 - 比较有无分子进化特性的差异"""
    
    print("\n🔬" + "=" * 79)
    print("🔬 MOLECULAR EVOLUTION COMPARISON DEMO")
    print("🔬" + "=" * 79)
    print("📊 Comparing evolution with and without molecular constraints")
    print("=" * 80)
    
    # 创建相同的初始基因组
    print("\n🧬 Creating identical initial genomes for comparison...")
    base_genome = create_initial_genome(gene_count=1000, avg_gene_length=800)
    
    # 场景1：无分子约束的进化
    print("\n🚀 Scenario 1: Evolution WITHOUT molecular constraints...")
    unconstrained_engine = MolecularEvolutionEngine(
        mutation_rate=2e-9,
        enable_selection=False,
        enable_codon_bias=False,
        enable_functional_constraints=False
    )
    
    unconstrained_genome = base_genome.copy()
    unconstrained_result, _ = unconstrained_engine.simulate_molecular_evolution(
        unconstrained_genome, generations=200, save_snapshots=False
    )
    
    # 场景2：有分子约束的进化
    print("\n🔬 Scenario 2: Evolution WITH molecular constraints...")
    constrained_engine = MolecularEvolutionEngine(
        mutation_rate=2e-9,
        enable_selection=True,
        enable_codon_bias=True,
        enable_functional_constraints=True
    )
    
    constrained_genome = base_genome.copy()
    constrained_result, _ = constrained_engine.simulate_molecular_evolution(
        constrained_genome, generations=200, save_snapshots=False
    )
    
    # 比较结果
    print("\n📊 COMPARISON RESULTS:")
    print("=" * 60)
    
    print(f"📈 Genome Size Changes:")
    unconstrained_size_change = unconstrained_result.size - base_genome.size
    constrained_size_change = constrained_result.size - base_genome.size
    print(f"   Without constraints: {unconstrained_size_change:+,} bp")
    print(f"   With constraints: {constrained_size_change:+,} bp")
    
    print(f"\n🧬 Gene Count Changes:")
    unconstrained_gene_change = unconstrained_result.gene_count - base_genome.gene_count
    constrained_gene_change = constrained_result.gene_count - base_genome.gene_count
    print(f"   Without constraints: {unconstrained_gene_change:+,} genes")
    print(f"   With constraints: {constrained_gene_change:+,} genes")
    
    print(f"\n⚙️  Mutation Accumulation:")
    print(f"   Without constraints: {unconstrained_result.total_mutations:,} mutations")
    print(f"   With constraints: {constrained_result.total_mutations:,} mutations")
    
    # ANI比较
    ani_calc = ANICalculator()
    
    unconstrained_ani = ani_calc.calculate_ani(base_genome, unconstrained_result)['ani']
    constrained_ani = ani_calc.calculate_ani(base_genome, constrained_result)['ani']
    
    print(f"\n🔍 ANI with Original Genome:")
    print(f"   Without constraints: {unconstrained_ani:.4f}")
    print(f"   With constraints: {constrained_ani:.4f}")
    print(f"   Difference: {constrained_ani - unconstrained_ani:+.4f}")
    
    print("\n💡 Interpretation:")
    if constrained_ani > unconstrained_ani:
        print("   ✓ Molecular constraints preserve genome integrity better")
    print("   ✓ Selection pressure reduces harmful mutations")
    print("   ✓ Functional constraints protect important genes")
    
    print("=" * 60)

if __name__ == "__main__":
    # 运行主演示
    demo_results = run_molecular_evolution_demo()
    
    # 询问是否运行对比演示
    print("\n" + "=" * 80)
    response = input("🤔 Would you like to run the comparison demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_comparison_demo()
    
    print("\n🎉 All demos completed! Thank you for using the Molecular Evolution Simulator!")