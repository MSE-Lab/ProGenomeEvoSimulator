#!/usr/bin/env python3
"""
ProGenomeEvoSimulator - 原核生物基因组进化模拟器
主程序入口 - 使用统一进化引擎

这是项目的主入口文件，提供简化的接口来运行基因组进化模拟。
对于更多高级功能，请使用 main_unified.py
"""

import numpy as np
from core.genome import create_initial_genome
from core.unified_evolution_engine import UnifiedEvolutionEngine


def run_basic_simulation():
    """运行基础的进化模拟"""
    
    print("🧬 ProGenomeEvoSimulator - Basic Simulation")
    print("=" * 60)
    
    # 创建初始基因组
    print("📊 Creating initial genome...")
    np.random.seed(42)  # 确保可重复性
    
    genome = create_initial_genome(
        gene_count=2000,
        avg_gene_length=500,
        min_gene_length=200
    )
    
    print(f"   Initial genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    
    # 创建进化引擎（推荐配置）
    print("⚙️  Initializing evolution engine...")
    
    engine = UnifiedEvolutionEngine(
        # 基本进化参数
        mutation_rate=1e-5,
        hgt_rate=0.01,
        recombination_rate=1e-3,
        
        # 基因丢失参数
        enable_gene_loss=True,
        loss_rate=1e-5,
        core_gene_protection=0.95,
        
        # 性能优化
        enable_parallel=True,
        enable_optimization=True
    )
    
    # 运行进化模拟
    print("🚀 Starting evolution simulation...")
    print("   Generations: 500")
    print("   Features: All mechanisms enabled (mutations, HGT, recombination, gene loss)")
    print("   Processing: Parallel optimization enabled")
    print("=" * 60)
    
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=500,
        save_snapshots=True,
        snapshot_interval=50
    )
    
    # 显示结果摘要
    print("\n📈 SIMULATION RESULTS")
    print("=" * 60)
    print(f"🧬 Genome Evolution:")
    print(f"   Initial: {genome.gene_count:,} genes, {genome.size:,} bp")
    print(f"   Final: {final_genome.gene_count:,} genes, {final_genome.size:,} bp")
    print(f"   Change: {final_genome.gene_count - genome.gene_count:+,} genes, {final_genome.size - genome.size:+,} bp")
    
    print(f"\n🔬 Evolution Events:")
    print(f"   Mutations: {final_genome.total_mutations:,}")
    print(f"   HGT events: {final_genome.total_hgt_events:,}")
    print(f"   Recombinations: {final_genome.total_recombination_events:,}")
    
    # 基因丢失统计
    if engine.gene_loss:
        loss_stats = engine.gene_loss.get_loss_statistics(final_genome)
        print(f"   Genes lost: {loss_stats['total_genes_lost']:,}")
    
    print(f"\n📊 Analysis:")
    print(f"   Snapshots saved: {len(snapshots)}")
    print(f"   Final generation: {final_genome.generation}")
    
    # 性能分析
    perf_analysis = engine.get_performance_analysis()
    if 'processing_modes' in perf_analysis:
        modes = perf_analysis['processing_modes']
        if modes.get('parallel_generations', 0) > 0:
            print(f"   Parallel processing: {modes['parallel_generations']} generations")
    
    print("=" * 60)
    print("✅ Simulation completed successfully!")
    print("\n💡 For more advanced features and options:")
    print("   - Run 'python main_unified.py' for interactive interface")
    print("   - Run 'python demo_unified_engine.py' for feature demonstrations")
    
    return final_genome, snapshots


def main():
    """主函数"""
    
    try:
        print("🧬 Welcome to ProGenomeEvoSimulator!")
        print("This is the basic simulation interface.")
        print("\nStarting simulation with recommended parameters...")
        
        final_genome, snapshots = run_basic_simulation()
        
        print("\n🎉 Thank you for using ProGenomeEvoSimulator!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Simulation interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that all core modules are present")
        print("   3. Try running 'python demo_unified_engine.py' for diagnostics")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
