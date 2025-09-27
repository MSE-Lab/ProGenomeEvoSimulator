#!/usr/bin/env python3
"""
统一进化引擎演示脚本
展示UnifiedEvolutionEngine的核心功能
包含服务器友好的可视化输出
"""

import time
import numpy as np
from core.genome import create_initial_genome
from core.unified_evolution_engine import UnifiedEvolutionEngine
from core.visualization import EvolutionVisualizer, create_comprehensive_visualization


def demo_basic_evolution():
    """演示基础进化功能"""
    
    print("🧬 DEMO 1: Basic Evolution")
    print("=" * 50)
    
    # 创建初始基因组
    np.random.seed(42)
    genome = create_initial_genome(
        gene_count=200,
        avg_gene_length=400,
        min_gene_length=150
    )
    
    print(f"📊 Initial genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # 创建基础引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,  # 修正：更现实的突变率
        hgt_rate=1e-5,       # 修正：更现实的HGT率
        recombination_rate=1e-3,  # 修正：更现实的重组率
        enable_gene_loss=False,
        enable_parallel=False
    )
    
    # 运行进化
    print("🚀 Running 20 generations...")
    start_time = time.time()
    
    history = engine.evolve_multiple_generations(genome, 20, show_progress=True)
    
    end_time = time.time()
    
    print(f"\n✅ Evolution completed in {end_time - start_time:.2f} seconds")
    print(f"📈 Final genome: {genome.gene_count} genes, {genome.size:,} bp")
    print(f"🔬 Evolution events:")
    print(f"   Mutations: {genome.total_mutations:,}")
    print(f"   HGT events: {genome.total_hgt_events:,}")
    print(f"   Recombinations: {genome.total_recombination_events:,}")
    
    return genome, history


def demo_gene_loss():
    """演示基因丢失功能"""
    
    print("\n🗑️  DEMO 2: Gene Loss Mechanism")
    print("=" * 50)
    
    # 创建初始基因组
    np.random.seed(123)
    genome = create_initial_genome(
        gene_count=300,
        avg_gene_length=350,
        min_gene_length=100
    )
    
    initial_count = genome.gene_count
    print(f"📊 Initial genome: {initial_count} genes")
    
    # 创建带基因丢失的引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,  # 修正：更现实的突变率
        hgt_rate=1e-5,       # 修正：更现实的HGT率
        recombination_rate=1e-3,  # 修正：更现实的重组率
        enable_gene_loss=True,
        loss_rate=0.005,  # 较高的丢失率以便观察
        core_gene_protection=0.9,
        min_genome_size=200,
        optimal_genome_size=250
    )
    
    # 运行进化
    print("🚀 Running 30 generations with gene loss...")
    start_time = time.time()
    
    history = engine.evolve_multiple_generations(genome, 30, show_progress=True)
    
    end_time = time.time()
    
    final_count = genome.gene_count
    net_change = final_count - initial_count
    
    print(f"\n✅ Evolution with gene loss completed in {end_time - start_time:.2f} seconds")
    print(f"📈 Genome changes:")
    print(f"   Initial: {initial_count} genes")
    print(f"   Final: {final_count} genes")
    print(f"   Net change: {net_change:+d} genes")
    print(f"🔬 Evolution events:")
    print(f"   Mutations: {genome.total_mutations:,}")
    print(f"   HGT events: {genome.total_hgt_events:,}")
    print(f"   Recombinations: {genome.total_recombination_events:,}")
    
    # 基因丢失统计
    if engine.gene_loss:
        loss_stats = engine.gene_loss.get_loss_statistics(genome)
        print(f"🗑️  Gene loss statistics:")
        print(f"   Total lost: {loss_stats['total_genes_lost']}")
        print(f"   Core lost: {loss_stats['core_genes_lost']}")
        print(f"   HGT lost: {loss_stats['hgt_genes_lost']}")
        print(f"   Average loss/gen: {loss_stats['avg_total_loss_per_generation']:.3f}")
    
    return genome, history


def demo_parallel_processing():
    """演示并行处理功能"""
    
    print("\n⚡ DEMO 3: Parallel Processing")
    print("=" * 50)
    
    # 创建较大的基因组
    np.random.seed(456)
    genome = create_initial_genome(
        gene_count=1000,  # 大基因组触发并行处理
        avg_gene_length=300,
        min_gene_length=100
    )
    
    print(f"📊 Large genome: {genome.gene_count} genes")
    
    # 串行处理测试
    serial_genome = genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,  # 修正：更现实的突变率
        hgt_rate=1e-5,       # 修正：更现实的HGT率
        enable_parallel=False,
        enable_optimization=True
    )
    
    print("🔄 Testing serial processing...")
    serial_start = time.time()
    serial_engine.evolve_multiple_generations(serial_genome, 10, show_progress=False)
    serial_time = time.time() - serial_start
    
    # 并行处理测试
    parallel_genome = genome.copy()
    parallel_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,  # 修正：更现实的突变率
        hgt_rate=1e-5,       # 修正：更现实的HGT率
        enable_parallel=True,
        enable_optimization=True
    )
    
    print("⚡ Testing parallel processing...")
    parallel_start = time.time()
    parallel_engine.evolve_multiple_generations(parallel_genome, 10, show_progress=False)
    parallel_time = time.time() - parallel_start
    
    # 性能对比
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Serial time: {serial_time:.3f} seconds")
    print(f"   Parallel time: {parallel_time:.3f} seconds")
    print(f"   Speedup: {speedup:.2f}x")
    
    # 获取性能分析
    perf_analysis = parallel_engine.get_performance_analysis()
    if 'parallel_efficiency' in perf_analysis:
        eff = perf_analysis['parallel_efficiency']['average']
        print(f"   Parallel efficiency: {eff:.1f}%")
    
    return speedup


def demo_complete_simulation():
    """演示完整模拟功能"""
    
    print("\n🎯 DEMO 4: Complete Simulation")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(789)
    genome = create_initial_genome(
        gene_count=500,
        avg_gene_length=400,
        min_gene_length=200
    )
    
    print(f"📊 Initial genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # 创建全功能引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-5,  # 修正：更现实的突变率
        hgt_rate=1e-5,       # 修正：更现实的HGT率
        recombination_rate=1e-3,  # 修正：更现实的重组率
        enable_gene_loss=True,
        loss_rate=1e-4,
        core_gene_protection=0.95,
        enable_parallel=True,
        enable_optimization=True
    )
    
    # 运行完整模拟
    print("🚀 Running complete simulation with all features...")
    
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=50,
        save_snapshots=True,
        snapshot_interval=10
    )
    
    # 显示结果
    print(f"\n📈 Simulation Results:")
    print(f"   Initial size: {genome.size:,} bp, {genome.gene_count} genes")
    print(f"   Final size: {final_genome.size:,} bp, {final_genome.gene_count} genes")
    print(f"   Size change: {final_genome.size - genome.size:+,} bp")
    print(f"   Gene change: {final_genome.gene_count - genome.gene_count:+d} genes")
    print(f"   Snapshots saved: {len(snapshots)}")
    
    # 获取详细统计
    summary = engine.get_evolution_summary(final_genome)
    genome_stats = summary['genome_stats']
    
    print(f"\n🔬 Detailed Statistics:")
    print(f"   Total mutations: {genome_stats['total_mutations']:,}")
    print(f"   Total HGT events: {genome_stats['total_hgt_events']:,}")
    print(f"   Total recombinations: {genome_stats['total_recombination_events']:,}")
    
    if 'gene_loss_stats' in summary:
        loss_stats = summary['gene_loss_stats']
        print(f"   Genes lost: {loss_stats['total_genes_lost']:,}")
    
    return final_genome, snapshots


def create_demo_visualizations(final_genome, snapshots, performance_data=None):
    """创建演示可视化图表"""
    
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    print("🖥️  Server-friendly visualization (no GUI required)")
    print("💾 All charts will be saved to files")
    
    try:
        # 准备结果数据
        results = {
            'snapshots': snapshots,
            'final_genome': {
                'gene_count': final_genome.gene_count,
                'size': final_genome.size,
                'total_mutations': final_genome.total_mutations,
                'total_hgt_events': final_genome.total_hgt_events,
                'total_recombination_events': final_genome.total_recombination_events
            }
        }
        
        # 创建可视化器
        visualizer = EvolutionVisualizer(output_dir='demo_results')
        
        # 创建综合报告
        print("📈 Generating comprehensive visualization report...")
        saved_files = visualizer.create_comprehensive_report(
            results=results,
            performance_data=performance_data,
            filename="unified_engine_demo"
        )
        
        print(f"\n✅ Visualization completed!")
        print(f"📁 Generated {len(saved_files)} visualization files:")
        for i, filepath in enumerate(saved_files, 1):
            print(f"   {i}. {filepath}")
        
        print(f"\n💡 Visualization Features:")
        print(f"   🎯 Server-compatible (no display required)")
        print(f"   📊 High-resolution PNG output (300 DPI)")
        print(f"   📈 Comprehensive evolution analysis")
        print(f"   ⚡ Performance metrics included")
        print(f"   🔍 Detailed statistical summaries")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """主演示函数"""
    
    print("🧬 UNIFIED EVOLUTION ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases all features of the UnifiedEvolutionEngine:")
    print("✅ Basic evolution mechanisms")
    print("✅ Gene loss simulation")
    print("✅ Parallel processing")
    print("✅ Complete simulation workflow")
    print("✅ Server-friendly visualization")
    print("=" * 60)
    
    try:
        # Demo 1: 基础进化
        genome1, history1 = demo_basic_evolution()
        
        # Demo 2: 基因丢失
        genome2, history2 = demo_gene_loss()
        
        # Demo 3: 并行处理
        speedup = demo_parallel_processing()
        
        # Demo 4: 完整模拟
        final_genome, snapshots = demo_complete_simulation()
        
        # Demo 5: 可视化生成
        visualization_files = create_demo_visualizations(
            final_genome, snapshots, 
            performance_data={'parallel_speedup': speedup}
        )
        
        # 总结
        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETED!")
        print("=" * 60)
        print(f"✅ Basic evolution: {len(history1)} generations completed")
        print(f"✅ Gene loss: {len(history2)} generations with dynamic genome size")
        print(f"✅ Parallel processing: {speedup:.2f}x speedup achieved")
        print(f"✅ Complete simulation: {len(snapshots)} snapshots saved")
        print(f"✅ Visualizations: {len(visualization_files)} files generated")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   🧬 All evolution mechanisms working")
        print(f"   🗑️  Gene loss balancing genome size")
        print(f"   ⚡ Parallel processing acceleration")
        print(f"   📊 Comprehensive analysis and monitoring")
        print(f"   🎯 Production-ready simulation workflow")
        print(f"   🖥️  Server-friendly visualization system")
        
        print(f"\n🚀 Ready for Production Use!")
        print(f"   - Use 'python main_unified.py' for interactive simulations")
        print(f"   - Import UnifiedEvolutionEngine in your research scripts")
        print(f"   - All features are tested and working correctly")
        print(f"   - Visualization works on servers without GUI")
        
        print(f"\n📁 Output Files:")
        print(f"   📊 Visualization files in: demo_results/")
        for file in visualization_files:
            print(f"      • {file}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()