#!/usr/bin/env python3
"""
并行化原核生物基因组进化模拟器
主程序入口 - 集成并行处理优化
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from typing import Optional, Tuple, Dict, Any

from core.genome import create_initial_genome
from core.parallel_evolution_engine import ParallelEvolutionEngine
from core.evolution_engine_with_conservation import OptimizedEvolutionEngine
from analysis.ani_calculator import ANICalculator


def run_parallel_evolution_simulation(generations: int = 1000,
                                    initial_gene_count: int = 3000,
                                    mutation_rate: float = 1e-8,
                                    hgt_rate: float = 0.001,
                                    recombination_rate: float = 1e-6,
                                    num_processes: Optional[int] = None,
                                    chunk_size: Optional[int] = None) -> Tuple[Any, Any, Dict, list]:
    """运行并行化进化模拟"""
    
    print("=" * 80)
    print("PARALLEL Prokaryotic Genome Evolution Simulator")
    print("=" * 80)
    
    # 显示系统信息
    cpu_count = mp.cpu_count()
    processes_to_use = num_processes or cpu_count
    print(f"🖥️  System Info:")
    print(f"   CPU cores available: {cpu_count}")
    print(f"   Processes to use: {processes_to_use}")
    print(f"   Chunk size: {chunk_size or 'Auto-calculated'}")
    print()
    
    # 1. 创建初始基因组
    print("1. Creating initial genome with realistic gene length distribution...")
    initial_genome = create_initial_genome(
        gene_count=initial_gene_count,
        avg_gene_length=1000,
        min_gene_length=100
    )
    
    print(f"   ✓ Genome created: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    print(f"   ✓ Average gene length: {initial_genome.size // initial_genome.gene_count:,} bp")
    print()
    
    # 2. 初始化并行进化引擎
    print("2. Initializing PARALLEL evolution engine...")
    parallel_engine = ParallelEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate,
        num_processes=num_processes,
        chunk_size=chunk_size,
        enable_progress_sharing=True
    )
    
    print(f"   Point mutation rate: {mutation_rate}")
    print(f"   Horizontal gene transfer rate: {hgt_rate}")
    print(f"   Homologous recombination rate: {recombination_rate}")
    print()
    
    # 3. 运行并行进化模拟
    print("3. Starting PARALLEL evolution simulation...")
    simulation_start_time = time.time()
    
    evolved_genome, snapshots = parallel_engine.simulate_evolution_parallel(
        initial_genome=initial_genome,
        generations=generations,
        save_snapshots=True,
        snapshot_interval=max(10, generations // 20)  # 自适应快照间隔
    )
    
    simulation_end_time = time.time()
    total_simulation_time = simulation_end_time - simulation_start_time
    
    print()
    print("4. Analyzing evolution results...")
    
    # 4. 计算ANI和直系同源基因分析
    ani_calculator = ANICalculator()
    comprehensive_analysis = ani_calculator.compare_genomes_comprehensive(
        ancestral_genome=initial_genome,
        evolved_genome=evolved_genome
    )
    
    # 5. 并行性能分析
    performance_analysis = parallel_engine.get_parallel_performance_analysis()
    
    # 6. 输出结果
    print_parallel_analysis_results(
        comprehensive_analysis, 
        performance_analysis, 
        generations, 
        total_simulation_time
    )
    
    # 7. 可视化结果
    visualize_parallel_results(
        comprehensive_analysis, 
        snapshots, 
        performance_analysis, 
        generations
    )
    
    return initial_genome, evolved_genome, comprehensive_analysis, snapshots


def run_performance_comparison(generations: int = 100,
                             initial_gene_count: int = 1000,
                             mutation_rate: float = 1e-5,
                             hgt_rate: float = 0.02,
                             recombination_rate: float = 1e-3) -> Dict:
    """运行性能对比测试：串行 vs 并行"""
    
    print("=" * 80)
    print("PERFORMANCE COMPARISON: Serial vs Parallel")
    print("=" * 80)
    
    # 创建相同的初始基因组用于对比
    np.random.seed(42)  # 确保可重现性
    initial_genome = create_initial_genome(
        gene_count=initial_gene_count,
        avg_gene_length=1000,
        min_gene_length=100
    )
    
    print(f"📊 Test genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    print(f"🎯 Test generations: {generations}")
    print(f"🖥️  Available CPU cores: {mp.cpu_count()}")
    print()
    
    results = {}
    
    # 1. 串行版本测试
    print("🔄 Testing SERIAL evolution engine...")
    serial_genome = initial_genome.copy()
    
    serial_engine = OptimizedEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate
    )
    
    serial_start_time = time.time()
    serial_evolved_genome, _ = serial_engine.simulate_evolution(
        initial_genome=serial_genome,
        generations=generations,
        save_snapshots=False
    )
    serial_end_time = time.time()
    serial_time = serial_end_time - serial_start_time
    
    results['serial'] = {
        'time': serial_time,
        'speed': generations / serial_time,
        'final_genome_size': serial_evolved_genome.size,
        'final_gene_count': serial_evolved_genome.gene_count,
        'total_mutations': serial_evolved_genome.total_mutations
    }
    
    print(f"   ✓ Serial completed in {serial_time:.2f}s ({generations/serial_time:.2f} gen/s)")
    print()
    
    # 2. 并行版本测试
    print("⚡ Testing PARALLEL evolution engine...")
    parallel_genome = initial_genome.copy()
    
    parallel_engine = ParallelEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate,
        num_processes=None,  # 使用所有可用核心
        enable_progress_sharing=False  # 关闭进度共享以获得最佳性能
    )
    
    parallel_start_time = time.time()
    parallel_evolved_genome, _ = parallel_engine.simulate_evolution_parallel(
        initial_genome=parallel_genome,
        generations=generations,
        save_snapshots=False
    )
    parallel_end_time = time.time()
    parallel_time = parallel_end_time - parallel_start_time
    
    # 获取并行性能分析
    parallel_performance = parallel_engine.get_parallel_performance_analysis()
    
    results['parallel'] = {
        'time': parallel_time,
        'speed': generations / parallel_time,
        'final_genome_size': parallel_evolved_genome.size,
        'final_gene_count': parallel_evolved_genome.gene_count,
        'total_mutations': parallel_evolved_genome.total_mutations,
        'processes_used': parallel_engine.num_processes,
        'parallel_efficiency': parallel_performance.get('avg_parallel_efficiency', 0),
        'actual_speedup': parallel_performance.get('actual_speedup', 0)
    }
    
    print(f"   ✓ Parallel completed in {parallel_time:.2f}s ({generations/parallel_time:.2f} gen/s)")
    print()
    
    # 3. 性能对比分析
    speedup = serial_time / parallel_time
    efficiency = speedup / mp.cpu_count() * 100
    
    results['comparison'] = {
        'speedup': speedup,
        'efficiency': efficiency,
        'time_saved': serial_time - parallel_time,
        'time_saved_percentage': ((serial_time - parallel_time) / serial_time) * 100
    }
    
    # 打印对比结果
    print_performance_comparison_results(results)
    
    return results


def print_parallel_analysis_results(analysis: dict, performance: dict, 
                                  generations: int, simulation_time: float):
    """打印并行分析结果"""
    
    print("=" * 80)
    print("PARALLEL Evolution Analysis Results")
    print("=" * 80)
    
    # 基本进化结果
    ani_data = analysis['ani_analysis']
    print(f"🧬 Evolution Results:")
    print(f"   Average Nucleotide Identity (ANI): {ani_data['ani']:.4f}")
    print(f"   Weighted ANI: {ani_data['weighted_ani']:.4f}")
    print(f"   Orthologous gene pairs: {ani_data['ortholog_count']}")
    print(f"   Ortholog ratio: {ani_data['ortholog_ratio']:.4f}")
    print()
    
    # 基因组变化
    composition = analysis['genome_composition']
    print(f"📊 Genome Changes:")
    print(f"   Core gene retention rate: {composition['core_gene_retention']:.4f}")
    print(f"   HGT acquired genes: {composition['evolved_hgt_genes']}")
    
    size_changes = analysis['size_changes']
    print(f"   Size change: {size_changes['size_change']:+,} bp ({size_changes['size_change_ratio']:+.2%})")
    
    gene_changes = analysis['gene_count_changes']
    print(f"   Gene count change: {gene_changes['gene_count_change']:+d} ({gene_changes['gene_count_change_ratio']:+.2%})")
    print()
    
    # 并行性能结果
    print(f"⚡ Parallel Performance:")
    print(f"   Simulation time: {simulation_time:.2f} seconds ({simulation_time/60:.2f} minutes)")
    print(f"   Evolution speed: {generations/simulation_time:.2f} generations/second")
    print(f"   Processes used: {performance['processes_used']}/{performance['cpu_cores_available']}")
    print(f"   Average parallel efficiency: {performance['avg_parallel_efficiency']:.1f}%")
    print(f"   Actual speedup: {performance['actual_speedup']:.2f}x")
    print(f"   Theoretical max speedup: {performance['theoretical_speedup']:.0f}x")
    print()


def print_performance_comparison_results(results: Dict):
    """打印性能对比结果"""
    
    print("=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)
    
    serial = results['serial']
    parallel = results['parallel']
    comparison = results['comparison']
    
    print(f"📊 Execution Times:")
    print(f"   Serial:   {serial['time']:.2f}s ({serial['speed']:.2f} gen/s)")
    print(f"   Parallel: {parallel['time']:.2f}s ({parallel['speed']:.2f} gen/s)")
    print()
    
    print(f"🚀 Performance Gains:")
    print(f"   Speedup: {comparison['speedup']:.2f}x")
    print(f"   Efficiency: {comparison['efficiency']:.1f}%")
    print(f"   Time saved: {comparison['time_saved']:.2f}s ({comparison['time_saved_percentage']:.1f}%)")
    print()
    
    print(f"⚙️  Parallel Details:")
    print(f"   Processes used: {parallel['processes_used']}")
    print(f"   Parallel efficiency: {parallel['parallel_efficiency']:.1f}%")
    print(f"   Actual speedup: {parallel['actual_speedup']:.2f}x")
    print()
    
    # 结果一致性检查
    size_diff = abs(serial['final_genome_size'] - parallel['final_genome_size'])
    gene_diff = abs(serial['final_gene_count'] - parallel['final_gene_count'])
    
    print(f"✅ Result Consistency:")
    print(f"   Genome size difference: {size_diff} bp")
    print(f"   Gene count difference: {gene_diff} genes")
    
    if size_diff < 1000 and gene_diff < 10:
        print("   ✓ Results are consistent between serial and parallel versions")
    else:
        print("   ⚠️  Significant differences detected - may need investigation")
    
    print("=" * 80)


def visualize_parallel_results(analysis: dict, snapshots: list, 
                             performance: dict, generations: int):
    """可视化并行结果"""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Parallel Genome Evolution Results ({generations} generations)', fontsize=16)
        
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
            
            axes[0, 2].plot(generations_list, gene_counts, 'g-o', label='Total genes', markersize=4)
            axes[0, 2].plot(generations_list, core_genes, 'b-s', label='Core genes', markersize=4)
            axes[0, 2].plot(generations_list, hgt_genes, 'r-^', label='HGT genes', markersize=4)
            axes[0, 2].set_xlabel('Generation')
            axes[0, 2].set_ylabel('Gene Count')
            axes[0, 2].set_title('Gene Count Changes')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 进化事件累积
        if len(snapshots) > 1:
            mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
            hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
            recombinations = [s['genome_stats']['total_recombination_events'] for s in snapshots]
            
            axes[1, 0].plot(generations_list, mutations, 'purple', label='Point mutations', linewidth=2)
            axes[1, 0].plot(generations_list, hgt_events, 'orange', label='HGT events', linewidth=2)
            axes[1, 0].plot(generations_list, recombinations, 'green', label='Recombination events', linewidth=2)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Cumulative Events')
            axes[1, 0].set_title('Evolution Event Accumulation')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 并行性能指标
        processes = performance['processes_used']
        efficiency = performance['avg_parallel_efficiency']
        speedup = performance['actual_speedup']
        theoretical_max = performance['theoretical_speedup']
        
        categories = ['Efficiency (%)', 'Speedup (x)', 'Theoretical Max (x)']
        values = [efficiency, speedup, theoretical_max]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title(f'Parallel Performance ({processes} processes)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 6. CPU利用率可视化
        cpu_cores = performance['cpu_cores_available']
        used_cores = performance['processes_used']
        
        # 创建CPU核心使用图
        core_labels = [f'Core {i+1}' for i in range(cpu_cores)]
        core_usage = [1 if i < used_cores else 0 for i in range(cpu_cores)]
        
        colors = ['green' if usage else 'lightgray' for usage in core_usage]
        axes[1, 2].bar(range(cpu_cores), [1]*cpu_cores, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('CPU Core')
        axes[1, 2].set_ylabel('Usage')
        axes[1, 2].set_title(f'CPU Core Utilization ({used_cores}/{cpu_cores} cores)')
        axes[1, 2].set_xticks(range(cpu_cores))
        axes[1, 2].set_xticklabels([f'{i+1}' for i in range(cpu_cores)])
        axes[1, 2].set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig('parallel_evolution_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Parallel visualization results saved as 'parallel_evolution_results.png'")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        print("Skipping visualization step...")


def main():
    """主函数"""
    
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    print("🚀 Welcome to Parallel Prokaryotic Genome Evolution Simulator!")
    print("Choose simulation mode:")
    print("1. Run parallel evolution simulation")
    print("2. Run performance comparison (serial vs parallel)")
    print("3. Run both")
    
    try:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            # 运行并行模拟
            initial_genome, evolved_genome, analysis, snapshots = run_parallel_evolution_simulation(
                generations=200,           # 进化代数
                initial_gene_count=3000,   # 初始基因数
                mutation_rate=1e-5,        # 点突变率
                hgt_rate=0.02,            # HGT率
                recombination_rate=1e-3,   # 重组率
                num_processes=None,        # 使用所有可用CPU核心
                chunk_size=None           # 自动计算分块大小
            )
            
        elif choice == "2":
            # 运行性能对比
            performance_results = run_performance_comparison(
                generations=100,
                initial_gene_count=1000,
                mutation_rate=1e-5,
                hgt_rate=0.02,
                recombination_rate=1e-3
            )
            
        elif choice == "3":
            # 运行两者
            print("\n" + "="*50)
            print("PART 1: Performance Comparison")
            print("="*50)
            
            performance_results = run_performance_comparison(
                generations=50,
                initial_gene_count=500,
                mutation_rate=1e-5,
                hgt_rate=0.02,
                recombination_rate=1e-3
            )
            
            print("\n" + "="*50)
            print("PART 2: Full Parallel Simulation")
            print("="*50)
            
            initial_genome, evolved_genome, analysis, snapshots = run_parallel_evolution_simulation(
                generations=200,
                initial_gene_count=2000,
                mutation_rate=1e-5,
                hgt_rate=0.02,
                recombination_rate=1e-3
            )
            
        else:
            print("❌ Invalid choice. Please run the program again.")
            return
        
        print("\n🎉 All simulations completed successfully!")
        print("📊 Check the generated visualization files for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        print("Please check your input parameters and try again.")


if __name__ == "__main__":
    main()