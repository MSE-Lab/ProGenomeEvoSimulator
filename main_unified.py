#!/usr/bin/env python3
"""
Unified Prokaryotic Genome Evolution Simulator
Main program using unified evolution engine
Integrates all features: optimized algorithms, parallel processing, gene loss, etc.

Version: 1.0.0
Author: Xiao-Yang Zhi
Date: 2025-09-27
"""

__version__ = "1.0.0"
__author__ = "XYZ"
__date__ = "2025-09-27"

import time
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.genome import create_initial_genome
    from core.unified_evolution_engine import UnifiedEvolutionEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)

# Import unified visualization system
try:
    from core.visualization import EvolutionVisualizer, create_comprehensive_visualization
    visualization_available = True
    # 只在主进程中打印加载信息，避免并行进程重复输出
    import multiprocessing as mp
    try:
        if mp.current_process().name == 'MainProcess':
            print("📊 Unified visualization system loaded (server-friendly)")
    except:
        pass
except ImportError:
    visualization_available = False
    create_comprehensive_visualization = None
    print("📊 Visualization system not available. Charts will be skipped.")


def create_test_configurations() -> Dict[str, Dict[str, any]]:
    """创建不同的测试配置"""
    
    configs = {
        'fast_test': {
            'name': '快速测试',
            'description': '适度提高参数值，快速观察进化效果',
            'params': {
                'mutation_rate': 1e-5,  # 修正：降低到更合理的水平
                'hgt_rate': 1e-4,       # 修正：大幅降低HGT率
                'recombination_rate': 1e-5,  # 修正：降低重组率
                'loss_rate': 1e-6,      # 修正：降低丢失率
                'enable_gene_loss': True,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'genome': {'gene_count': 500, 'avg_gene_length': 1000},
            'generations': 100
        },
        
        'realistic': {
            'name': '真实参数',
            'description': '基于文献的真实原核生物进化参数',
            'params': {
                'mutation_rate': 1e-9,   # 修正：每bp每代的真实突变率
                'hgt_rate': 1e-6,        # 修正：更真实的HGT率
                'recombination_rate': 1e-8,  # 修正：更真实的重组率
                'loss_rate': 1e-8,       # 修正：更真实的丢失率
                'enable_gene_loss': True,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'genome': {'gene_count': 3000, 'avg_gene_length': 800},
            'generations': 1000
        },
        
        'large_scale': {
            'name': '大规模模拟',
            'description': '大基因组，长时间进化（生物学合理参数）',
            'params': {
                'mutation_rate': 1e-8,   # 修正：适中的突变率
                'hgt_rate': 1e-5,        # 修正：大幅降低HGT率
                'recombination_rate': 1e-7,  # 修正：降低重组率
                'loss_rate': 1e-7,       # 修正：适中的丢失率
                'enable_gene_loss': True,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'genome': {'gene_count': 5000, 'avg_gene_length': 600},
            'generations': 500
        },
        
        'no_gene_loss': {
            'name': '无基因丢失对照',
            'description': '关闭基因丢失功能的对照实验',
            'params': {
                'mutation_rate': 1e-6,   # 修正：降低突变率
                'hgt_rate': 1e-5,        # 修正：大幅降低HGT率
                'recombination_rate': 1e-6,  # 修正：降低重组率
                'enable_gene_loss': False,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'genome': {'gene_count': 2000, 'avg_gene_length': 500},
            'generations': 100
        },
        
        'serial_only': {
            'name': '串行处理',
            'description': '关闭并行处理的性能对比',
            'params': {
                'mutation_rate': 1e-6,   # 修正：降低突变率
                'hgt_rate': 1e-5,        # 修正：大幅降低HGT率
                'recombination_rate': 1e-6,  # 修正：降低重组率
                'loss_rate': 1e-7,       # 修正：降低丢失率
                'enable_gene_loss': True,
                'enable_parallel': False,
                'enable_optimization': True
            },
            'genome': {'gene_count': 1500, 'avg_gene_length': 400},
            'generations': 50
        }
    }
    
    return configs


def run_single_simulation(config_name: str, config: Dict) -> Tuple[Dict, float]:
    """运行单个模拟配置"""
    
    print(f"\n🚀 Running simulation: {config['name']}")
    print(f"📝 Description: {config['description']}")
    print("=" * 80)
    
    # 创建初始基因组
    np.random.seed(42)  # 确保可重复性
    initial_genome = create_initial_genome(
        gene_count=config['genome']['gene_count'],
        avg_gene_length=config['genome']['avg_gene_length'],
        min_gene_length=200
    )
    
    print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    
    # 创建进化引擎
    engine = UnifiedEvolutionEngine(**config['params'])
    
    # 运行模拟
    start_time = time.time()
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=initial_genome,
        generations=config['generations'],
        save_snapshots=True,
        snapshot_interval=max(1, config['generations'] // 10)
    )
    simulation_time = time.time() - start_time
    
    # 收集结果
    results = {
        'config_name': config_name,
        'config': config,
        'initial_genome': initial_genome,  # 传递完整的基因组对象用于ANI分析
        'final_genome': final_genome,      # 传递完整的基因组对象用于ANI分析
        'initial_genome_stats': initial_genome.get_statistics(),  # 保留统计信息
        'final_genome_stats': final_genome.get_statistics(),      # 保留统计信息
        'snapshots': snapshots,
        'simulation_time': simulation_time,
        'performance_analysis': engine.get_performance_analysis(),
        'evolution_summary': engine.get_evolution_summary(final_genome)
    }
    
    # 显示结果摘要
    print(f"\n📈 SIMULATION RESULTS SUMMARY")
    print(f"⏱️  Total time: {simulation_time/60:.2f} minutes")
    print(f"🧬 Genome changes:")
    print(f"   Size: {initial_genome.size:,} → {final_genome.size:,} bp ({final_genome.size - initial_genome.size:+,})")
    print(f"   Genes: {initial_genome.gene_count:,} → {final_genome.gene_count:,} ({final_genome.gene_count - initial_genome.gene_count:+,})")
    print(f"🔬 Evolution events:")
    print(f"   Mutations: {final_genome.total_mutations:,}")
    print(f"   HGT events: {final_genome.total_hgt_events:,}")
    print(f"   Recombinations: {final_genome.total_recombination_events:,}")
    
    if config['params'].get('enable_gene_loss', False):
        try:
            if hasattr(engine, 'gene_loss') and engine.gene_loss is not None:
                loss_stats = engine.gene_loss.get_loss_statistics(final_genome)
                print(f"   Genes lost: {loss_stats.get('total_genes_lost', 0):,}")
            else:
                print(f"   Genes lost: N/A (gene loss engine not available)")
        except Exception as e:
            print(f"   Genes lost: Error retrieving statistics ({e})")
    
    return results, simulation_time


def compare_configurations(config_names: List[str], configs: Dict) -> Dict:
    """比较多个配置的性能"""
    
    print(f"\n🔬 CONFIGURATION COMPARISON")
    print(f"Comparing {len(config_names)} configurations...")
    print("=" * 80)
    
    comparison_results = {}
    
    for config_name in config_names:
        if config_name not in configs:
            print(f"❌ Configuration '{config_name}' not found!")
            continue
        
        try:
            results, sim_time = run_single_simulation(config_name, configs[config_name])
            comparison_results[config_name] = results
            
        except Exception as e:
            print(f"❌ Error in configuration '{config_name}': {e}")
            import traceback
            traceback.print_exc()
    
    # 生成比较报告
    if len(comparison_results) > 1:
        print(f"\n📊 COMPARISON REPORT")
        print("=" * 80)
        
        print(f"{'Configuration':<20} {'Time(min)':<10} {'Genes':<10} {'Mutations':<12} {'HGT':<8} {'Lost':<8}")
        print("-" * 80)
        
        for config_name, results in comparison_results.items():
            final_stats = results['final_genome']
            sim_time = results['simulation_time']
            
            # 基因丢失数量
            if results['config']['params'].get('enable_gene_loss', False):
                loss_count = results['evolution_summary'].get('gene_loss_stats', {}).get('total_genes_lost', 0)
            else:
                loss_count = 0
            
            print(f"{config_name:<20} {sim_time/60:<10.2f} {final_stats['gene_count']:<10,} "
                  f"{final_stats['total_mutations']:<12,} {final_stats['total_hgt_events']:<8,} {loss_count:<8,}")
    
    return comparison_results


def create_visualization(results: Dict, performance_data: Dict = None):
    """创建可视化图表 - 使用统一可视化系统"""
    if not visualization_available:
        print("📊 Visualization system not available. Skipping charts.")
        return
    
    try:
        print("📊 Creating server-friendly visualizations...")
        
        # 使用统一可视化系统
        saved_files = create_comprehensive_visualization(
            results=results,
            performance_data=performance_data,
            output_dir='simulation_results'
        )
        
        if saved_files:
            print(f"✅ Generated {len(saved_files)} visualization files:")
            for i, filepath in enumerate(saved_files, 1):
                print(f"   {i}. {filepath}")
            
            print("💡 Features:")
            print("   🖥️  Server-compatible (no GUI required)")
            print("   📊 High-resolution output (300 DPI)")
            print("   📁 Organized in simulation_results/ directory")
        else:
            print("⚠️  No visualization files were generated")
            
    except Exception as e:
        print(f"📊 Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def interactive_menu():
    """交互式菜单"""
    
    configs = create_test_configurations()
    
    while True:
        print(f"\n🧬 UNIFIED PROKARYOTIC GENOME EVOLUTION SIMULATOR")
        print("=" * 60)
        print("Available options:")
        print("1. Run single simulation")
        print("2. Compare multiple configurations")
        print("3. Run all configurations")
        print("4. Custom configuration")
        print("5. Performance benchmark")
        print("6. Exit")
        print("=" * 60)
        
        try:
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                # 单个模拟
                print("\nAvailable configurations:")
                for i, (name, config) in enumerate(configs.items(), 1):
                    print(f"{i}. {config['name']} - {config['description']}")
                
                try:
                    config_idx = int(input(f"Select configuration (1-{len(configs)}): ")) - 1
                    config_names = list(configs.keys())
                    
                    if 0 <= config_idx < len(config_names):
                        config_name = config_names[config_idx]
                        results, _ = run_single_simulation(config_name, configs[config_name])
                        
                        # 询问是否创建可视化
                        if input("\nCreate visualization? (y/n): ").lower().startswith('y'):
                            create_visualization(results)
                    else:
                        print("❌ Invalid selection!")
                        
                except ValueError:
                    print("❌ Please enter a valid number!")
            
            elif choice == '2':
                # 比较多个配置
                print("\nSelect configurations to compare:")
                config_list = list(configs.keys())
                for i, name in enumerate(config_list, 1):
                    print(f"{i}. {configs[name]['name']}")
                
                try:
                    selections = input("Enter numbers separated by commas (e.g., 1,2,3): ").strip()
                    indices = [int(x.strip()) - 1 for x in selections.split(',')]
                    selected_configs = [config_list[i] for i in indices if 0 <= i < len(config_list)]
                    
                    if len(selected_configs) >= 2:
                        comparison_results = compare_configurations(selected_configs, configs)
                    else:
                        print("❌ Please select at least 2 configurations!")
                        
                except (ValueError, IndexError):
                    print("❌ Invalid selection format!")
            
            elif choice == '3':
                # 运行所有配置
                print("\n🚀 Running all configurations...")
                all_configs = list(configs.keys())
                comparison_results = compare_configurations(all_configs, configs)
            
            elif choice == '4':
                # 自定义配置
                print("\n⚙️ Custom Configuration Builder")
                print("Enter parameters (press Enter for default values):")
                
                try:
                    custom_config = {
                        'name': 'Custom Configuration',
                        'description': 'User-defined parameters',
                        'params': {},
                        'genome': {},
                        'generations': 100
                    }
                    
                    # Basic parameters
                    mutation_rate = input("Mutation rate (default: 1e-5): ").strip()
                    custom_config['params']['mutation_rate'] = float(mutation_rate) if mutation_rate else 1e-5
                    
                    hgt_rate = input("HGT rate (default: 1e-5): ").strip()
                    custom_config['params']['hgt_rate'] = float(hgt_rate) if hgt_rate else 1e-5
                    
                    recombination_rate = input("Recombination rate (default: 1e-3): ").strip()
                    custom_config['params']['recombination_rate'] = float(recombination_rate) if recombination_rate else 1e-3
                    
                    # Gene loss
                    enable_loss = input("Enable gene loss? (y/n, default: y): ").strip().lower()
                    custom_config['params']['enable_gene_loss'] = not enable_loss.startswith('n')
                    
                    if custom_config['params']['enable_gene_loss']:
                        loss_rate = input("Gene loss rate (default: 1e-5): ").strip()
                        custom_config['params']['loss_rate'] = float(loss_rate) if loss_rate else 1e-5
                    
                    # Parallel processing
                    enable_parallel = input("Enable parallel processing? (y/n, default: y): ").strip().lower()
                    custom_config['params']['enable_parallel'] = not enable_parallel.startswith('n')
                    
                    # Genome parameters
                    gene_count = input("Initial gene count (default: 2000): ").strip()
                    custom_config['genome']['gene_count'] = int(gene_count) if gene_count else 2000
                    
                    avg_length = input("Average gene length (default: 500): ").strip()
                    custom_config['genome']['avg_gene_length'] = int(avg_length) if avg_length else 500
                    
                    generations = input("Number of generations (default: 100): ").strip()
                    custom_config['generations'] = int(generations) if generations else 100
                    
                    # Set default values
                    custom_config['params']['enable_optimization'] = True
                    
                    # 运行自定义配置
                    results, _ = run_single_simulation('custom', custom_config)
                    
                    if input("\nCreate visualization? (y/n): ").lower().startswith('y'):
                        create_visualization(results)
                        
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
            
            elif choice == '5':
                # 性能基准测试
                print("\n⚡ Performance Benchmark")
                print("Testing parallel vs serial performance...")
                
                benchmark_configs = {
                    'parallel': configs['fast_test'].copy(),
                    'serial': configs['serial_only'].copy()
                }
                
                # 确保相同的基因组大小
                benchmark_configs['serial']['genome'] = benchmark_configs['parallel']['genome'].copy()
                benchmark_configs['serial']['generations'] = benchmark_configs['parallel']['generations']
                
                comparison_results = compare_configurations(['parallel', 'serial'], benchmark_configs)
                
                if len(comparison_results) == 2:
                    parallel_time = comparison_results['parallel']['simulation_time']
                    serial_time = comparison_results['serial']['simulation_time']
                    speedup = serial_time / parallel_time if parallel_time > 0 else 0
                    
                    print(f"\n🏆 BENCHMARK RESULTS:")
                    print(f"   Parallel time: {parallel_time/60:.2f} minutes")
                    print(f"   Serial time: {serial_time/60:.2f} minutes")
                    print(f"   Speedup: {speedup:.2f}x")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option! Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    
    print("🧬 UNIFIED PROKARYOTIC GENOME EVOLUTION SIMULATOR")
    print("=" * 60)
    print("Features:")
    print("✅ Optimized algorithms")
    print("✅ Parallel processing")
    print("✅ Gene loss simulation")
    print("✅ Comprehensive analysis")
    print("✅ Interactive interface")
    print("=" * 60)
    
    try:
        interactive_menu()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()