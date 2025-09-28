#!/usr/bin/env python3
"""
Main Persistent Simulation Script - 持久化模拟主程序
使用持久化进化引擎运行原核生物基因组进化模拟

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from core.genome import Genome, create_initial_genome
from core.persistent_evolution_engine import PersistentEvolutionEngine
from analysis.conservation_analyzer import ConservationAnalyzer
from analysis.persistent_data_analyzer import PersistentDataAnalyzer


def create_test_configurations() -> Dict[str, Dict[str, Any]]:
    """创建测试配置"""
    return {
        'fast_test': {
            'description': '快速测试配置 - 适合验证功能（重组修复版）',
            'generations': 500,  # 减少到50代以便快速测试
            'initial_genes': 500,
            'snapshot_interval': 10,
            'engine_config': {
                'mutation_rate': 1e-8,
                'hgt_rate': 1e-2,  # 修正：更符合生物学实际的HGT频率
                'recombination_rate': 1e-6,  # 适中的重组率
                'mutations_per_recombination_event': (5, 15),  # 每次重组的突变数量
                'recombination_debug': True,  # 启用重组调试
                'enable_gene_loss': True,
                'loss_rate': 1e-2,
                'core_gene_protection': 0.98,
                'hgt_gene_loss_multiplier': 20.0,
                'min_genome_size': 1200,
                'min_core_genes': 1000,
                'optimal_genome_size': 3000,
                'enable_parallel': False,  # 关闭并行以确保重组正常工作
                'enable_optimization': True
            },
            'storage_config': {
                'compress_data': False,
                'save_detailed_events': True,
                'save_sequences': False,
                'stats_flush_interval': 10
            }
        },
        
        'realistic': {
            'description': '真实参数配置 - 基于文献的生物学参数',
            'generations': 1000,
            'initial_genes': 3000,
            'snapshot_interval': 100,
            'engine_config': {
                'mutation_rate': 1e-9,
                'hgt_rate': 1e-6,
                'recombination_rate': 1e-8,
                'min_similarity_for_recombination': 0.85,
                'enable_gene_loss': True,
                'loss_rate': 1e-8,
                'core_gene_protection': 0.98,
                'hgt_gene_loss_multiplier': 20.0,
                'min_genome_size': 2000,
                'min_core_genes': 1500,
                'optimal_genome_size': 4000,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'storage_config': {
                'compress_data': True,
                'save_detailed_events': True,
                'save_sequences': True,
                'stats_flush_interval': 20
            }
        },
        
        'large_scale': {
            'description': '大规模模拟配置 - 长期进化研究',
            'generations': 5000,
            'initial_genes': 5000,
            'snapshot_interval': 200,
            'engine_config': {
                'mutation_rate': 1e-8,
                'hgt_rate': 1e-5,
                'recombination_rate': 1e-7,
                'min_similarity_for_recombination': 0.85,
                'enable_gene_loss': True,
                'loss_rate': 1e-7,
                'core_gene_protection': 0.98,
                'hgt_gene_loss_multiplier': 20.0,
                'min_genome_size': 3000,
                'min_core_genes': 2000,
                'optimal_genome_size': 6000,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'storage_config': {
                'compress_data': True,
                'save_detailed_events': False,  # 节省存储空间
                'save_sequences': False,        # 节省存储空间
                'stats_flush_interval': 50
            }
        },
        
        'detailed_analysis': {
            'description': '详细分析配置 - 保存所有数据用于深度分析',
            'generations': 500,
            'initial_genes': 2000,
            'snapshot_interval': 25,
            'engine_config': {
                'mutation_rate': 1e-6,
                'hgt_rate': 1e-5,
                'recombination_rate': 1e-7,
                'min_similarity_for_recombination': 0.85,
                'enable_gene_loss': True,
                'loss_rate': 1e-7,
                'core_gene_protection': 0.98,
                'hgt_gene_loss_multiplier': 20.0,
                'min_genome_size': 1500,
                'min_core_genes': 1200,
                'optimal_genome_size': 3500,
                'enable_parallel': True,
                'enable_optimization': True
            },
            'storage_config': {
                'compress_data': False,  # 不压缩便于分析
                'save_detailed_events': True,
                'save_sequences': True,
                'stats_flush_interval': 5
            }
        }
    }


def run_persistent_simulation(config_name: str = 'fast_test', 
                            custom_config: Dict[str, Any] = None,
                            output_dir: str = "simulation_results",
                            run_analysis: bool = True) -> str:
    """
    运行持久化模拟
    
    Args:
        config_name: 配置名称
        custom_config: 自定义配置（覆盖预设配置）
        output_dir: 输出目录
        run_analysis: 是否运行后续分析
    
    Returns:
        运行目录路径
    """
    
    # 获取配置
    configs = create_test_configurations()
    
    if custom_config:
        config = custom_config
    elif config_name in configs:
        config = configs[config_name]
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(configs.keys())}")
    
    print("🗄️  PERSISTENT PROKARYOTIC GENOME EVOLUTION SIMULATION")
    print("=" * 80)
    print(f"📋 Configuration: {config_name}")
    print(f"📝 Description: {config['description']}")
    print(f"🎯 Generations: {config['generations']:,}")
    print(f"🧬 Initial genes: {config['initial_genes']:,}")
    print(f"📸 Snapshot interval: {config['snapshot_interval']}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)
    
    # 创建初始基因组
    print("🧬 Creating initial genome...")
    initial_genome = create_initial_genome(
        gene_count=config['initial_genes'],
        avg_gene_length=1000,
        min_gene_length=150,
        use_biological_sequences=True
    )
    print(f"✅ Initial genome created: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    
    # 创建持久化进化引擎
    print("🔧 Initializing persistent evolution engine...")
    
    # 过滤引擎配置参数，只保留PersistentEvolutionEngine支持的参数
    engine_config = config['engine_config'].copy()
    
    # 移除不支持的参数
    unsupported_params = ['mutations_per_recombination_event', 'recombination_debug']
    for param in unsupported_params:
        engine_config.pop(param, None)
    
    engine = PersistentEvolutionEngine(
        base_output_dir=output_dir,
        snapshot_interval=config['snapshot_interval'],
        **engine_config,
        **config['storage_config']
    )
    
    # 运行模拟
    print("🚀 Starting persistent simulation...")
    start_time = time.time()
    
    try:
        final_genome, snapshots = engine.simulate_evolution(
            initial_genome=initial_genome,
            generations=config['generations'],
            save_snapshots=True,
            snapshot_interval=config['snapshot_interval']
        )
        
        simulation_time = time.time() - start_time
        
        print(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {simulation_time:.2f} seconds")
        print(f"📊 Final genome: {final_genome.gene_count:,} genes, {final_genome.size:,} bp")
        print(f"🧬 Evolution events: {final_genome.total_mutations:,} mutations, {final_genome.total_hgt_events:,} HGT events")
        
        # 获取运行目录
        run_directory = str(engine.get_run_directory())
        
        # 运行保守性分析
        if run_analysis:
            print("\n🔬 Running conservation analysis...")
            try:
                analyzer = ConservationAnalyzer()
                conservation_results = analyzer.analyze_conservation(
                    initial_genome, final_genome, 
                    snapshots=snapshots if snapshots else []
                )
                
                # 保存保守性分析结果
                engine.save_conservation_analysis(conservation_results)
                print("✅ Conservation analysis completed")
                
            except Exception as e:
                print(f"⚠️  Conservation analysis failed: {e}")
            
            # 计算ANI身份数据 - 使用PersistentDataAnalyzer
            try:
                print("🧮 Calculating ANI identities...")
                run_dir = engine.get_run_directory()
                data_analyzer = PersistentDataAnalyzer(str(run_dir))
                
                # 准备基因组列表进行ANI分析
                genomes_for_ani = [initial_genome, final_genome]
                if snapshots:
                    # 添加一些快照基因组用于ANI分析
                    genomes_for_ani.extend(snapshots[-3:])  # 添加最后3个快照
                
                ani_data = data_analyzer.calculate_ani_matrix(genomes_for_ani)
                print("✅ ANI analysis completed")
                
            except Exception as e:
                print(f"⚠️  ANI analysis failed: {e}")
        
        # 清理资源
        engine.cleanup_parallel_resources()
        
        return run_directory
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        engine.cleanup_parallel_resources()
        raise


def analyze_simulation_results(run_directory: str, 
                             generate_plots: bool = True,
                             generate_report: bool = True) -> None:
    """
    分析模拟结果
    
    Args:
        run_directory: 运行目录路径
        generate_plots: 是否生成图表
        generate_report: 是否生成报告
    """
    print(f"\n📊 ANALYZING SIMULATION RESULTS")
    print("=" * 80)
    print(f"📁 Data directory: {run_directory}")
    
    try:
        # 创建数据分析器
        analyzer = PersistentDataAnalyzer(run_directory)
        
        # 生成可视化图表
        if generate_plots:
            print("📈 Generating visualization plots...")
            analyzer.plot_genome_evolution_timeline(save_plot=True)
            analyzer.plot_evolution_events_analysis(save_plot=True)
            print("✅ Plots generated and saved")
        
        # 生成综合报告
        if generate_report:
            print("📋 Generating comprehensive analysis report...")
            report = analyzer.generate_comprehensive_report()
            print("✅ Report generated and saved")
            
            # 显示报告摘要
            print("\n📋 ANALYSIS SUMMARY")
            print("-" * 40)
            genome_analysis = analyzer.analyze_genome_evolution()
            if genome_analysis:
                size_change = genome_analysis['size_evolution']['size_change_percent']
                gene_change = genome_analysis['gene_count_evolution']['gene_change_percent']
                total_mutations = genome_analysis['mutation_accumulation']['total_mutations']
                hgt_percentage = genome_analysis['hgt_evolution']['hgt_gene_percentage']
                
                print(f"Genome size change: {size_change:+.2f}%")
                print(f"Gene count change: {gene_change:+.2f}%")
                print(f"Total mutations: {total_mutations:,}")
                print(f"HGT genes: {hgt_percentage:.2f}%")
        
        print("✅ Analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Persistent Prokaryotic Genome Evolution Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available configurations:
  fast_test        - Quick test (200 generations, 1500 genes)
  realistic        - Realistic parameters (1000 generations, 3000 genes)  
  large_scale      - Large scale simulation (5000 generations, 5000 genes)
  detailed_analysis - Detailed analysis (500 generations, full data)

Examples:
  python main_persistent.py --config fast_test
  python main_persistent.py --config realistic --output-dir my_results
  python main_persistent.py --analyze-only simulation_results/run_20250927_140000
        """
    )
    
    parser.add_argument('--config', '-c', 
                       choices=['fast_test', 'realistic', 'large_scale', 'detailed_analysis'],
                       default='fast_test',
                       help='Simulation configuration to use')
    
    parser.add_argument('--output-dir', '-o',
                       default='simulation_results',
                       help='Output directory for simulation data')
    
    parser.add_argument('--no-analysis', 
                       action='store_true',
                       help='Skip post-simulation analysis')
    
    parser.add_argument('--analyze-only',
                       help='Only analyze existing results (provide run directory)')
    
    parser.add_argument('--no-plots',
                       action='store_true', 
                       help='Skip plot generation during analysis')
    
    parser.add_argument('--no-report',
                       action='store_true',
                       help='Skip report generation during analysis')
    
    args = parser.parse_args()
    
    try:
        if args.analyze_only:
            # 仅分析现有结果
            analyze_simulation_results(
                run_directory=args.analyze_only,
                generate_plots=not args.no_plots,
                generate_report=not args.no_report
            )
        else:
            # 运行新的模拟
            run_directory = run_persistent_simulation(
                config_name=args.config,
                output_dir=args.output_dir,
                run_analysis=not args.no_analysis
            )
            
            # 运行后续分析
            if not args.no_analysis:
                analyze_simulation_results(
                    run_directory=run_directory,
                    generate_plots=not args.no_plots,
                    generate_report=not args.no_report
                )
            
            print(f"\n🎉 ALL TASKS COMPLETED!")
            print(f"📁 Results saved to: {run_directory}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()