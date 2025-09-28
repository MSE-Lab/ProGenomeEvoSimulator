#!/usr/bin/env python3
"""
Persistent Evolution Demo - 持久化进化演示脚本
展示持久化数据存储和分析功能

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from main_persistent import run_persistent_simulation, analyze_simulation_results
from analysis.persistent_data_analyzer import PersistentDataAnalyzer
from core.genome import create_initial_genome


def demo_basic_persistent_simulation():
    """演示基本的持久化模拟"""
    print("🎬 DEMO 1: Basic Persistent Simulation")
    print("=" * 60)
    
    # 运行快速测试配置
    run_directory = run_persistent_simulation(
        config_name='fast_test',
        output_dir='demo_results',
        run_analysis=True
    )
    
    print(f"\n✅ Demo 1 completed. Data saved to: {run_directory}")
    return run_directory


def demo_data_analysis(run_directory: str):
    """演示数据分析功能"""
    print(f"\n🎬 DEMO 2: Data Analysis and Visualization")
    print("=" * 60)
    
    # 创建数据分析器
    analyzer = PersistentDataAnalyzer(run_directory)
    
    # 加载和显示基本信息
    print("📋 Loading simulation metadata...")
    run_info = analyzer.load_run_info()
    config = analyzer.load_config()
    
    print(f"Run ID: {run_info.get('run_id', 'Unknown')}")
    print(f"Generations: {config.get('generations', 'Unknown')}")
    print(f"Initial genes: {config.get('initial_genes', 'Unknown')}")
    
    # 加载统计数据
    print("\n📊 Loading statistical data...")
    genome_stats = analyzer.load_genome_stats()
    evolution_stats = analyzer.load_evolution_stats()
    
    if not genome_stats.empty:
        print(f"Genome statistics records: {len(genome_stats)}")
        print(f"Final genome size: {genome_stats['total_size'].iloc[-1]:,} bp")
        print(f"Final gene count: {genome_stats['gene_count'].iloc[-1]:,}")
    
    if not evolution_stats.empty:
        print(f"Evolution statistics records: {len(evolution_stats)}")
        total_mutations = evolution_stats['mutations_this_gen'].sum()
        total_hgt = evolution_stats['hgt_events_this_gen'].sum()
        print(f"Total mutations: {total_mutations:,}")
        print(f"Total HGT events: {total_hgt:,}")
    
    # 加载快照数据
    print("\n📸 Loading snapshot data...")
    snapshots = analyzer.load_snapshots()
    print(f"Available snapshots: {len(snapshots)}")
    
    if snapshots:
        generations = sorted(snapshots.keys())
        print(f"Snapshot generations: {generations}")
        
        # 显示第一个和最后一个快照的信息
        first_gen = generations[0]
        last_gen = generations[-1]
        
        first_snapshot = snapshots[first_gen]
        last_snapshot = snapshots[last_gen]
        
        print(f"\nFirst snapshot (Gen {first_gen}):")
        print(f"  Genes: {len(first_snapshot['genes'])}")
        print(f"  Total mutations: {first_snapshot.get('total_mutations', 0)}")
        
        print(f"\nLast snapshot (Gen {last_gen}):")
        print(f"  Genes: {len(last_snapshot['genes'])}")
        print(f"  Total mutations: {last_snapshot.get('total_mutations', 0)}")
    
    # 进化分析
    print("\n🔬 Running evolution analysis...")
    genome_analysis = analyzer.analyze_genome_evolution()
    
    if genome_analysis:
        size_evolution = genome_analysis['size_evolution']
        gene_evolution = genome_analysis['gene_count_evolution']
        
        print(f"Genome size change: {size_evolution['size_change']:+,} bp ({size_evolution['size_change_percent']:+.2f}%)")
        print(f"Gene count change: {gene_evolution['gene_change']:+,} ({gene_evolution['gene_change_percent']:+.2f}%)")
    
    # 基因组比较
    print("\n🧬 Comparing initial vs final genome...")
    comparison = analyzer.compare_initial_vs_final_genome()
    
    if comparison:
        basic_stats = comparison['basic_stats']
        gene_origin = comparison['gene_origin_analysis']
        
        print(f"Gene count: {basic_stats['initial_genes']} → {basic_stats['final_genes']}")
        print(f"Core gene retention: {gene_origin['core_gene_retention']:.2%}")
        print(f"HGT genes acquired: {gene_origin['final_hgt_genes']}")
    
    print(f"\n✅ Demo 2 completed. Analysis data available in: {analyzer.analysis_dir}")


def demo_event_analysis(run_directory: str):
    """演示进化事件分析"""
    print(f"\n🎬 DEMO 3: Evolution Events Analysis")
    print("=" * 60)
    
    analyzer = PersistentDataAnalyzer(run_directory)
    
    # 加载不同类型的进化事件
    event_types = ['mutations', 'hgt_events', 'recombination', 'gene_loss']
    
    for event_type in event_types:
        print(f"\n📝 Loading {event_type} events...")
        events = analyzer.load_evolution_events(event_type)
        
        if events:
            print(f"Total {event_type}: {len(events)}")
            
            # 显示前几个事件的示例
            if len(events) > 0:
                print(f"Sample event:")
                sample_event = events[0]
                print(f"  Generation: {sample_event.get('generation', 'Unknown')}")
                print(f"  Timestamp: {sample_event.get('timestamp', 'Unknown')}")
                print(f"  Data: {sample_event.get('data', {})}")
        else:
            print(f"No {event_type} events found")
    
    print(f"\n✅ Demo 3 completed. Event logs available in: {analyzer.events_dir}")


def demo_comprehensive_analysis(run_directory: str):
    """演示综合分析功能"""
    print(f"\n🎬 DEMO 4: Comprehensive Analysis and Reporting")
    print("=" * 60)
    
    # 运行完整的分析和可视化
    analyze_simulation_results(
        run_directory=run_directory,
        generate_plots=True,
        generate_report=True
    )
    
    # 显示生成的文件
    analyzer = PersistentDataAnalyzer(run_directory)
    
    print(f"\n📁 Generated files:")
    print(f"📊 Visualizations: {analyzer.viz_dir}")
    viz_files = list(analyzer.viz_dir.glob('*.png'))
    for viz_file in viz_files:
        print(f"  - {viz_file.name}")
    
    print(f"\n📋 Analysis files: {analyzer.analysis_dir}")
    analysis_files = list(analyzer.analysis_dir.glob('*.json')) + list(analyzer.analysis_dir.glob('*.txt'))
    for analysis_file in analysis_files:
        print(f"  - {analysis_file.name}")
    
    print(f"\n✅ Demo 4 completed. All analysis files generated.")


def demo_data_export(run_directory: str):
    """演示数据导出功能"""
    print(f"\n🎬 DEMO 5: Data Export and Summary")
    print("=" * 60)
    
    analyzer = PersistentDataAnalyzer(run_directory)
    
    # 导出数据摘要
    print("📤 Exporting data summary...")
    summary = analyzer.export_data_summary()
    
    print(f"Summary keys: {list(summary.keys())}")
    
    if 'run_info' in summary:
        run_info = summary['run_info']
        print(f"Run ID: {run_info.get('run_id', 'Unknown')}")
        print(f"Engine version: {run_info.get('engine_version', 'Unknown')}")
    
    if 'data_files' in summary:
        data_files = summary['data_files']
        print(f"Data files summary:")
        for file_type, count in data_files.items():
            print(f"  {file_type}: {count} files")
    
    if 'genome_evolution' in summary:
        genome_evolution = summary['genome_evolution']
        if 'size_evolution' in genome_evolution:
            size_info = genome_evolution['size_evolution']
            print(f"Genome size evolution:")
            print(f"  Initial: {size_info.get('initial_size', 0):,} bp")
            print(f"  Final: {size_info.get('final_size', 0):,} bp")
            print(f"  Change: {size_info.get('size_change_percent', 0):+.2f}%")
    
    # 保存摘要到文件
    summary_file = analyzer.analysis_dir / "exported_summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 Summary exported to: {summary_file}")
    print(f"✅ Demo 5 completed.")


def main():
    """主演示函数"""
    print("🎬 PERSISTENT EVOLUTION SIMULATOR - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo will showcase all persistent data storage and analysis features.")
    print("=" * 80)
    
    try:
        # Demo 1: 运行基本的持久化模拟
        run_directory = demo_basic_persistent_simulation()
        
        # 等待一下让用户看到结果
        time.sleep(2)
        
        # Demo 2: 数据分析
        demo_data_analysis(run_directory)
        
        # Demo 3: 事件分析
        demo_event_analysis(run_directory)
        
        # Demo 4: 综合分析
        demo_comprehensive_analysis(run_directory)
        
        # Demo 5: 数据导出
        demo_data_export(run_directory)
        
        print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 All data and results are saved in: {run_directory}")
        print("📊 You can explore the following directories:")
        print(f"  - Snapshots: {Path(run_directory) / 'snapshots'}")
        print(f"  - Statistics: {Path(run_directory) / 'statistics'}")
        print(f"  - Events: {Path(run_directory) / 'events'}")
        print(f"  - Analysis: {Path(run_directory) / 'analysis'}")
        print(f"  - Visualizations: {Path(run_directory) / 'visualizations'}")
        print("=" * 80)
        
        # 显示如何继续分析
        print("\n🔍 To analyze this data later, you can use:")
        print(f"python main_persistent.py --analyze-only {run_directory}")
        print("\n📊 Or use the analyzer directly:")
        print(f"from analysis.persistent_data_analyzer import PersistentDataAnalyzer")
        print(f"analyzer = PersistentDataAnalyzer('{run_directory}')")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()