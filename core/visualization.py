#!/usr/bin/env python3
"""
统一可视化模块 - 服务器友好的图表生成
支持无图形界面环境，自动保存图表到文件
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# 配置matplotlib为非交互式后端（服务器友好）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 导入ANI和保守性分析模块
try:
    from analysis.ani_calculator import ANICalculator
    from analysis.conservation_analyzer import ConservationAnalyzer
    ANI_AVAILABLE = True
except ImportError:
    ANI_AVAILABLE = False
    print("⚠️  ANI analysis modules not available")

__version__ = "1.0.0"
__author__ = "ProGenomeEvoSimulator Team"

# 全局配置
VISUALIZATION_CONFIG = {
    'dpi': 300,
    'figsize_small': (12, 8),
    'figsize_medium': (15, 10),
    'figsize_large': (18, 12),
    'color_palette': {
        'genome_size': '#1f77b4',
        'gene_count': '#ff7f0e', 
        'mutations': '#2ca02c',
        'hgt_events': '#d62728',
        'recombinations': '#9467bd',
        'gene_loss': '#8c564b',
        'core_genes': '#e377c2',
        'hgt_genes': '#7f7f7f'
    },
    'output_dir': 'results',
    'auto_save': True,
    'show_plots': False  # 默认不显示，只保存
}


class EvolutionVisualizer:
    """进化模拟结果可视化器"""
    
    def __init__(self, output_dir: str = None, auto_save: bool = True):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            auto_save: 是否自动保存图表
        """
        self.output_dir = output_dir or VISUALIZATION_CONFIG['output_dir']
        self.auto_save = auto_save
        self.colors = VISUALIZATION_CONFIG['color_palette']
        
        # 确保输出目录存在
        if self.auto_save:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def create_evolution_summary(self, results: Dict, filename: str = None) -> str:
        """
        创建进化模拟总结图表
        
        Args:
            results: 模拟结果字典
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_summary_{timestamp}.png"
        
        # 提取数据
        snapshots = results.get('snapshots', [])
        if not snapshots:
            print("⚠️  No snapshots available for visualization")
            return None
        
        # 兼容不同的代数键名
        generations = [s.get('generation', s.get('snapshot_generation', 0)) for s in snapshots]
        genome_sizes = [s['genome_stats']['total_size'] for s in snapshots]
        gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
        mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
        hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
        recombinations = [s['genome_stats']['total_recombination_events'] for s in snapshots]
        
        # 创建图表
        fig = plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle('Prokaryotic Genome Evolution Simulation Results', 
                    fontsize=16, fontweight='bold')
        
        # 1. 基因组大小和基因数量
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(generations, genome_sizes, 
                        color=self.colors['genome_size'], 
                        linewidth=2, marker='o', markersize=4,
                        label='Genome Size (bp)')
        line2 = ax1_twin.plot(generations, gene_counts, 
                             color=self.colors['gene_count'], 
                             linewidth=2, marker='s', markersize=4,
                             label='Gene Count')
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Genome Size (bp)', color=self.colors['genome_size'])
        ax1_twin.set_ylabel('Gene Count', color=self.colors['gene_count'])
        ax1.set_title('Genome Size & Gene Count Evolution')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 2. 进化事件统计
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(generations, mutations, 
                color=self.colors['mutations'], 
                linewidth=2, marker='o', markersize=4,
                label='Mutations')
        ax2.plot(generations, hgt_events, 
                color=self.colors['hgt_events'], 
                linewidth=2, marker='s', markersize=4,
                label='HGT Events')
        ax2.plot(generations, recombinations, 
                color=self.colors['recombinations'], 
                linewidth=2, marker='^', markersize=4,
                label='Recombinations')
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Cumulative Events')
        ax2.set_title('Evolution Events Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 基因类型分布
        ax3 = fig.add_subplot(gs[1, 0])
        if 'core_genes' in snapshots[0]['genome_stats']:
            core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
            hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
            
            ax3.plot(generations, core_genes, 
                    color=self.colors['core_genes'], 
                    linewidth=2, marker='o', markersize=4,
                    label='Core Genes')
            ax3.plot(generations, hgt_genes, 
                    color=self.colors['hgt_genes'], 
                    linewidth=2, marker='s', markersize=4,
                    label='HGT Genes')
            
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Gene Count')
            ax3.set_title('Gene Type Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Gene type data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, alpha=0.7)
            ax3.set_title('Gene Type Distribution')
        
        # 4. 进化速率分析
        ax4 = fig.add_subplot(gs[1, 1])
        if len(generations) > 1:
            # 计算变化率
            size_changes = np.diff(genome_sizes)
            gene_changes = np.diff(gene_counts)
            gen_intervals = generations[1:]
            
            ax4.plot(gen_intervals, size_changes, 
                    color=self.colors['genome_size'], 
                    linewidth=2, marker='o', markersize=4,
                    label='Size Change/Gen')
            
            ax4_twin = ax4.twinx()
            ax4_twin.plot(gen_intervals, gene_changes, 
                         color=self.colors['gene_count'], 
                         linewidth=2, marker='s', markersize=4,
                         label='Gene Change/Gen')
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Size Change (bp)', color=self.colors['genome_size'])
            ax4_twin.set_ylabel('Gene Change', color=self.colors['gene_count'])
            ax4.set_title('Evolution Rate Analysis')
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor rate analysis', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, alpha=0.7)
            ax4.set_title('Evolution Rate Analysis')
        
        # 保存图表
        if self.auto_save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=VISUALIZATION_CONFIG['dpi'], 
                       bbox_inches='tight', facecolor='white')
            print(f"📊 Evolution summary saved: {filepath}")
        
        # 不显示图表（服务器友好）
        plt.close(fig)
        
        return filepath if self.auto_save else None
    
    def create_gene_loss_analysis(self, results: Dict, filename: str = None) -> str:
        """
        创建基因丢失分析图表
        
        Args:
            results: 包含基因丢失数据的结果
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gene_loss_analysis_{timestamp}.png"
        
        snapshots = results.get('snapshots', [])
        if not snapshots:
            print("⚠️  No snapshots available for gene loss analysis")
            return None
        
        # 提取基因丢失相关数据
        # 兼容不同的代数键名
        generations = [s.get('generation', s.get('snapshot_generation', 0)) for s in snapshots]
        gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figsize_medium'])
        fig.suptitle('Gene Loss Mechanism Analysis', fontsize=16, fontweight='bold')
        
        # 1. 基因数量变化
        axes[0, 0].plot(generations, gene_counts, 
                       color=self.colors['gene_count'], 
                       linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Gene Count')
        axes[0, 0].set_title('Gene Count Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 基因丢失率
        if len(generations) > 1:
            gene_changes = np.diff(gene_counts)
            loss_events = [-x for x in gene_changes if x < 0]  # 只统计丢失事件
            
            if loss_events:
                axes[0, 1].hist(loss_events, bins=max(5, len(loss_events)//2), 
                               color=self.colors['gene_loss'], alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Genes Lost per Generation')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Gene Loss Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No gene loss\nevents detected', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Gene Loss Distribution')
        
        # 3. 核心基因 vs HGT基因丢失
        if 'core_genes' in snapshots[0]['genome_stats']:
            core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
            hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
            
            axes[1, 0].plot(generations, core_genes, 
                           color=self.colors['core_genes'], 
                           linewidth=2, marker='o', markersize=4,
                           label='Core Genes')
            axes[1, 0].plot(generations, hgt_genes, 
                           color=self.colors['hgt_genes'], 
                           linewidth=2, marker='s', markersize=4,
                           label='HGT Genes')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Gene Count')
            axes[1, 0].set_title('Core vs HGT Gene Evolution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 净基因变化
        if len(generations) > 1:
            net_changes = np.diff(gene_counts)
            axes[1, 1].plot(generations[1:], net_changes, 
                           color='darkgreen', linewidth=2, 
                           marker='o', markersize=4)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Net Gene Change')
            axes[1, 1].set_title('Net Gene Change per Generation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if self.auto_save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=VISUALIZATION_CONFIG['dpi'], 
                       bbox_inches='tight', facecolor='white')
            print(f"📊 Gene loss analysis saved: {filepath}")
        
        plt.close(fig)
        return filepath if self.auto_save else None
    
    def create_performance_analysis(self, performance_data: Dict, filename: str = None) -> str:
        """
        创建性能分析图表
        
        Args:
            performance_data: 性能数据
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_analysis_{timestamp}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figsize_medium'])
        fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 执行时间分析
        if 'execution_times' in performance_data:
            times = performance_data['execution_times']
            mechanisms = list(times.keys())
            values = list(times.values())
            
            axes[0, 0].bar(mechanisms, values, color='skyblue', edgecolor='black')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_title('Execution Time by Mechanism')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 并行效率
        if 'parallel_efficiency' in performance_data:
            eff_data = performance_data['parallel_efficiency']
            if isinstance(eff_data, dict) and 'history' in eff_data:
                generations = list(range(len(eff_data['history'])))
                efficiencies = eff_data['history']
                
                axes[0, 1].plot(generations, efficiencies, 
                               color='green', linewidth=2, marker='o', markersize=4)
                axes[0, 1].set_xlabel('Generation')
                axes[0, 1].set_ylabel('Efficiency (%)')
                axes[0, 1].set_title('Parallel Processing Efficiency')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 内存使用
        if 'memory_usage' in performance_data:
            mem_data = performance_data['memory_usage']
            if isinstance(mem_data, dict):
                components = list(mem_data.keys())
                usage = list(mem_data.values())
                
                axes[1, 0].pie(usage, labels=components, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Memory Usage Distribution')
        
        # 4. 吞吐量分析
        if 'throughput' in performance_data:
            throughput = performance_data['throughput']
            if isinstance(throughput, dict) and 'generations_per_second' in throughput:
                gps = throughput['generations_per_second']
                axes[1, 1].bar(['Throughput'], [gps], color='orange', edgecolor='black')
                axes[1, 1].set_ylabel('Generations/Second')
                axes[1, 1].set_title('Processing Throughput')
        
        plt.tight_layout()
        
        # 保存图表
        if self.auto_save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=VISUALIZATION_CONFIG['dpi'], 
                       bbox_inches='tight', facecolor='white')
            print(f"📊 Performance analysis saved: {filepath}")
        
        plt.close(fig)
        return filepath if self.auto_save else None
    
    def create_ani_identity_analysis(self, initial_genome, final_genome, filename: str = None) -> str:
        """
        创建ANI和同源基因identity分布分析图表
        
        Args:
            initial_genome: 初始基因组
            final_genome: 最终基因组
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if not ANI_AVAILABLE:
            print("⚠️  ANI analysis not available - missing analysis modules")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ani_identity_analysis_{timestamp}.png"
        
        try:
            # 创建ANI计算器
            ani_calculator = ANICalculator(
                ortholog_identity_threshold=0.1,  # 进一步降低阈值以捕获更多同源基因
                min_alignment_length=30
            )
            
            # 计算ANI
            print("🧬 Computing ANI and ortholog analysis...")
            ani_result = ani_calculator.calculate_ani(initial_genome, final_genome)
            
            # 尝试保守性分析（可选）
            conservation_result = {}
            try:
                conservation_analyzer = ConservationAnalyzer(
                    conservation_threshold=0.7,
                    moderate_conservation_threshold=0.85,
                    high_conservation_threshold=0.95
                )
                conservation_result = conservation_analyzer.analyze_genome_conservation(
                    initial_genome, final_genome
                )
            except Exception as cons_error:
                print(f"⚠️  Conservation analysis failed: {cons_error}")
                conservation_result = {}
            
            # 创建图表
            fig = plt.figure(figsize=VISUALIZATION_CONFIG['figsize_large'])
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # 主标题
            fig.suptitle('ANI and Homologous Gene Identity Analysis', 
                        fontsize=16, fontweight='bold')
            
            # 1. 同源基因identity分布直方图
            ax1 = fig.add_subplot(gs[0, 0])
            identities = ani_result.get('identity_distribution', [])
            if identities and len(identities) > 0:
                ax1.hist(identities, bins=min(20, len(identities)), alpha=0.7, color='skyblue', 
                        edgecolor='black', density=True)
                ax1.axvline(np.mean(identities), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {np.mean(identities):.3f}')
                ax1.axvline(np.median(identities), color='orange', linestyle='--', 
                           linewidth=2, label=f'Median: {np.median(identities):.3f}')
                ax1.set_xlabel('Sequence Identity')
                ax1.set_ylabel('Density')
                ax1.set_title('Ortholog Identity Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                # 显示基因组比较信息，即使没有同源基因
                ax1.text(0.5, 0.7, 'No ortholog pairs found', 
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=14, fontweight='bold', color='red')
                ax1.text(0.5, 0.5, f'Initial genes: {len(initial_genome.genes)}', 
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, alpha=0.8)
                ax1.text(0.5, 0.3, f'Final genes: {len(final_genome.genes)}', 
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, alpha=0.8)
                ax1.set_title('Ortholog Identity Distribution')
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
            
            # 2. ANI统计摘要
            ax2 = fig.add_subplot(gs[0, 1])
            ani_stats = [
                f"Simple ANI: {ani_result.get('ani', 0):.3f}",
                f"Weighted ANI: {ani_result.get('weighted_ani', 0):.3f}",
                f"Ortholog Pairs: {ani_result.get('ortholog_count', 0)}",
                f"Ortholog Ratio: {ani_result.get('ortholog_ratio', 0):.3f}",
                f"Total Genes (Initial): {len(initial_genome.genes)}",
                f"Total Genes (Final): {len(final_genome.genes)}"
            ]
            
            ax2.text(0.1, 0.9, '\n'.join(ani_stats), 
                    transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('ANI Statistics Summary')
            
            # 3. 保守性分类饼图
            ax3 = fig.add_subplot(gs[0, 2])
            if 'conservation_results' in conservation_result:
                conservation_categories = {}
                for result in conservation_result['conservation_results']:
                    category = result.conservation_category
                    conservation_categories[category] = conservation_categories.get(category, 0) + 1
                
                if conservation_categories:
                    labels = list(conservation_categories.keys())
                    sizes = list(conservation_categories.values())
                    colors = ['lightgreen', 'yellow', 'orange', 'red'][:len(labels)]
                    
                    ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                           colors=colors, startangle=90)
                    ax3.set_title('Conservation Categories')
                else:
                    ax3.text(0.5, 0.5, 'No conservation\ndata available', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Conservation Categories')
            
            # 4. Identity vs 基因长度散点图
            ax4 = fig.add_subplot(gs[1, 0])
            if identities and 'alignment_lengths' in ani_result:
                lengths = ani_result['alignment_lengths']
                ax4.scatter(lengths, identities, alpha=0.6, s=30, color='blue')
                ax4.set_xlabel('Alignment Length (bp)')
                ax4.set_ylabel('Sequence Identity')
                ax4.set_title('Identity vs Alignment Length')
                ax4.grid(True, alpha=0.3)
                
                # 添加趋势线
                if len(lengths) > 1:
                    z = np.polyfit(lengths, identities, 1)
                    p = np.poly1d(z)
                    ax4.plot(lengths, p(lengths), "r--", alpha=0.8, 
                            label=f'Trend: y={z[0]:.2e}x+{z[1]:.3f}')
                    ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor scatter plot', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Identity vs Alignment Length')
            
            # 5. 基因类型保守性比较
            ax5 = fig.add_subplot(gs[1, 1])
            if 'gene_type_analysis' in conservation_result:
                gene_type_data = conservation_result['gene_type_analysis']
                categories = []
                mean_identities = []
                
                for gene_type, data in gene_type_data.items():
                    if data['count'] > 0:
                        categories.append(gene_type.replace('_', ' ').title())
                        mean_identities.append(data['mean_identity'])
                
                if categories:
                    bars = ax5.bar(categories, mean_identities, 
                                  color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(categories)],
                                  edgecolor='black', alpha=0.7)
                    ax5.set_ylabel('Mean Identity')
                    ax5.set_title('Conservation by Gene Type')
                    ax5.tick_params(axis='x', rotation=45)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, mean_identities):
                        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax5.text(0.5, 0.5, 'No gene type\ndata available', 
                            ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('Conservation by Gene Type')
            
            # 6. Identity分布箱线图
            ax6 = fig.add_subplot(gs[1, 2])
            if identities:
                box_data = [identities]
                labels = ['All Orthologs']
                
                # 如果有基因类型数据，添加分类箱线图
                if 'gene_type_analysis' in conservation_result:
                    gene_type_data = conservation_result['gene_type_analysis']
                    for gene_type, data in gene_type_data.items():
                        if data['count'] > 0 and 'identities' in data:
                            box_data.append(data['identities'])
                            # 缩短标签以避免布局问题
                            short_label = gene_type.replace('_', ' ').title()[:10]
                            labels.append(short_label)
                
                bp = ax6.boxplot(box_data, labels=labels, patch_artist=True)
                
                # 设置颜色
                colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                
                ax6.set_ylabel('Sequence Identity')
                ax6.set_title('Identity Distribution by Category')
                # 调整x轴标签角度和字体大小
                ax6.tick_params(axis='x', rotation=30, labelsize=9)
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No identity data\navailable', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Identity Distribution by Category')
            
            # 使用更宽松的布局参数
            plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
            
            # 保存图表
            if self.auto_save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=VISUALIZATION_CONFIG['dpi'], 
                           bbox_inches='tight', facecolor='white')
                print(f"📊 ANI identity analysis saved: {filepath}")
            
            plt.close(fig)
            return filepath if self.auto_save else None
            
        except Exception as e:
            print(f"❌ Error creating ANI analysis: {e}")
            return None
    
    def create_comprehensive_report(self, results: Dict, 
                                  performance_data: Dict = None,
                                  filename: str = None) -> List[str]:
        """
        创建综合报告（多个图表）
        
        Args:
            results: 模拟结果
            performance_data: 性能数据
            filename: 基础文件名
            
        Returns:
            保存的文件路径列表
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_report_{timestamp}"
        
        saved_files = []
        
        # 1. 进化总结
        summary_file = self.create_evolution_summary(
            results, f"{filename}_evolution.png")
        if summary_file:
            saved_files.append(summary_file)
        
        # 2. 基因丢失分析（如果有相关数据）
        if 'gene_loss_stats' in results or any('gene_loss' in str(s) for s in results.get('snapshots', [])):
            loss_file = self.create_gene_loss_analysis(
                results, f"{filename}_gene_loss.png")
            if loss_file:
                saved_files.append(loss_file)
        
        # 3. ANI和同源基因identity分析
        if 'initial_genome' in results and 'final_genome' in results:
            ani_file = self.create_ani_identity_analysis(
                results['initial_genome'], 
                results['final_genome'], 
                f"{filename}_ani_identity.png"
            )
            if ani_file:
                saved_files.append(ani_file)
        elif 'snapshots' in results and len(results['snapshots']) >= 2:
            # 如果没有直接的初始和最终基因组，使用快照中的第一个和最后一个
            snapshots = results['snapshots']
            if 'genome' in snapshots[0] and 'genome' in snapshots[-1]:
                ani_file = self.create_ani_identity_analysis(
                    snapshots[0]['genome'], 
                    snapshots[-1]['genome'], 
                    f"{filename}_ani_identity.png"
                )
                if ani_file:
                    saved_files.append(ani_file)
        
        # 4. 性能分析（如果有性能数据）
        if performance_data:
            perf_file = self.create_performance_analysis(
                performance_data, f"{filename}_performance.png")
            if perf_file:
                saved_files.append(perf_file)
        
        print(f"📋 Comprehensive report generated: {len(saved_files)} files")
        return saved_files


# 便捷函数
def create_evolution_visualization(results: Dict, output_dir: str = None) -> str:
    """
    便捷函数：创建进化可视化
    
    Args:
        results: 模拟结果
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    visualizer = EvolutionVisualizer(output_dir=output_dir)
    return visualizer.create_evolution_summary(results)


def create_comprehensive_visualization(results: Dict, 
                                     performance_data: Dict = None,
                                     output_dir: str = None) -> List[str]:
    """
    便捷函数：创建综合可视化报告
    
    Args:
        results: 模拟结果
        performance_data: 性能数据
        output_dir: 输出目录
        
    Returns:
        保存的文件路径列表
    """
    visualizer = EvolutionVisualizer(output_dir=output_dir)
    return visualizer.create_comprehensive_report(results, performance_data)


# 配置函数
def configure_visualization(dpi: int = None, 
                          output_dir: str = None,
                          color_palette: Dict = None):
    """
    配置可视化参数
    
    Args:
        dpi: 图像分辨率
        output_dir: 默认输出目录
        color_palette: 颜色配置
    """
    if dpi is not None:
        VISUALIZATION_CONFIG['dpi'] = dpi
    if output_dir is not None:
        VISUALIZATION_CONFIG['output_dir'] = output_dir
    if color_palette is not None:
        VISUALIZATION_CONFIG['color_palette'].update(color_palette)


def test_visualization():
    """测试可视化功能"""
    print("🧪 Testing visualization module...")
    
    # 创建测试数据
    test_results = {
        'snapshots': [
            {
                'generation': i,
                'genome_stats': {
                    'total_size': 1000000 + i * 1000 + np.random.randint(-500, 500),
                    'gene_count': 1000 + i * 2 + np.random.randint(-5, 5),
                    'core_genes': 900 + i + np.random.randint(-3, 3),
                    'hgt_genes': 100 + i + np.random.randint(-2, 2),
                    'total_mutations': i * 100 + np.random.randint(0, 50),
                    'total_hgt_events': i * 10 + np.random.randint(0, 5),
                    'total_recombination_events': i * 5 + np.random.randint(0, 3)
                }
            }
            for i in range(0, 51, 5)
        ]
    }
    
    # 测试可视化
    visualizer = EvolutionVisualizer(output_dir='test_results')
    files = visualizer.create_comprehensive_report(test_results)
    
    print(f"✅ Test completed. Generated {len(files)} files:")
    for file in files:
        print(f"   📊 {file}")


if __name__ == "__main__":
    test_visualization()