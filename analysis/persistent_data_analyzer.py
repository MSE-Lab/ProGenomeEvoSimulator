#!/usr/bin/env python3
"""
Persistent Data Analyzer - 持久化数据分析工具
用于加载、分析和可视化保存的进化模拟数据

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import json
import gzip
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 导入ANI和保守性分析器
try:
    from analysis.ani_calculator import ANICalculator
    from analysis.conservation_analyzer import ConservationAnalyzer
    ANI_AVAILABLE = True
except ImportError:
    print("⚠️  ANI Calculator or Conservation Analyzer not available")
    ANI_AVAILABLE = False


class PersistentDataAnalyzer:
    """
    持久化数据分析器
    
    功能特性：
    - 加载所有类型的持久化数据
    - 时间序列分析和可视化
    - 进化事件统计分析
    - ANI身份数据分析
    - 基因组快照比较
    - 生成综合分析报告
    """
    
    def __init__(self, run_directory: str):
        """
        初始化数据分析器
        
        Args:
            run_directory: 运行数据目录路径
        """
        self.run_dir = Path(run_directory)
        
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_directory}")
        
        # 子目录路径
        self.metadata_dir = self.run_dir / "metadata"
        self.snapshots_dir = self.run_dir / "snapshots"
        self.events_dir = self.run_dir / "events"
        self.stats_dir = self.run_dir / "statistics"
        self.analysis_dir = self.run_dir / "analysis"
        self.viz_dir = self.run_dir / "visualizations"
        
        # 确保可视化目录存在
        self.viz_dir.mkdir(exist_ok=True)
        
        # 数据缓存
        self._config = None
        self._run_info = None
        self._initial_genome = None
        self._evolved_genome = None
        self._genome_stats = None
        self._evolution_stats = None
        self._performance_stats = None
        self._ani_identities = None
        self._conservation_analysis = None
        self._final_summary = None
        
        print(f"📊 Persistent Data Analyzer initialized")
        print(f"📁 Analyzing data from: {self.run_dir}")
        
        # 验证数据完整性
        self.validate_data_integrity()
    
    def validate_data_integrity(self):
        """验证数据完整性"""
        required_files = [
            self.metadata_dir / "config.json",
            self.metadata_dir / "run_info.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
        else:
            print("✅ Data integrity check passed")
    
    def load_config(self) -> Dict:
        """加载模拟配置"""
        if self._config is None:
            config_path = self.metadata_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config
    
    def load_run_info(self) -> Dict:
        """加载运行信息"""
        if self._run_info is None:
            run_info_path = self.metadata_dir / "run_info.json"
            if run_info_path.exists():
                with open(run_info_path, 'r', encoding='utf-8') as f:
                    self._run_info = json.load(f)
            else:
                self._run_info = {}
        return self._run_info
    
    def load_genome_data(self, genome_type: str = "initial") -> Optional[Dict]:
        """
        加载基因组数据
        
        Args:
            genome_type: "initial" 或 "evolved"
        """
        if genome_type == "initial":
            if self._initial_genome is None:
                self._initial_genome = self._load_genome_file("initial_genome")
            return self._initial_genome
        elif genome_type == "evolved":
            if self._evolved_genome is None:
                self._evolved_genome = self._load_genome_file("evolved_genome")
            return self._evolved_genome
        else:
            raise ValueError("genome_type must be 'initial' or 'evolved'")
    
    def _load_genome_file(self, filename_prefix: str) -> Optional[Dict]:
        """加载基因组文件（支持压缩和非压缩）"""
        # 尝试加载压缩文件
        compressed_path = self.metadata_dir / f"{filename_prefix}.json.gz"
        if compressed_path.exists():
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        
        # 尝试加载非压缩文件
        regular_path = self.metadata_dir / f"{filename_prefix}.json"
        if regular_path.exists():
            with open(regular_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    
    def load_snapshots(self, generation_range: Optional[Tuple[int, int]] = None) -> Dict[int, Dict]:
        """
        加载基因组快照
        
        Args:
            generation_range: 可选的代数范围 (start, end)
        
        Returns:
            Dict[generation, genome_data]
        """
        snapshots = {}
        
        # 获取所有快照文件
        snapshot_files = list(self.snapshots_dir.glob("generation_*.json*"))
        snapshot_files.sort()
        
        for snapshot_file in snapshot_files:
            # 从文件名提取代数
            filename = snapshot_file.stem
            if filename.endswith('.json'):
                filename = filename[:-5]  # 移除 .json
            
            generation = int(filename.split('_')[1])
            
            # 检查是否在指定范围内
            if generation_range:
                start, end = generation_range
                if generation < start or generation > end:
                    continue
            
            # 加载快照数据
            if snapshot_file.suffix == '.gz':
                with gzip.open(snapshot_file, 'rt', encoding='utf-8') as f:
                    snapshots[generation] = json.load(f)
            else:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    snapshots[generation] = json.load(f)
        
        print(f"📸 Loaded {len(snapshots)} snapshots")
        return snapshots
    
    def load_genome_stats(self) -> pd.DataFrame:
        """加载基因组统计数据"""
        if self._genome_stats is None:
            stats_path = self.stats_dir / "genome_stats.csv"
            if stats_path.exists():
                self._genome_stats = pd.read_csv(stats_path)
                self._genome_stats['timestamp'] = pd.to_datetime(self._genome_stats['timestamp'])
            else:
                self._genome_stats = pd.DataFrame()
        return self._genome_stats
    
    def load_evolution_stats(self) -> pd.DataFrame:
        """加载进化事件统计数据"""
        if self._evolution_stats is None:
            stats_path = self.stats_dir / "evolution_stats.csv"
            if stats_path.exists():
                self._evolution_stats = pd.read_csv(stats_path)
                self._evolution_stats['timestamp'] = pd.to_datetime(self._evolution_stats['timestamp'])
            else:
                self._evolution_stats = pd.DataFrame()
        return self._evolution_stats
    
    def load_performance_stats(self) -> pd.DataFrame:
        """加载性能统计数据"""
        if self._performance_stats is None:
            stats_path = self.stats_dir / "performance_stats.csv"
            if stats_path.exists():
                self._performance_stats = pd.read_csv(stats_path)
                self._performance_stats['timestamp'] = pd.to_datetime(self._performance_stats['timestamp'])
            else:
                self._performance_stats = pd.DataFrame()
        return self._performance_stats
    
    def load_evolution_events(self, event_type: str) -> List[Dict]:
        """
        加载进化事件日志
        
        Args:
            event_type: 事件类型 ('mutations', 'hgt_events', 'recombination', 'gene_loss')
        """
        events = []
        event_file = self.events_dir / f"{event_type}.jsonl"
        
        if event_file.exists():
            with open(event_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num == 0:  # 跳过头部信息
                        continue
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        print(f"📝 Loaded {len(events)} {event_type} events")
        return events
    
    def load_ani_identities(self) -> List[Dict]:
        """加载ANI身份数据"""
        if self._ani_identities is None:
            ani_path = self.analysis_dir / "ani_identities.json"
            if ani_path.exists():
                with open(ani_path, 'r', encoding='utf-8') as f:
                    self._ani_identities = json.load(f)
            else:
                self._ani_identities = []
        return self._ani_identities
    
    def load_conservation_analysis(self) -> Dict:
        """加载保守性分析结果"""
        if self._conservation_analysis is None:
            conservation_path = self.analysis_dir / "conservation_analysis.json"
            if conservation_path.exists():
                with open(conservation_path, 'r', encoding='utf-8') as f:
                    self._conservation_analysis = json.load(f)
            else:
                self._conservation_analysis = {}
        return self._conservation_analysis
    
    def load_final_summary(self) -> Dict:
        """加载最终摘要"""
        if self._final_summary is None:
            summary_path = self.analysis_dir / "final_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    self._final_summary = json.load(f)
            else:
                self._final_summary = {}
        return self._final_summary
    
    def analyze_genome_evolution(self) -> Dict:
        """分析基因组进化趋势"""
        genome_stats = self.load_genome_stats()
        
        if genome_stats.empty:
            return {}
        
        analysis = {
            'size_evolution': {
                'initial_size': genome_stats['total_size'].iloc[0],
                'final_size': genome_stats['total_size'].iloc[-1],
                'size_change': genome_stats['total_size'].iloc[-1] - genome_stats['total_size'].iloc[0],
                'size_change_percent': ((genome_stats['total_size'].iloc[-1] - genome_stats['total_size'].iloc[0]) / 
                                      genome_stats['total_size'].iloc[0]) * 100,
                'max_size': genome_stats['total_size'].max(),
                'min_size': genome_stats['total_size'].min()
            },
            'gene_count_evolution': {
                'initial_genes': genome_stats['gene_count'].iloc[0],
                'final_genes': genome_stats['gene_count'].iloc[-1],
                'gene_change': genome_stats['gene_count'].iloc[-1] - genome_stats['gene_count'].iloc[0],
                'gene_change_percent': ((genome_stats['gene_count'].iloc[-1] - genome_stats['gene_count'].iloc[0]) / 
                                      genome_stats['gene_count'].iloc[0]) * 100,
                'max_genes': genome_stats['gene_count'].max(),
                'min_genes': genome_stats['gene_count'].min()
            },
            'hgt_evolution': {
                'initial_hgt_genes': genome_stats['hgt_genes'].iloc[0],
                'final_hgt_genes': genome_stats['hgt_genes'].iloc[-1],
                'total_hgt_events': genome_stats['total_hgt_events'].iloc[-1],
                'hgt_gene_percentage': (genome_stats['hgt_genes'].iloc[-1] / 
                                      genome_stats['gene_count'].iloc[-1]) * 100
            },
            'mutation_accumulation': {
                'total_mutations': genome_stats['total_mutations'].iloc[-1],
                'mutations_per_gene': genome_stats['total_mutations'].iloc[-1] / genome_stats['gene_count'].iloc[-1],
                'total_recombination': genome_stats['total_recombination_events'].iloc[-1]
            }
        }
        
        return analysis
    
    def plot_genome_evolution_timeline(self, save_plot: bool = True) -> None:
        """绘制基因组进化时间线"""
        genome_stats = self.load_genome_stats()
        
        if genome_stats.empty:
            print("⚠️  No genome statistics data available")
            return
        
        # 清理NaN值和无效数据
        genome_stats = genome_stats.dropna()
        if genome_stats.empty:
            print("⚠️  No valid genome statistics data after cleaning")
            return
        
        # 确保数值列是数值类型
        numeric_columns = ['generation', 'total_size', 'gene_count', 'core_genes', 'hgt_genes', 
                          'total_mutations', 'total_hgt_events', 'total_recombination_events', 'avg_gene_length']
        for col in numeric_columns:
            if col in genome_stats.columns:
                genome_stats[col] = pd.to_numeric(genome_stats[col], errors='coerce')
        
        # 再次清理转换后的NaN值
        genome_stats = genome_stats.dropna(subset=numeric_columns)
        if genome_stats.empty:
            print("⚠️  No valid numeric data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Genome Evolution Timeline', fontsize=16, fontweight='bold')
        
        # 基因组大小变化
        axes[0, 0].plot(genome_stats['generation'], genome_stats['total_size'], 
                       color='blue', linewidth=2)
        axes[0, 0].set_title('Genome Size Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Total Size (bp)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 基因数量变化
        axes[0, 1].plot(genome_stats['generation'], genome_stats['gene_count'], 
                       color='green', linewidth=2, label='Total Genes')
        axes[0, 1].plot(genome_stats['generation'], genome_stats['core_genes'], 
                       color='darkgreen', linewidth=2, label='Core Genes')
        axes[0, 1].plot(genome_stats['generation'], genome_stats['hgt_genes'], 
                       color='orange', linewidth=2, label='HGT Genes')
        axes[0, 1].set_title('Gene Count Evolution')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Gene Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 累积突变和HGT事件
        axes[1, 0].plot(genome_stats['generation'], genome_stats['total_mutations'], 
                       color='red', linewidth=2, label='Total Mutations')
        axes[1, 0].plot(genome_stats['generation'], genome_stats['total_hgt_events'], 
                       color='purple', linewidth=2, label='Total HGT Events')
        axes[1, 0].plot(genome_stats['generation'], genome_stats['total_recombination_events'], 
                       color='brown', linewidth=2, label='Total Recombination')
        axes[1, 0].set_title('Cumulative Evolution Events')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Event Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 平均基因长度和GC含量
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(genome_stats['generation'], genome_stats['avg_gene_length'], 
                        color='cyan', linewidth=2, label='Avg Gene Length')
        ax1.set_ylabel('Average Gene Length (bp)', color='cyan')
        ax1.tick_params(axis='y', labelcolor='cyan')
        
        if 'gc_content' in genome_stats.columns:
            line2 = ax2.plot(genome_stats['generation'], genome_stats['gc_content'], 
                           color='magenta', linewidth=2, label='GC Content')
            ax2.set_ylabel('GC Content', color='magenta')
            ax2.tick_params(axis='y', labelcolor='magenta')
        
        ax1.set_title('Gene Length & GC Content')
        ax1.set_xlabel('Generation')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.viz_dir / "genome_evolution_timeline.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_evolution_events_analysis(self, save_plot: bool = True) -> None:
        """分析和可视化进化事件"""
        evolution_stats = self.load_evolution_stats()
        
        if evolution_stats.empty:
            print("⚠️  No evolution statistics data available")
            return
        
        # 清理NaN值和无效数据
        evolution_stats = evolution_stats.dropna()
        if evolution_stats.empty:
            print("⚠️  No valid evolution statistics data after cleaning")
            return
        
        # 确保数值列是数值类型
        numeric_columns = ['generation', 'mutations_this_gen', 'hgt_events_this_gen', 
                          'recombination_events_this_gen', 'genes_lost_this_gen']
        for col in numeric_columns:
            if col in evolution_stats.columns:
                evolution_stats[col] = pd.to_numeric(evolution_stats[col], errors='coerce')
        
        # 再次清理转换后的NaN值
        evolution_stats = evolution_stats.dropna(subset=numeric_columns)
        if evolution_stats.empty:
            print("⚠️  No valid numeric evolution data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Evolution Events Analysis', fontsize=16, fontweight='bold')
        
        # 每代进化事件数量
        axes[0, 0].plot(evolution_stats['generation'], evolution_stats['mutations_this_gen'], 
                       color='red', linewidth=1, alpha=0.7, label='Mutations')
        axes[0, 0].plot(evolution_stats['generation'], evolution_stats['hgt_events_this_gen'], 
                       color='blue', linewidth=1, alpha=0.7, label='HGT Events')
        axes[0, 0].plot(evolution_stats['generation'], evolution_stats['recombination_events_this_gen'], 
                       color='green', linewidth=1, alpha=0.7, label='Recombination')
        axes[0, 0].plot(evolution_stats['generation'], evolution_stats['genes_lost_this_gen'], 
                       color='orange', linewidth=1, alpha=0.7, label='Gene Loss')
        axes[0, 0].set_title('Evolution Events per Generation')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Event Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 事件类型分布（饼图）
        total_mutations = evolution_stats['mutations_this_gen'].sum()
        total_hgt = evolution_stats['hgt_events_this_gen'].sum()
        total_recombination = evolution_stats['recombination_events_this_gen'].sum()
        total_gene_loss = evolution_stats['genes_lost_this_gen'].sum()
        
        # 处理NaN值和零值
        event_counts = [
            max(0, int(total_mutations) if not pd.isna(total_mutations) else 0),
            max(0, int(total_hgt) if not pd.isna(total_hgt) else 0),
            max(0, int(total_recombination) if not pd.isna(total_recombination) else 0),
            max(0, int(total_gene_loss) if not pd.isna(total_gene_loss) else 0)
        ]
        
        # 只显示非零事件
        non_zero_indices = [i for i, count in enumerate(event_counts) if count > 0]
        if non_zero_indices:
            event_labels = ['Mutations', 'HGT Events', 'Recombination', 'Gene Loss']
            colors = ['red', 'blue', 'green', 'orange']
            
            filtered_counts = [event_counts[i] for i in non_zero_indices]
            filtered_labels = [event_labels[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            axes[0, 1].pie(filtered_counts, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%')
            axes[0, 1].set_title('Total Evolution Events Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Evolution Events', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Total Evolution Events Distribution')
        
        # 处理时间分析
        if 'wall_clock_time' in evolution_stats.columns:
            axes[1, 0].plot(evolution_stats['generation'], evolution_stats['wall_clock_time'], 
                           color='purple', linewidth=2)
            axes[1, 0].set_title('Processing Time per Generation')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Wall Clock Time (s)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 事件频率热图
        window_size = max(1, len(evolution_stats) // 20)  # 20个时间窗口
        windowed_data = []
        
        for i in range(0, len(evolution_stats), window_size):
            window = evolution_stats.iloc[i:i+window_size]
            # 处理NaN值，使用0替代
            windowed_data.append([
                window['mutations_this_gen'].fillna(0).mean(),
                window['hgt_events_this_gen'].fillna(0).mean(),
                window['recombination_events_this_gen'].fillna(0).mean(),
                window['genes_lost_this_gen'].fillna(0).mean()
            ])
        
        if windowed_data:
            # 转换为numpy数组并处理NaN值
            heatmap_data = np.array(windowed_data).T
            heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)  # 将NaN替换为0
            
            if heatmap_data.size > 0 and not np.all(heatmap_data == 0):
                im = axes[1, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                axes[1, 1].set_title('Evolution Events Heatmap')
                axes[1, 1].set_xlabel('Time Window')
                axes[1, 1].set_ylabel('Event Type')
                axes[1, 1].set_yticks(range(4))
                axes[1, 1].set_yticklabels(['Mutations', 'HGT', 'Recombination', 'Gene Loss'])
                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'No Event Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Evolution Events Heatmap')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.viz_dir / "evolution_events_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    def compare_initial_vs_final_genome(self) -> Dict:
        """比较初始和最终基因组"""
        initial_genome = self.load_genome_data("initial")
        evolved_genome = self.load_genome_data("evolved")
        
        if not initial_genome or not evolved_genome:
            print("⚠️  Cannot load genome data for comparison")
            return {}
        
        comparison = {
            'basic_stats': {
                'initial_genes': len(initial_genome['genes']),
                'final_genes': len(evolved_genome['genes']),
                'gene_change': len(evolved_genome['genes']) - len(initial_genome['genes']),
                'initial_size': sum(gene['length'] for gene in initial_genome['genes']),
                'final_size': sum(gene['length'] for gene in evolved_genome['genes'])
            },
            'gene_origin_analysis': {},
            'mutation_analysis': {},
            'hgt_analysis': {}
        }
        
        # 分析基因来源
        initial_core_genes = sum(1 for gene in initial_genome['genes'] if gene['is_core'])
        final_core_genes = sum(1 for gene in evolved_genome['genes'] if gene['is_core'])
        final_hgt_genes = sum(1 for gene in evolved_genome['genes'] if hasattr(gene, 'hgt_origin'))
        
        comparison['gene_origin_analysis'] = {
            'initial_core_genes': initial_core_genes,
            'final_core_genes': final_core_genes,
            'final_hgt_genes': final_hgt_genes,
            'core_gene_retention': final_core_genes / initial_core_genes if initial_core_genes > 0 else 0
        }
        
        # 分析突变累积
        total_mutations = sum(gene['mutation_count'] for gene in evolved_genome['genes'])
        total_recombination = sum(gene['recombination_count'] for gene in evolved_genome['genes'])
        
        comparison['mutation_analysis'] = {
            'total_mutations': total_mutations,
            'mutations_per_gene': total_mutations / len(evolved_genome['genes']),
            'total_recombination': total_recombination,
            'recombination_per_gene': total_recombination / len(evolved_genome['genes'])
        }
        
        return comparison
    
    def calculate_ani_matrix(self, genomes: List[Dict]) -> Dict:
        """
        计算ANI身份矩阵
        
        Args:
            genomes: 基因组数据列表
            
        Returns:
            ANI分析结果
        """
        if not ANI_AVAILABLE:
            print("⚠️  ANI Calculator not available")
            return {}
        
        if len(genomes) < 2:
            print("⚠️  Need at least 2 genomes for ANI calculation")
            return {}
        
        try:
            # 创建ANI计算器
            ani_calculator = ANICalculator(
                ortholog_identity_threshold=0.1,
                min_alignment_length=30
            )
            
            print(f"🧮 Calculating ANI for {len(genomes)} genomes...")
            
            # 计算ANI矩阵
            ani_results = {}
            identity_data = []
            
            for i in range(len(genomes)):
                for j in range(i + 1, len(genomes)):
                    genome1 = genomes[i]
                    genome2 = genomes[j]
                    
                    # 转换为Genome对象（如果需要）
                    if isinstance(genome1, dict):
                        from core.genome import Genome
                        g1 = Genome()
                        g1.genes = [gene for gene in genome1.get('genes', [])]
                        g1.total_length = genome1.get('total_length', 0)
                    else:
                        g1 = genome1
                    
                    if isinstance(genome2, dict):
                        from core.genome import Genome
                        g2 = Genome()
                        g2.genes = [gene for gene in genome2.get('genes', [])]
                        g2.total_length = genome2.get('total_length', 0)
                    else:
                        g2 = genome2
                    
                    # 计算ANI
                    ani_result = ani_calculator.calculate_ani(g1, g2)
                    
                    # 创建可序列化的结果副本（移除OrthologPair对象）
                    serializable_result = {
                        'ani': ani_result.get('ani', 0.0),
                        'weighted_ani': ani_result.get('weighted_ani', 0.0),
                        'ortholog_count': ani_result.get('ortholog_count', 0),
                        'total_genes_genome1': ani_result.get('total_genes_genome1', 0),
                        'total_genes_genome2': ani_result.get('total_genes_genome2', 0),
                        'ortholog_ratio': ani_result.get('ortholog_ratio', 0.0),
                        'identity_distribution': ani_result.get('identity_distribution', []),
                        'alignment_lengths': ani_result.get('alignment_lengths', [])
                    }
                    
                    pair_key = f"genome_{i}_vs_genome_{j}"
                    ani_results[pair_key] = serializable_result
                    
                    # 收集身份数据
                    if 'identity_distribution' in ani_result:
                        identity_data.extend(ani_result['identity_distribution'])
            
            # 汇总结果
            summary_result = {
                'timestamp': datetime.now().isoformat(),
                'genome_count': len(genomes),
                'pairwise_comparisons': len(ani_results),
                'ani_results': ani_results,
                'identity_statistics': {
                    'total_identities': len(identity_data),
                    'mean_identity': np.mean(identity_data) if identity_data else 0,
                    'median_identity': np.median(identity_data) if identity_data else 0,
                    'std_identity': np.std(identity_data) if identity_data else 0,
                    'min_identity': np.min(identity_data) if identity_data else 0,
                    'max_identity': np.max(identity_data) if identity_data else 0
                },
                'identity_distribution': identity_data
            }
            
            # 保存ANI结果
            ani_path = self.analysis_dir / "ani_identities.json"
            with open(ani_path, 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False)
            
            print(f"🧬 ANI analysis completed: {len(identity_data)} identity values")
            print(f"📊 Mean identity: {summary_result['identity_statistics']['mean_identity']:.3f}")
            print(f"💾 ANI results saved to: {ani_path}")
            
            # 生成ANI可视化
            self.plot_ani_analysis(summary_result)
            
            return summary_result
            
        except Exception as e:
            print(f"❌ ANI calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def plot_ani_analysis(self, ani_data: Dict, save_plot: bool = True) -> None:
        """绘制ANI分析结果"""
        if not ani_data or 'identity_distribution' not in ani_data:
            print("⚠️  No ANI data available for plotting")
            return
        
        identities = ani_data['identity_distribution']
        if not identities:
            print("⚠️  No identity data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ANI (Average Nucleotide Identity) Analysis', fontsize=16, fontweight='bold')
        
        # 1. 身份分布直方图
        axes[0, 0].hist(identities, bins=min(30, len(identities)//10 + 1), 
                       alpha=0.7, color='skyblue', edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(identities), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(identities):.3f}')
        axes[0, 0].axvline(np.median(identities), color='orange', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(identities):.3f}')
        axes[0, 0].set_xlabel('Sequence Identity')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Ortholog Identity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 累积分布函数
        sorted_identities = np.sort(identities)
        cumulative = np.arange(1, len(sorted_identities) + 1) / len(sorted_identities)
        axes[0, 1].plot(sorted_identities, cumulative, linewidth=2, color='green')
        axes[0, 1].set_xlabel('Sequence Identity')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 箱线图
        axes[1, 0].boxplot(identities, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 0].set_ylabel('Sequence Identity')
        axes[1, 0].set_title('Identity Distribution Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 统计摘要
        stats = ani_data.get('identity_statistics', {})
        stats_text = f"""
        Total Comparisons: {len(identities):,}
        Mean Identity: {stats.get('mean_identity', 0):.4f}
        Median Identity: {stats.get('median_identity', 0):.4f}
        Std Deviation: {stats.get('std_identity', 0):.4f}
        Min Identity: {stats.get('min_identity', 0):.4f}
        Max Identity: {stats.get('max_identity', 0):.4f}
        
        Genome Count: {ani_data.get('genome_count', 0)}
        Pairwise Comparisons: {ani_data.get('pairwise_comparisons', 0)}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('ANI Statistics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.viz_dir / "ani_identity_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 ANI plot saved to: {plot_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, output_file: Optional[str] = None) -> str:
        """生成综合分析报告"""
        report_lines = []
        
        # 报告头部
        run_info = self.load_run_info()
        config = self.load_config()
        
        report_lines.extend([
            "=" * 80,
            "PROKARYOTIC GENOME EVOLUTION SIMULATION - COMPREHENSIVE ANALYSIS REPORT",
            "=" * 80,
            f"Run ID: {run_info.get('run_id', 'Unknown')}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Directory: {self.run_dir}",
            "",
            "SIMULATION CONFIGURATION",
            "-" * 40
        ])
        
        # 配置信息
        if config:
            report_lines.extend([
                f"Generations: {config.get('generations', 'Unknown')}",
                f"Snapshot Interval: {config.get('snapshot_interval', 'Unknown')}",
                f"Engine Config: {config.get('engine_config', {})}"
            ])
        
        report_lines.append("")
        
        # 基因组进化分析
        genome_analysis = self.analyze_genome_evolution()
        if genome_analysis:
            report_lines.extend([
                "GENOME EVOLUTION ANALYSIS",
                "-" * 40,
                f"Initial Size: {genome_analysis['size_evolution']['initial_size']:,} bp",
                f"Final Size: {genome_analysis['size_evolution']['final_size']:,} bp",
                f"Size Change: {genome_analysis['size_evolution']['size_change']:+,} bp ({genome_analysis['size_evolution']['size_change_percent']:+.2f}%)",
                "",
                f"Initial Genes: {genome_analysis['gene_count_evolution']['initial_genes']:,}",
                f"Final Genes: {genome_analysis['gene_count_evolution']['final_genes']:,}",
                f"Gene Change: {genome_analysis['gene_count_evolution']['gene_change']:+,} ({genome_analysis['gene_count_evolution']['gene_change_percent']:+.2f}%)",
                "",
                f"Total Mutations: {genome_analysis['mutation_accumulation']['total_mutations']:,}",
                f"Mutations per Gene: {genome_analysis['mutation_accumulation']['mutations_per_gene']:.2f}",
                f"Total HGT Events: {genome_analysis['hgt_evolution']['total_hgt_events']:,}",
                f"HGT Gene Percentage: {genome_analysis['hgt_evolution']['hgt_gene_percentage']:.2f}%",
                ""
            ])
        
        # 基因组比较
        comparison = self.compare_initial_vs_final_genome()
        if comparison:
            report_lines.extend([
                "INITIAL vs FINAL GENOME COMPARISON",
                "-" * 40,
                f"Gene Count Change: {comparison['basic_stats']['initial_genes']} → {comparison['basic_stats']['final_genes']} ({comparison['basic_stats']['gene_change']:+})",
                f"Genome Size Change: {comparison['basic_stats']['initial_size']:,} → {comparison['basic_stats']['final_size']:,} bp",
                f"Core Gene Retention: {comparison['gene_origin_analysis']['core_gene_retention']:.2%}",
                f"HGT Genes Acquired: {comparison['gene_origin_analysis']['final_hgt_genes']:,}",
                ""
            ])
        
        # 数据文件统计
        snapshots_count = len(list(self.snapshots_dir.glob('*.json*')))
        stats_files_count = len(list(self.stats_dir.glob('*.csv')))
        event_files_count = len(list(self.events_dir.glob('*.jsonl')))
        
        report_lines.extend([
            "DATA PERSISTENCE SUMMARY",
            "-" * 40,
            f"Snapshots Saved: {snapshots_count}",
            f"Statistics Files: {stats_files_count}",
            f"Event Log Files: {event_files_count}",
            f"Analysis Files: {len(list(self.analysis_dir.glob('*.json')))}",
            "",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if output_file is None:
            output_file = self.analysis_dir / "comprehensive_analysis_report.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📋 Comprehensive report saved to: {output_file}")
        return report_text
    
    def export_data_summary(self) -> Dict:
        """导出数据摘要用于进一步分析"""
        summary = {
            'run_info': self.load_run_info(),
            'config': self.load_config(),
            'genome_evolution': self.analyze_genome_evolution(),
            'genome_comparison': self.compare_initial_vs_final_genome(),
            'data_files': {
                'snapshots': len(list(self.snapshots_dir.glob('*.json*'))),
                'statistics': len(list(self.stats_dir.glob('*.csv'))),
                'events': len(list(self.events_dir.glob('*.jsonl'))),
                'analysis': len(list(self.analysis_dir.glob('*.json')))
            },
            'final_summary': self.load_final_summary()
        }
        
        return summary


def analyze_run(run_directory: str, generate_plots: bool = True, generate_report: bool = True) -> PersistentDataAnalyzer:
    """
    便捷函数：分析指定运行的数据
    
    Args:
        run_directory: 运行数据目录
        generate_plots: 是否生成可视化图表
        generate_report: 是否生成综合报告
    
    Returns:
        PersistentDataAnalyzer实例
    """
    analyzer = PersistentDataAnalyzer(run_directory)
    
    if generate_plots:
        print("📊 Generating visualization plots...")
        analyzer.plot_genome_evolution_timeline()
        analyzer.plot_evolution_events_analysis()
    
    if generate_report:
        print("📋 Generating comprehensive report...")
        analyzer.generate_comprehensive_report()
    
    return analyzer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
        analyzer = analyze_run(run_dir)
        print(f"\n🎉 Analysis completed for: {run_dir}")
    else:
        print("Usage: python persistent_data_analyzer.py <run_directory>")