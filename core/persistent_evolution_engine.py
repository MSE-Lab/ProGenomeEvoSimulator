#!/usr/bin/env python3
"""
Persistent Evolution Engine - 数据持久化进化引擎
将所有模拟数据保存到硬盘，支持中断恢复和历史数据分析

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import json
import gzip
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from core.genome import Genome, Gene
from core.unified_evolution_engine import UnifiedEvolutionEngine


class PersistentEvolutionEngine(UnifiedEvolutionEngine):
    """
    持久化进化引擎 - 继承统一进化引擎，添加数据持久化功能
    
    功能特性：
    - 自动保存所有模拟数据到硬盘
    - 支持压缩存储节省空间
    - 详细的进化事件日志
    - 定期快照和统计数据
    - ANI身份数据集保存
    - 支持中断恢复（未来扩展）
    """
    
    def __init__(self, 
                 # 继承父类参数
                 mutation_rate: float = 1e-6,
                 hgt_rate: float = 1e-8,
                 recombination_rate: float = 1e-9,
                 min_similarity_for_recombination: float = 0.85,
                 enable_gene_loss: bool = True,
                 loss_rate: float = 1e-7,
                 core_gene_protection: float = 0.98,
                 hgt_gene_loss_multiplier: float = 20.0,
                 min_genome_size: int = 1200,
                 min_core_genes: int = 1000,
                 optimal_genome_size: int = 3000,
                 enable_parallel: bool = True,
                 num_processes: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 parallel_threshold: int = 500,
                 enable_optimization: bool = True,
                 enable_progress_sharing: bool = True,
                 
                 # 持久化参数
                 base_output_dir: str = "simulation_results",
                 snapshot_interval: int = 100,
                 stats_flush_interval: int = 10,
                 compress_data: bool = True,
                 save_detailed_events: bool = True,
                 save_sequences: bool = True):
        """
        初始化持久化进化引擎
        
        Args:
            # 父类参数（进化机制相关）
            ...
            
            # 持久化参数
            base_output_dir: 基础输出目录
            snapshot_interval: 快照保存间隔（代数）
            stats_flush_interval: 统计数据刷新间隔（代数）
            compress_data: 是否压缩数据文件
            save_detailed_events: 是否保存详细的进化事件
            save_sequences: 是否保存完整的基因序列
        """
        
        # 初始化父类
        super().__init__(
            mutation_rate=mutation_rate,
            hgt_rate=hgt_rate,
            recombination_rate=recombination_rate,
            min_similarity_for_recombination=min_similarity_for_recombination,
            enable_gene_loss=enable_gene_loss,
            loss_rate=loss_rate,
            core_gene_protection=core_gene_protection,
            hgt_gene_loss_multiplier=hgt_gene_loss_multiplier,
            min_genome_size=min_genome_size,
            min_core_genes=min_core_genes,
            optimal_genome_size=optimal_genome_size,
            enable_parallel=enable_parallel,
            num_processes=num_processes,
            chunk_size=chunk_size,
            parallel_threshold=parallel_threshold,
            enable_optimization=enable_optimization,
            enable_progress_sharing=enable_progress_sharing
        )
        
        # 持久化配置
        self.base_output_dir = Path(base_output_dir)
        self.snapshot_interval = snapshot_interval
        self.stats_flush_interval = stats_flush_interval
        self.compress_data = compress_data
        self.save_detailed_events = save_detailed_events
        self.save_sequences = save_sequences
        
        # 运行标识和目录
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{self.run_id}"
        
        # 子目录
        self.metadata_dir = self.run_dir / "metadata"
        self.snapshots_dir = self.run_dir / "snapshots"
        self.events_dir = self.run_dir / "events"
        self.stats_dir = self.run_dir / "statistics"
        self.analysis_dir = self.run_dir / "analysis"
        self.viz_dir = self.run_dir / "visualizations"
        
        # 事件日志文件句柄
        self.event_files = {}
        
        # 统计数据缓存
        self.stats_buffer = []
        self.performance_buffer = []
        
        # ANI身份数据集
        self.ani_identities = []
        
        # 初始化标志
        self.storage_initialized = False
        
        print(f"🗄️  Persistent Evolution Engine initialized:")
        print(f"   Output directory: {self.run_dir}")
        print(f"   Snapshot interval: {snapshot_interval} generations")
        print(f"   Data compression: {'Enabled' if compress_data else 'Disabled'}")
        print(f"   Detailed events: {'Enabled' if save_detailed_events else 'Disabled'}")
    
    def initialize_storage(self, config: Dict, initial_genome: Genome):
        """初始化存储系统"""
        print(f"🗄️  Initializing persistent storage...")
        
        # 创建所有目录
        for dir_path in [self.metadata_dir, self.snapshots_dir, 
                        self.events_dir, self.stats_dir, 
                        self.analysis_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置和初始基因组
        self.save_config(config)
        self.save_initial_genome(initial_genome)
        self.save_run_info()
        
        # 初始化事件日志文件
        self.initialize_event_logs()
        
        # 初始化统计CSV文件
        self.initialize_stats_files()
        
        self.storage_initialized = True
        print(f"✅ Storage initialized at: {self.run_dir}")
    
    def save_config(self, config: Dict):
        """保存模拟配置"""
        config_path = self.metadata_dir / "config.json"
        
        # 确保配置可序列化
        serializable_config = self._make_serializable(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Configuration saved to: {config_path}")
    
    def save_run_info(self):
        """保存运行元信息"""
        run_info = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'engine_version': '1.0.0',
            'storage_config': {
                'snapshot_interval': self.snapshot_interval,
                'stats_flush_interval': self.stats_flush_interval,
                'compress_data': self.compress_data,
                'save_detailed_events': self.save_detailed_events,
                'save_sequences': self.save_sequences
            },
            'directories': {
                'metadata': str(self.metadata_dir),
                'snapshots': str(self.snapshots_dir),
                'events': str(self.events_dir),
                'statistics': str(self.stats_dir),
                'analysis': str(self.analysis_dir),
                'visualizations': str(self.viz_dir)
            }
        }
        
        run_info_path = self.metadata_dir / "run_info.json"
        with open(run_info_path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)
    
    def save_initial_genome(self, genome: Genome):
        """保存初始基因组"""
        genome_data = self.serialize_genome(genome, include_sequences=self.save_sequences)
        
        filename = "initial_genome.json"
        if self.compress_data:
            filename += ".gz"
            with gzip.open(self.metadata_dir / filename, 'wt', encoding='utf-8') as f:
                json.dump(genome_data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.metadata_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(genome_data, f, indent=2, ensure_ascii=False)
        
        print(f"🧬 Initial genome saved: {len(genome.genes)} genes, {genome.size} bp")
    
    def save_evolved_genome(self, genome: Genome):
        """保存最终进化基因组"""
        genome_data = self.serialize_genome(genome, include_sequences=self.save_sequences)
        
        filename = "evolved_genome.json"
        if self.compress_data:
            filename += ".gz"
            with gzip.open(self.metadata_dir / filename, 'wt', encoding='utf-8') as f:
                json.dump(genome_data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.metadata_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(genome_data, f, indent=2, ensure_ascii=False)
        
        print(f"🧬 Final genome saved: {len(genome.genes)} genes, {genome.size} bp")
    
    def save_snapshot(self, genome: Genome, generation: int):
        """保存基因组快照"""
        if generation % self.snapshot_interval == 0:
            snapshot_data = self.serialize_genome(genome, include_sequences=self.save_sequences)
            snapshot_data['snapshot_info'] = {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'snapshot_type': 'regular'
            }
            
            filename = f"generation_{generation:06d}.json"
            if self.compress_data:
                filename += ".gz"
                with gzip.open(self.snapshots_dir / filename, 'wt', encoding='utf-8') as f:
                    json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            else:
                with open(self.snapshots_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
    
    def serialize_genome(self, genome: Genome, include_sequences: bool = True) -> Dict:
        """序列化基因组对象"""
        genome_data = {
            'generation': genome.generation,
            'total_mutations': genome.total_mutations,
            'total_hgt_events': genome.total_hgt_events,
            'total_recombination_events': genome.total_recombination_events,
            'statistics': genome.get_statistics(),
            'genes': []
        }
        
        for gene in genome.genes:
            gene_data = {
                'id': gene.id,
                'start_pos': gene.start_pos,
                'length': gene.length,
                'is_core': gene.is_core,
                'origin': gene.origin,
                'mutation_count': gene.mutation_count,
                'recombination_count': gene.recombination_count
            }
            
            if include_sequences:
                gene_data['sequence'] = gene.sequence
            
            # 保存HGT来源信息（如果存在）
            if hasattr(gene, 'hgt_origin'):
                gene_data['hgt_origin'] = {
                    'source_type': gene.hgt_origin.source_type,
                    'gc_content': gene.hgt_origin.gc_content,
                    'codon_usage_bias': gene.hgt_origin.codon_usage_bias,
                    'metabolic_category': gene.hgt_origin.metabolic_category,
                    'transfer_frequency': gene.hgt_origin.transfer_frequency
                }
            
            genome_data['genes'].append(gene_data)
        
        return genome_data
    
    def initialize_event_logs(self):
        """初始化事件日志文件"""
        if not self.save_detailed_events:
            return
        
        event_types = ['mutations', 'hgt_events', 'recombination', 'gene_loss']
        
        for event_type in event_types:
            filename = f"{event_type}.jsonl"
            filepath = self.events_dir / filename
            
            # 创建文件并写入头部信息
            with open(filepath, 'w', encoding='utf-8') as f:
                header = {
                    'log_type': event_type,
                    'created': datetime.now().isoformat(),
                    'format': 'JSON Lines (one JSON object per line)'
                }
                json.dump(header, f, ensure_ascii=False)
                f.write('\n')
    
    def initialize_stats_files(self):
        """初始化统计CSV文件"""
        # 基因组统计文件
        genome_stats_path = self.stats_dir / "genome_stats.csv"
        with open(genome_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'timestamp', 'total_size', 'gene_count', 
                'core_genes', 'hgt_genes', 'total_mutations', 
                'total_hgt_events', 'total_recombination_events',
                'avg_gene_length', 'gc_content'
            ])
        
        # 进化事件统计文件
        evolution_stats_path = self.stats_dir / "evolution_stats.csv"
        with open(evolution_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'timestamp', 'mutations_this_gen', 
                'hgt_events_this_gen', 'recombination_events_this_gen',
                'genes_lost_this_gen', 'processing_mode', 'wall_clock_time'
            ])
        
        # 性能统计文件
        performance_stats_path = self.stats_dir / "performance_stats.csv"
        with open(performance_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'timestamp', 'processing_mode', 
                'wall_clock_time', 'parallel_processing_time',
                'total_processing_time', 'chunks_processed', 'chunk_size'
            ])
    
    def log_evolution_event(self, event_type: str, event_data: Dict):
        """记录进化事件"""
        if not self.save_detailed_events:
            return
        
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'generation': event_data.get('generation', 0),
            'event_type': event_type,
            'data': self._make_serializable(event_data)
        }
        
        # 写入对应的事件日志文件
        filename = f"{event_type}.jsonl"
        filepath = self.events_dir / filename
        
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(event_record, f, ensure_ascii=False)
            f.write('\n')
    
    def save_generation_stats(self, genome: Genome, generation_stats: Dict):
        """保存代数统计数据"""
        # 添加到缓存
        stats_record = {
            'generation': genome.generation,
            'timestamp': datetime.now().isoformat(),
            'genome_stats': genome.get_statistics(),
            'generation_stats': generation_stats
        }
        
        self.stats_buffer.append(stats_record)
        
        # 定期刷新到文件
        if len(self.stats_buffer) >= self.stats_flush_interval:
            self.flush_stats_to_files()
    
    def flush_stats_to_files(self):
        """将缓存的统计数据刷新到文件"""
        if not self.stats_buffer:
            return
        
        # 基因组统计
        genome_stats_path = self.stats_dir / "genome_stats.csv"
        with open(genome_stats_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for record in self.stats_buffer:
                genome_stats = record['genome_stats']
                writer.writerow([
                    record['generation'],
                    record['timestamp'],
                    genome_stats['total_size'],
                    genome_stats['gene_count'],
                    genome_stats['core_genes'],
                    genome_stats['hgt_genes'],
                    genome_stats['total_mutations'],
                    genome_stats['total_hgt_events'],
                    genome_stats['total_recombination_events'],
                    genome_stats['avg_gene_length'],
                    genome_stats.get('gc_content', 0.5)
                ])
        
        # 进化事件统计
        evolution_stats_path = self.stats_dir / "evolution_stats.csv"
        with open(evolution_stats_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for record in self.stats_buffer:
                gen_stats = record['generation_stats']
                writer.writerow([
                    record['generation'],
                    record['timestamp'],
                    gen_stats.get('mutations', 0),
                    gen_stats.get('hgt_events', 0),
                    gen_stats.get('recombination_events', 0),
                    gen_stats.get('genes_lost', 0),
                    gen_stats.get('processing_mode', 'unknown'),
                    gen_stats.get('wall_clock_time', 0)
                ])
        
        # 性能统计
        if self.performance_buffer:
            performance_stats_path = self.stats_dir / "performance_stats.csv"
            with open(performance_stats_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for record in self.performance_buffer:
                    writer.writerow([
                        record.get('generation', 0),
                        record.get('timestamp', ''),
                        record.get('processing_mode', ''),
                        record.get('wall_clock_time', 0),
                        record.get('parallel_processing_time', 0),
                        record.get('total_processing_time', 0),
                        record.get('chunks_processed', 0),
                        record.get('chunk_size', 0)
                    ])
        
        # 清空缓存
        self.stats_buffer.clear()
        self.performance_buffer.clear()
    
    def save_ani_identities(self, identities_data: Dict):
        """保存ANI身份数据集"""
        self.ani_identities.append({
            'timestamp': datetime.now().isoformat(),
            'data': self._make_serializable(identities_data)
        })
        
        # 保存到文件
        ani_path = self.analysis_dir / "ani_identities.json"
        with open(ani_path, 'w', encoding='utf-8') as f:
            json.dump(self.ani_identities, f, indent=2, ensure_ascii=False)
    
    def save_conservation_analysis(self, analysis_results: Dict):
        """保存保守性分析结果"""
        analysis_path = self.analysis_dir / "conservation_analysis.json"
        
        serializable_results = self._make_serializable(analysis_results)
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"🔬 Conservation analysis saved to: {analysis_path}")
    
    def save_final_summary(self, summary_data: Dict):
        """保存最终分析摘要"""
        summary_data['completion_time'] = datetime.now().isoformat()
        summary_data['run_id'] = self.run_id
        
        summary_path = self.analysis_dir / "final_summary.json"
        
        serializable_summary = self._make_serializable(summary_data)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Final summary saved to: {summary_path}")
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def evolve_one_generation(self, genome: Genome) -> Dict:
        """重写父类方法，添加持久化功能"""
        # 调用父类方法
        generation_stats = super().evolve_one_generation(genome)
        
        # 保存快照
        self.save_snapshot(genome, genome.generation)
        
        # 保存统计数据
        self.save_generation_stats(genome, generation_stats)
        
        # 记录性能数据
        if 'parallel_processing_time' in generation_stats:
            self.performance_buffer.append({
                'generation': genome.generation,
                'timestamp': datetime.now().isoformat(),
                **generation_stats
            })
        
        # 记录进化事件
        if self.save_detailed_events:
            if generation_stats.get('mutations', 0) > 0:
                self.log_evolution_event('mutations', {
                    'generation': genome.generation,
                    'count': generation_stats['mutations']
                })
            
            if generation_stats.get('hgt_events', 0) > 0:
                self.log_evolution_event('hgt_events', {
                    'generation': genome.generation,
                    'count': generation_stats['hgt_events']
                })
            
            if generation_stats.get('recombination_events', 0) > 0:
                self.log_evolution_event('recombination', {
                    'generation': genome.generation,
                    'count': generation_stats['recombination_events']
                })
            
            if generation_stats.get('genes_lost', 0) > 0:
                self.log_evolution_event('gene_loss', {
                    'generation': genome.generation,
                    'count': generation_stats['genes_lost']
                })
        
        return generation_stats
    
    def simulate_evolution(self, 
                          initial_genome: Genome, 
                          generations: int,
                          save_snapshots: bool = True,
                          snapshot_interval: int = 100) -> Tuple[Genome, List[Dict]]:
        """重写父类方法，添加完整的持久化支持"""
        
        # 初始化存储（如果还没有初始化）
        if not self.storage_initialized:
            config = {
                'generations': generations,
                'save_snapshots': save_snapshots,
                'snapshot_interval': snapshot_interval,
                'engine_config': self.config
            }
            self.initialize_storage(config, initial_genome)
        
        print("🗄️  PERSISTENT PROKARYOTIC GENOME EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📁 Data will be saved to: {self.run_dir}")
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"📸 Snapshots: Every {self.snapshot_interval} generations")
        print("=" * 80)
        
        # 调用父类方法进行模拟
        final_genome, snapshots = super().simulate_evolution(
            initial_genome, generations, save_snapshots, snapshot_interval
        )
        
        # 保存最终基因组
        self.save_evolved_genome(final_genome)
        
        # 刷新所有缓存数据
        self.flush_stats_to_files()
        
        # 保存最终摘要
        final_summary = {
            'run_info': {
                'run_id': self.run_id,
                'generations': generations,
                'start_genome_size': initial_genome.size,
                'final_genome_size': final_genome.size,
                'start_gene_count': initial_genome.gene_count,
                'final_gene_count': final_genome.gene_count
            },
            'evolution_summary': self.get_evolution_summary(final_genome),
            'performance_analysis': self.get_performance_analysis(),
            'storage_info': {
                'snapshots_saved': len(list(self.snapshots_dir.glob('*.json*'))),
                'events_logged': self.save_detailed_events,
                'data_compressed': self.compress_data,
                'sequences_saved': self.save_sequences
            }
        }
        
        self.save_final_summary(final_summary)
        
        # 关闭事件日志文件
        for f in self.event_files.values():
            if not f.closed:
                f.close()
        
        print(f"\n🎉 PERSISTENT SIMULATION COMPLETED!")
        print(f"📁 All data saved to: {self.run_dir}")
        print(f"📊 Snapshots saved: {len(list(self.snapshots_dir.glob('*.json*')))}")
        print(f"📈 Statistics files: {len(list(self.stats_dir.glob('*.csv')))}")
        print(f"📝 Event logs: {len(list(self.events_dir.glob('*.jsonl')))}")
        print("=" * 80)
        
        return final_genome, snapshots
    
    def get_run_directory(self) -> Path:
        """获取当前运行的数据目录"""
        return self.run_dir
    
    def cleanup_parallel_resources(self):
        """清理资源时也关闭文件句柄"""
        # 关闭事件日志文件
        for f in self.event_files.values():
            if not f.closed:
                f.close()
        
        # 刷新缓存数据
        self.flush_stats_to_files()
        
        # 调用父类清理方法
        super().cleanup_parallel_resources()