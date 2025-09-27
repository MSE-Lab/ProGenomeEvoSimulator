#!/usr/bin/env python3
"""
Unified Evolution Engine
统一的进化引擎 - 集成所有功能的完整解决方案
包含：优化算法、并行处理、基因丢失、保守性分析等所有功能

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

__version__ = "1.0.0"
__author__ = "ProGenomeEvoSimulator Team"
__date__ = "2025-09-27"

import time
import copy
import multiprocessing as mp
from multiprocessing import Pool, Manager
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from functools import partial

from core.genome import Genome, Gene
from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
from mechanisms.horizontal_transfer import HorizontalGeneTransfer
from mechanisms.homologous_recombination import HomologousRecombination
from mechanisms.gene_loss import GeneLossEngine


class UnifiedEvolutionEngine:
    """
    统一进化引擎 - 集成所有进化机制和优化功能
    
    功能特性：
    - 点突变（优化算法）
    - 横向基因转移
    - 同源重组
    - 基因丢失
    - 并行处理
    - 性能优化
    - 详细统计
    """
    
    def __init__(self, 
                 # 基本进化参数
                 mutation_rate: float = 1e-6,
                 hgt_rate: float = 1e-8,
                 recombination_rate: float = 1e-9,
                 min_similarity_for_recombination: float = 0.7,
                 
                 # 基因丢失参数
                 enable_gene_loss: bool = True,
                 loss_rate: float = 1e-6,
                 core_gene_protection: float = 0.95,
                 hgt_gene_loss_multiplier: float = 10.0,
                 min_genome_size: int = 1000,
                 min_core_genes: int = 800,
                 optimal_genome_size: int = 3000,
                 
                 # 并行处理参数
                 enable_parallel: bool = True,
                 num_processes: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 parallel_threshold: int = 500,  # 基因数超过此值时启用并行
                 
                 # 性能优化参数
                 enable_optimization: bool = True,
                 enable_progress_sharing: bool = True):
        """
        初始化统一进化引擎
        
        Args:
            # 基本进化参数
            mutation_rate: 点突变率
            hgt_rate: 横向基因转移率
            recombination_rate: 同源重组率
            min_similarity_for_recombination: 重组所需最小相似度
            
            # 基因丢失参数
            enable_gene_loss: 是否启用基因丢失
            loss_rate: 基因丢失率
            core_gene_protection: 核心基因保护系数
            hgt_gene_loss_multiplier: HGT基因丢失倍数
            min_genome_size: 最小基因组大小
            min_core_genes: 最小核心基因数
            optimal_genome_size: 最优基因组大小
            
            # 并行处理参数
            enable_parallel: 是否启用并行处理
            num_processes: 并行进程数
            chunk_size: 基因分块大小
            parallel_threshold: 启用并行的基因数阈值
            
            # 性能优化参数
            enable_optimization: 是否启用性能优化
            enable_progress_sharing: 是否启用进度共享
        """
        
        # 存储配置参数
        self.config = {
            'mutation_rate': mutation_rate,
            'hgt_rate': hgt_rate,
            'recombination_rate': recombination_rate,
            'min_similarity_for_recombination': min_similarity_for_recombination,
            'enable_gene_loss': enable_gene_loss,
            'loss_rate': loss_rate,
            'core_gene_protection': core_gene_protection,
            'hgt_gene_loss_multiplier': hgt_gene_loss_multiplier,
            'min_genome_size': min_genome_size,
            'min_core_genes': min_core_genes,
            'optimal_genome_size': optimal_genome_size,
            'enable_parallel': enable_parallel,
            'num_processes': num_processes or mp.cpu_count(),
            'chunk_size': chunk_size,
            'parallel_threshold': parallel_threshold,
            'enable_optimization': enable_optimization,
            'enable_progress_sharing': enable_progress_sharing
        }
        
        # 初始化进化机制
        self._initialize_mechanisms()
        
        # 进化历史记录
        self.evolution_history = []
        
        # 打印初始化信息
        self._print_initialization_info()
    
    def _initialize_mechanisms(self):
        """初始化所有进化机制"""
        
        # 1. 点突变引擎（优化版）
        if self.config['enable_optimization']:
            self.point_mutation = OptimizedPointMutationEngine(
                mutation_rate=self.config['mutation_rate'],
                enable_transition_bias=True,
                enable_hotspots=True
            )
        else:
            # 如果需要，可以在这里添加基础版本
            self.point_mutation = OptimizedPointMutationEngine(
                mutation_rate=self.config['mutation_rate']
            )
        
        # 2. 横向基因转移
        self.hgt = HorizontalGeneTransfer(self.config['hgt_rate'])
        
        # 3. 同源重组
        self.recombination = HomologousRecombination(
            self.config['recombination_rate'],
            self.config['min_similarity_for_recombination']
        )
        
        # 4. 基因丢失（可选）
        if self.config['enable_gene_loss']:
            self.gene_loss = GeneLossEngine(
                loss_rate=self.config['loss_rate'],
                core_gene_protection=self.config['core_gene_protection'],
                hgt_gene_loss_multiplier=self.config['hgt_gene_loss_multiplier'],
                min_genome_size=self.config['min_genome_size'],
                min_core_genes=self.config['min_core_genes'],
                optimal_genome_size=self.config['optimal_genome_size']
            )
        else:
            self.gene_loss = None
    
    def _print_initialization_info(self):
        """打印初始化信息"""
        print(f"🧬 Unified Evolution Engine initialized:")
        print(f"   Mechanisms: Point mutations, HGT, Recombination" + 
              (", Gene loss" if self.config['enable_gene_loss'] else ""))
        print(f"   Optimization: {'Enabled' if self.config['enable_optimization'] else 'Disabled'}")
        print(f"   Parallel processing: {'Enabled' if self.config['enable_parallel'] else 'Disabled'}")
        if self.config['enable_parallel']:
            print(f"   Processes: {self.config['num_processes']}")
            print(f"   Parallel threshold: {self.config['parallel_threshold']} genes")
        if self.config['enable_gene_loss']:
            print(f"   Gene loss rate: {self.config['loss_rate']}")
            print(f"   Core protection: {self.config['core_gene_protection']*100:.1f}%")
    
    def _should_use_parallel(self, genome: Genome) -> bool:
        """判断是否应该使用并行处理"""
        return (self.config['enable_parallel'] and 
                genome.gene_count >= self.config['parallel_threshold'] and
                self.config['num_processes'] > 1)
    
    def _calculate_optimal_chunk_size(self, total_genes: int) -> int:
        """计算最优的基因分块大小"""
        if self.config['chunk_size']:
            return self.config['chunk_size']
        
        # 自动计算最优分块大小
        base_chunk_size = max(1, total_genes // (self.config['num_processes'] * 4))
        
        if total_genes < 100:
            return max(1, total_genes // self.config['num_processes'])
        elif total_genes < 1000:
            return min(50, base_chunk_size)
        else:
            return min(200, base_chunk_size)
    
    def _split_genes_into_chunks(self, genes: List[Gene]) -> List[List[Gene]]:
        """将基因列表分割成适合并行处理的块"""
        chunk_size = self._calculate_optimal_chunk_size(len(genes))
        chunks = []
        
        for i in range(0, len(genes), chunk_size):
            chunk = genes[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def evolve_one_generation_serial(self, genome: Genome) -> Dict:
        """串行进化一代"""
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0,
            'genes_lost': 0,
            'processing_mode': 'serial'
        }
        
        # 1. 应用点突变
        mutations = self.point_mutation.apply_mutations(genome, generations=1)
        generation_stats['mutations'] = mutations
        
        # 2. 应用横向基因转移
        hgt_events = self.hgt.apply_hgt(genome, generations=1)
        generation_stats['hgt_events'] = hgt_events
        
        # 3. 应用同源重组
        recombination_events = self.recombination.apply_recombination(genome, generations=1)
        generation_stats['recombination_events'] = recombination_events
        
        # 4. 应用基因丢失（如果启用）
        if self.gene_loss:
            genes_lost = self.gene_loss.apply_gene_loss(genome, generations=1)
            generation_stats['genes_lost'] = genes_lost
        
        # 更新代数
        genome.generation += 1
        
        # 记录最终统计
        generation_stats['final_stats'] = genome.get_statistics()
        
        return generation_stats
    
    def evolve_one_generation_parallel(self, genome: Genome) -> Dict:
        """并行进化一代"""
        generation_start_time = time.time()
        
        # 准备进化参数
        evolution_params = {
            'mutation_rate': self.config['mutation_rate'],
            'hgt_rate': self.config['hgt_rate'],
            'recombination_rate': self.config['recombination_rate'],
            'min_similarity_for_recombination': self.config['min_similarity_for_recombination'],
            'enable_transition_bias': True,
            'enable_hotspots': True
        }
        
        # 分割基因到不同的块
        gene_chunks = self._split_genes_into_chunks(genome.genes)
        
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'total_mutations': 0,
            'total_hgt_events': 0,
            'total_recombination_events': 0,
            'genes_lost': 0,
            'chunks_processed': len(gene_chunks),
            'parallel_processing_time': 0,
            'processing_mode': 'parallel'
        }
        
        # 设置共享进度计数器
        if self.config['enable_progress_sharing']:
            manager = Manager()
            shared_progress = manager.Value('i', 0)
        else:
            shared_progress = None
        
        parallel_start_time = time.time()
        
        # 并行处理所有基因块
        with Pool(processes=self.config['num_processes']) as pool:
            # 创建部分函数
            process_func = partial(
                evolve_genes_chunk_worker,
                evolution_params=evolution_params,
                shared_progress=shared_progress
            )
            
            # 为每个块添加进程ID
            chunk_args = [(chunk, i) for i, chunk in enumerate(gene_chunks)]
            
            # 并行执行
            results = pool.map(process_func, chunk_args)
        
        parallel_end_time = time.time()
        generation_stats['parallel_processing_time'] = parallel_end_time - parallel_start_time
        
        # 合并结果
        evolved_genes = []
        for evolved_chunk, chunk_stats in results:
            evolved_genes.extend(evolved_chunk)
            generation_stats['total_mutations'] += chunk_stats['mutations']
            generation_stats['total_hgt_events'] += chunk_stats['hgt_events']
            generation_stats['total_recombination_events'] += chunk_stats['recombination_events']
        
        # 更新基因组
        genome.genes = evolved_genes
        genome.total_mutations += generation_stats['total_mutations']
        genome.total_hgt_events += generation_stats['total_hgt_events']
        genome.total_recombination_events += generation_stats['total_recombination_events']
        
        # 应用基因丢失（在主进程中处理，避免并行复杂性）
        if self.gene_loss:
            genes_lost = self.gene_loss.apply_gene_loss(genome, generations=1)
            generation_stats['genes_lost'] = genes_lost
        
        # 更新代数
        genome.generation += 1
        
        # 记录最终统计
        generation_stats['final_stats'] = genome.get_statistics()
        generation_stats['total_processing_time'] = time.time() - generation_start_time
        
        return generation_stats
    
    def evolve_one_generation(self, genome: Genome) -> Dict:
        """进化一代（自动选择串行或并行）"""
        if self._should_use_parallel(genome):
            return self.evolve_one_generation_parallel(genome)
        else:
            return self.evolve_one_generation_serial(genome)
    
    def evolve_multiple_generations(self, genome: Genome, generations: int, 
                                  show_progress: bool = True) -> List[Dict]:
        """进化多代"""
        history = []
        start_time = time.time()
        
        # 确定处理模式
        use_parallel = self._should_use_parallel(genome)
        processing_mode = "PARALLEL" if use_parallel else "SERIAL"
        
        if show_progress:
            # 确定显示频率
            if generations <= 50:
                display_freq = 5
            elif generations <= 100:
                display_freq = 10
            elif generations <= 1000:
                display_freq = 50
            else:
                display_freq = 100
            
            print(f"Starting {processing_mode} evolution simulation: {generations:,} generations")
            if use_parallel:
                print(f"Parallel processes: {self.config['num_processes']}")
            print(f"Progress updates every {display_freq} generation(s)")
            print("=" * 80)
        else:
            display_freq = max(1, generations // 10)
        
        for gen in range(generations):
            gen_start_time = time.time()
            
            # 进化一代
            gen_stats = self.evolve_one_generation(genome)
            
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            gen_stats['wall_clock_time'] = gen_duration
            
            history.append(gen_stats)
            
            # 显示进度
            if show_progress and ((gen + 1) % display_freq == 0 or gen == 0):
                elapsed_total = time.time() - start_time
                avg_time_per_gen = elapsed_total / (gen + 1)
                remaining_gens = generations - (gen + 1)
                estimated_remaining = remaining_gens * avg_time_per_gen
                
                # 进度条
                progress = (gen + 1) / generations
                bar_width = 30
                filled_width = int(bar_width * progress)
                bar = '█' * filled_width + '░' * (bar_width - filled_width)
                
                # 基因组信息
                genome_info = (f"Genes: {genome.gene_count:,} | "
                             f"Events: {genome.total_mutations:,}mut "
                             f"{genome.total_hgt_events:,}HGT "
                             f"{genome.total_recombination_events:,}rec")
                
                # 基因丢失信息
                if self.gene_loss:
                    loss_stats = self.gene_loss.get_loss_statistics(genome)
                    loss_info = f" {loss_stats['total_genes_lost']:,}lost"
                    genome_info += loss_info
                
                # 并行性能信息
                if use_parallel and 'parallel_processing_time' in gen_stats:
                    parallel_efficiency = (gen_stats['parallel_processing_time'] / 
                                         gen_stats['total_processing_time']) * 100 if gen_stats['total_processing_time'] > 0 else 0
                    parallel_info = f" | Parallel: {parallel_efficiency:.1f}%"
                    genome_info += parallel_info
                
                print(f"\r[{bar}] {progress*100:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"{1/avg_time_per_gen:.1f} gen/s | ETA: {estimated_remaining/60:.1f}min | "
                      f"{genome_info}", end="", flush=True)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n\n🚀 {processing_mode} evolution completed!")
            print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
            print(f"Average speed: {generations/total_time:.2f} generations/second")
            
            # 显示统计摘要
            if self.gene_loss:
                loss_stats = self.gene_loss.get_loss_statistics(genome)
                print(f"Gene loss: {loss_stats['total_genes_lost']} genes lost "
                      f"({loss_stats['avg_total_loss_per_generation']:.3f}/gen)")
            
            if use_parallel:
                avg_parallel_efficiency = np.mean([
                    (h['parallel_processing_time'] / h['total_processing_time']) * 100 
                    for h in history if 'parallel_processing_time' in h and h['total_processing_time'] > 0
                ])
                print(f"Parallel efficiency: {avg_parallel_efficiency:.1f}%")
            
            print("=" * 80)
        
        self.evolution_history.extend(history)
        return history
    
    def simulate_evolution(self, 
                          initial_genome: Genome, 
                          generations: int,
                          save_snapshots: bool = True,
                          snapshot_interval: int = 100) -> Tuple[Genome, List[Dict]]:
        """完整的进化模拟"""
        
        print("🧬 UNIFIED PROKARYOTIC GENOME EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"📸 Snapshots: {'Enabled' if save_snapshots else 'Disabled'} (interval: {snapshot_interval})")
        
        # 显示启用的功能
        features = []
        if self.config['enable_optimization']:
            features.append("Optimized algorithms")
        if self.config['enable_parallel']:
            features.append(f"Parallel processing ({self.config['num_processes']} cores)")
        if self.config['enable_gene_loss']:
            features.append("Gene loss")
        
        print(f"⚡ Features: {', '.join(features)}")
        print(f"🧬 Mechanisms: Point mutations, HGT, Homologous recombination" + 
              (", Gene loss" if self.config['enable_gene_loss'] else ""))
        print("=" * 80)
        
        # 创建基因组副本
        evolving_genome = initial_genome.copy()
        simulation_start_time = time.time()
        
        # 记录初始状态
        snapshots = []
        if save_snapshots:
            initial_summary = self.get_evolution_summary(evolving_genome)
            initial_summary['snapshot_generation'] = 0
            snapshots.append(initial_summary)
        
        # 进化过程
        evolution_history = self.evolve_multiple_generations(
            evolving_genome, generations, show_progress=True
        )
        
        # 保存快照
        if save_snapshots:
            print(f"📸 Saving snapshots every {snapshot_interval} generations...")
            for i in range(0, len(evolution_history), snapshot_interval):
                if i < len(evolution_history):
                    snapshot = self.get_evolution_summary(evolving_genome)
                    snapshot['snapshot_generation'] = evolution_history[i]['generation']
                    snapshots.append(snapshot)
        
        # 最终总结
        total_simulation_time = time.time() - simulation_start_time
        
        print(f"\n🎉 UNIFIED SIMULATION COMPLETED!")
        print(f"🧬 Final genome: {evolving_genome.gene_count:,} genes, {evolving_genome.size:,} bp")
        print(f"📈 Changes: {evolving_genome.size - initial_genome.size:+,} bp, "
              f"{evolving_genome.gene_count - initial_genome.gene_count:+,} genes")
        
        # 性能总结
        use_parallel = self._should_use_parallel(initial_genome)
        if use_parallel:
            print(f"⚡ Performance: {self.config['num_processes']} processes, "
                  f"{generations/total_simulation_time:.2f} gen/s")
        else:
            print(f"⚡ Performance: Serial processing, {generations/total_simulation_time:.2f} gen/s")
        
        return evolving_genome, snapshots
    
    def get_evolution_summary(self, genome: Genome) -> Dict:
        """获取进化总结"""
        summary = {
            'genome_stats': genome.get_statistics(),
            'mutation_stats': self.point_mutation.get_mutation_statistics(genome),
            'hgt_stats': self.hgt.get_hgt_statistics(genome),
            'recombination_stats': self.recombination.get_recombination_statistics(genome),
            'engine_config': self.config.copy()
        }
        
        # 添加基因丢失统计
        if self.gene_loss:
            summary['gene_loss_stats'] = self.gene_loss.get_loss_statistics(genome)
            summary['gene_loss_patterns'] = self.gene_loss.analyze_loss_patterns(genome)
        
        return summary
    
    def get_performance_analysis(self) -> Dict:
        """获取性能分析"""
        if not self.evolution_history:
            return {'error': 'No evolution history available'}
        
        analysis = {
            'total_generations': len(self.evolution_history),
            'processing_modes': {},
            'average_times': {},
            'engine_config': self.config.copy()
        }
        
        # 分析处理模式
        serial_gens = [h for h in self.evolution_history if h.get('processing_mode') == 'serial']
        parallel_gens = [h for h in self.evolution_history if h.get('processing_mode') == 'parallel']
        
        analysis['processing_modes'] = {
            'serial_generations': len(serial_gens),
            'parallel_generations': len(parallel_gens)
        }
        
        # 计算平均时间
        if serial_gens:
            analysis['average_times']['serial'] = np.mean([h['wall_clock_time'] for h in serial_gens])
        
        if parallel_gens:
            analysis['average_times']['parallel'] = np.mean([h['wall_clock_time'] for h in parallel_gens])
            
            # 并行效率分析
            parallel_efficiencies = []
            for h in parallel_gens:
                if 'parallel_processing_time' in h and h['total_processing_time'] > 0:
                    eff = (h['parallel_processing_time'] / h['total_processing_time']) * 100
                    parallel_efficiencies.append(eff)
            
            if parallel_efficiencies:
                analysis['parallel_efficiency'] = {
                    'average': np.mean(parallel_efficiencies),
                    'min': np.min(parallel_efficiencies),
                    'max': np.max(parallel_efficiencies)
                }
        
        return analysis
    
    def clear_caches(self):
        """清理所有缓存"""
        if hasattr(self.point_mutation, 'clear_cache'):
            self.point_mutation.clear_cache()
        print("🧹 Caches cleared for memory optimization")


def evolve_genes_chunk_worker(chunk_args: Tuple[List[Gene], int], 
                            evolution_params: Dict,
                            shared_progress: Optional[Any] = None) -> Tuple[List[Gene], Dict]:
    """
    工作进程函数：处理基因块的进化
    """
    genes_chunk, process_id = chunk_args
    
    # 直接在工作进程中处理
    from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
    from mechanisms.horizontal_transfer import HorizontalGeneTransfer
    from mechanisms.homologous_recombination import HomologousRecombination
    from core.genome import Genome
    
    # 创建本地进化机制实例
    point_mutation = OptimizedPointMutationEngine(
        mutation_rate=evolution_params['mutation_rate'],
        enable_transition_bias=evolution_params.get('enable_transition_bias', True),
        enable_hotspots=evolution_params.get('enable_hotspots', True)
    )
    
    hgt = HorizontalGeneTransfer(evolution_params['hgt_rate'])
    recombination = HomologousRecombination(
        evolution_params['recombination_rate'],
        evolution_params['min_similarity_for_recombination']
    )
    
    # 统计信息
    chunk_stats = {
        'process_id': process_id,
        'genes_processed': len(genes_chunk),
        'mutations': 0,
        'hgt_events': 0,
        'recombination_events': 0,
        'processing_time': 0
    }
    
    import time
    start_time = time.time()
    
    # 创建临时基因组用于处理这个块
    temp_genome = Genome(genes_chunk)
    
    # 应用进化机制
    try:
        # 点突变
        mutations = point_mutation.apply_mutations(temp_genome, generations=1)
        chunk_stats['mutations'] = mutations
        
        # HGT
        hgt_events = hgt.apply_hgt(temp_genome, generations=1)
        chunk_stats['hgt_events'] = hgt_events
        
        # 同源重组
        recombination_events = recombination.apply_recombination(temp_genome, generations=1)
        chunk_stats['recombination_events'] = recombination_events
        
    except Exception as e:
        print(f"❌ Error in process {process_id}: {e}")
        chunk_stats['error'] = str(e)
    
    chunk_stats['processing_time'] = time.time() - start_time
    
    # 更新共享进度计数器
    if shared_progress is not None:
        try:
            with shared_progress.get_lock():
                shared_progress.value += len(genes_chunk)
        except:
            pass  # 忽略共享进度更新错误
    
    return temp_genome.genes, chunk_stats