#!/usr/bin/env python3
"""
优化的并行进化引擎
解决并行处理性能瓶颈问题

主要优化：
1. 减少进程初始化开销
2. 优化分块策略
3. 减少进程间通信
4. 预计算和缓存优化

Version: 1.1.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

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


# 全局变量用于工作进程初始化（避免重复创建）
_worker_engines = None


def init_worker_process(evolution_params: Dict):
    """
    工作进程初始化函数 - 只在进程启动时调用一次
    避免每次任务都重新创建进化机制实例
    """
    global _worker_engines
    
    try:
        # 创建进化机制实例（每个进程只创建一次）
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
        
        _worker_engines = {
            'point_mutation': point_mutation,
            'hgt': hgt,
            'recombination': recombination
        }
        
    except Exception as e:
        print(f"❌ Worker initialization error: {e}")
        _worker_engines = None


def optimized_evolve_genes_chunk(chunk_data: Tuple[List[Gene], int]) -> Tuple[List[Gene], Dict]:
    """
    优化的基因块进化函数
    使用预初始化的进化机制实例
    """
    global _worker_engines
    
    genes_chunk, process_id = chunk_data
    
    # 检查工作进程是否正确初始化
    if _worker_engines is None:
        return genes_chunk, {
            'process_id': process_id,
            'genes_processed': len(genes_chunk),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0,
            'processing_time': 0,
            'error': 'Worker not initialized'
        }
    
    # 统计信息
    chunk_stats = {
        'process_id': process_id,
        'genes_processed': len(genes_chunk),
        'mutations': 0,
        'hgt_events': 0,
        'recombination_events': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # 创建临时基因组用于处理这个块
        temp_genome = Genome(genes_chunk)
        
        # 使用预初始化的进化机制
        engines = _worker_engines
        
        # 应用进化机制
        mutations = engines['point_mutation'].apply_mutations(temp_genome, generations=1)
        chunk_stats['mutations'] = mutations
        
        hgt_events = engines['hgt'].apply_hgt(temp_genome, generations=1)
        chunk_stats['hgt_events'] = hgt_events
        
        recombination_events = engines['recombination'].apply_recombination(temp_genome, generations=1)
        chunk_stats['recombination_events'] = recombination_events
        
        # 返回进化后的基因
        evolved_genes = temp_genome.genes
        
    except Exception as e:
        print(f"❌ Error in process {process_id}: {e}")
        chunk_stats['error'] = str(e)
        evolved_genes = genes_chunk  # 返回原始基因
    
    chunk_stats['processing_time'] = time.time() - start_time
    
    return evolved_genes, chunk_stats


class OptimizedParallelEvolutionEngine:
    """
    优化的并行进化引擎
    解决原有并行实现的性能瓶颈
    """
    
    def __init__(self, 
                 # 基本进化参数
                 mutation_rate: float = 1e-6,
                 hgt_rate: float = 1e-8,
                 recombination_rate: float = 1e-9,
                 min_similarity_for_recombination: float = 0.85,  # 修正：更严格的重组相似度要求
                 
                 # 基因丢失参数
                 enable_gene_loss: bool = True,
                 loss_rate: float = 1e-6,
                 core_gene_protection: float = 0.95,
                 hgt_gene_loss_multiplier: float = 10.0,
                 min_genome_size: int = 1000,
                 min_core_genes: int = 800,
                 optimal_genome_size: int = 3000,
                 
                 # 优化的并行处理参数
                 enable_parallel: bool = True,
                 num_processes: Optional[int] = None,
                 min_chunk_size: int = 100,  # 最小分块大小
                 max_chunk_size: int = 1000,  # 最大分块大小
                 parallel_threshold: int = 500,  # 启用并行的基因数阈值
                 
                 # 性能优化参数
                 enable_optimization: bool = True):
        """
        初始化优化的并行进化引擎
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
            'min_chunk_size': min_chunk_size,
            'max_chunk_size': max_chunk_size,
            'parallel_threshold': parallel_threshold,
            'enable_optimization': enable_optimization
        }
        
        # 初始化进化机制
        self._initialize_mechanisms()
        
        # 进化历史记录
        self.evolution_history = []
        
        # 进程池（重用以避免重复创建）
        self._pool = None
        
        # 打印初始化信息
        self._print_initialization_info()
    
    def _initialize_mechanisms(self):
        """初始化所有进化机制"""
        
        # 1. 点突变引擎（优化版）
        self.point_mutation = OptimizedPointMutationEngine(
            mutation_rate=self.config['mutation_rate'],
            enable_transition_bias=True,
            enable_hotspots=True
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
        print(f"🚀 Optimized Parallel Evolution Engine initialized:")
        print(f"   Mechanisms: Point mutations, HGT, Recombination" + 
              (", Gene loss" if self.config['enable_gene_loss'] else ""))
        print(f"   Optimization: {'Enabled' if self.config['enable_optimization'] else 'Disabled'}")
        print(f"   Parallel processing: {'Enabled' if self.config['enable_parallel'] else 'Disabled'}")
        if self.config['enable_parallel']:
            print(f"   Processes: {self.config['num_processes']}")
            print(f"   Chunk size: {self.config['min_chunk_size']}-{self.config['max_chunk_size']} genes")
            print(f"   Parallel threshold: {self.config['parallel_threshold']} genes")
        if self.config['enable_gene_loss']:
            print(f"   Gene loss rate: {self.config['loss_rate']}")
    
    def _should_use_parallel(self, genome: Genome) -> bool:
        """判断是否应该使用并行处理"""
        return (self.config['enable_parallel'] and 
                genome.gene_count >= self.config['parallel_threshold'] and
                self.config['num_processes'] > 1)
    
    def _calculate_optimal_chunk_size(self, total_genes: int) -> int:
        """
        计算最优的基因分块大小
        优化策略：减少分块数量，增加每块大小
        """
        min_size = self.config['min_chunk_size']
        max_size = self.config['max_chunk_size']
        num_processes = self.config['num_processes']
        
        # 目标：每个进程处理2-4个块
        target_chunks_per_process = 3
        target_total_chunks = num_processes * target_chunks_per_process
        
        # 计算理想分块大小
        ideal_chunk_size = max(min_size, total_genes // target_total_chunks)
        
        # 限制在合理范围内
        optimal_chunk_size = min(max_size, max(min_size, ideal_chunk_size))
        
        return optimal_chunk_size
    
    def _split_genes_into_chunks(self, genes: List[Gene]) -> List[List[Gene]]:
        """将基因列表分割成适合并行处理的块"""
        chunk_size = self._calculate_optimal_chunk_size(len(genes))
        chunks = []
        
        for i in range(0, len(genes), chunk_size):
            chunk = genes[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _get_or_create_pool(self) -> Pool:
        """获取或创建进程池（重用以提高性能）"""
        if self._pool is None:
            # 准备进化参数
            evolution_params = {
                'mutation_rate': self.config['mutation_rate'],
                'hgt_rate': self.config['hgt_rate'],
                'recombination_rate': self.config['recombination_rate'],
                'min_similarity_for_recombination': self.config['min_similarity_for_recombination'],
                'enable_transition_bias': True,
                'enable_hotspots': True
            }
            
            # 创建进程池并初始化工作进程
            self._pool = Pool(
                processes=self.config['num_processes'],
                initializer=init_worker_process,
                initargs=(evolution_params,)
            )
        
        return self._pool
    
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
        """优化的并行进化一代"""
        generation_start_time = time.time()
        
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
        
        parallel_start_time = time.time()
        
        # 获取进程池
        pool = self._get_or_create_pool()
        
        # 准备任务数据
        chunk_args = [(chunk, i) for i, chunk in enumerate(gene_chunks)]
        
        # 并行执行（使用预初始化的工作进程）
        try:
            results = pool.map(optimized_evolve_genes_chunk, chunk_args)
        except Exception as e:
            print(f"❌ Parallel processing error: {e}")
            # 回退到串行处理
            return self.evolve_one_generation_serial(genome)
        
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
        
        # 应用基因丢失（在主进程中处理）
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
        processing_mode = "OPTIMIZED PARALLEL" if use_parallel else "SERIAL"
        
        if show_progress:
            print(f"🚀 Starting {processing_mode} evolution: {generations:,} generations")
            if use_parallel:
                print(f"   Processes: {self.config['num_processes']}")
                chunk_size = self._calculate_optimal_chunk_size(genome.gene_count)
                estimated_chunks = max(1, genome.gene_count // chunk_size)
                print(f"   Chunk strategy: ~{chunk_size} genes/chunk (~{estimated_chunks} chunks)")
            print("=" * 80)
        
        # 确定显示频率
        if generations <= 50:
            display_freq = 5
        elif generations <= 100:
            display_freq = 10
        elif generations <= 1000:
            display_freq = 50
        else:
            display_freq = 100
        
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
                
                # 并行性能信息
                if use_parallel and 'parallel_processing_time' in gen_stats:
                    parallel_efficiency = (gen_stats['parallel_processing_time'] / 
                                         gen_stats['total_processing_time']) * 100 if gen_stats['total_processing_time'] > 0 else 0
                    parallel_info = f" | Eff: {parallel_efficiency:.1f}%"
                    genome_info += parallel_info
                
                print(f"\r[{bar}] {progress*100:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"{1/avg_time_per_gen:.1f} gen/s | ETA: {estimated_remaining/60:.1f}min | "
                      f"{genome_info}", end="", flush=True)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n\n🎉 {processing_mode} evolution completed!")
            print(f"Total time: {total_time/60:.2f} minutes")
            print(f"Average speed: {generations/total_time:.2f} generations/second")
            
            if use_parallel:
                avg_parallel_efficiency = np.mean([
                    (h['parallel_processing_time'] / h['total_processing_time']) * 100 
                    for h in history if 'parallel_processing_time' in h and h['total_processing_time'] > 0
                ])
                print(f"Average parallel efficiency: {avg_parallel_efficiency:.1f}%")
            
            print("=" * 80)
        
        self.evolution_history.extend(history)
        return history
    
    def cleanup(self):
        """清理资源"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        print("🧹 Parallel resources cleaned up")
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        self.cleanup()