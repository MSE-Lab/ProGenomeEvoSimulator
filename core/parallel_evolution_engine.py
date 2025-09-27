#!/usr/bin/env python3
"""
Parallel Evolution Engine
使用多进程并行化的进化引擎，专为多CPU服务器环境优化
"""

import time
import copy
import multiprocessing as mp
from multiprocessing import Pool, Manager, Value, Lock
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from functools import partial
import os

from core.genome import Genome, Gene
from core.evolution_engine_optimized import OptimizedEvolutionEngine
from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
from mechanisms.horizontal_transfer import HorizontalGeneTransfer
from mechanisms.homologous_recombination import HomologousRecombination


class ParallelEvolutionEngine:
    """并行化进化引擎 - 利用多CPU核心加速基因组进化模拟"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-9,
                 hgt_rate: float = 0.001,
                 recombination_rate: float = 1e-6,
                 min_similarity_for_recombination: float = 0.7,
                 num_processes: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 enable_progress_sharing: bool = True):
        """
        初始化并行化进化引擎
        
        Args:
            num_processes: 并行进程数，None表示使用CPU核心数
            chunk_size: 基因分块大小，None表示自动计算
            enable_progress_sharing: 是否启用进程间进度共享
        """
        
        # 进化参数
        self.mutation_rate = mutation_rate
        self.hgt_rate = hgt_rate
        self.recombination_rate = recombination_rate
        self.min_similarity_for_recombination = min_similarity_for_recombination
        
        # 并行化配置
        self.num_processes = num_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        self.enable_progress_sharing = enable_progress_sharing
        
        # 确保不超过系统CPU核心数
        max_cores = mp.cpu_count()
        if self.num_processes > max_cores:
            print(f"⚠️  Warning: Requested {self.num_processes} processes, but only {max_cores} CPU cores available")
            self.num_processes = max_cores
        
        # 进化历史记录
        self.evolution_history = []
        
        print(f"🚀 Parallel Evolution Engine initialized:")
        print(f"   CPU cores available: {max_cores}")
        print(f"   Processes to use: {self.num_processes}")
        print(f"   Progress sharing: {'Enabled' if enable_progress_sharing else 'Disabled'}")
    
    def _calculate_optimal_chunk_size(self, total_genes: int) -> int:
        """计算最优的基因分块大小"""
        if self.chunk_size:
            return self.chunk_size
        
        # 基于基因数量和进程数计算最优分块大小
        # 目标：每个进程处理的基因数量相对均衡，避免负载不均
        base_chunk_size = max(1, total_genes // (self.num_processes * 4))  # 每个进程4个块
        
        # 根据基因数量调整
        if total_genes < 100:
            return max(1, total_genes // self.num_processes)
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
    
    def evolve_genes_chunk_parallel(self, genes_chunk: List[Gene], 
                                  evolution_params: Dict,
                                  shared_progress: Optional[Any] = None,
                                  process_id: int = 0) -> Tuple[List[Gene], Dict]:
        """
        并行处理基因块的进化
        
        Args:
            genes_chunk: 要处理的基因块
            evolution_params: 进化参数
            shared_progress: 共享的进度计数器
            process_id: 进程ID
        
        Returns:
            (evolved_genes, stats): 进化后的基因和统计信息
        """
        
        # 创建本地进化机制实例（避免进程间共享问题）
        point_mutation = OptimizedPointMutationEngine(
            mutation_rate=evolution_params['mutation_rate'],
            enable_transition_bias=evolution_params.get('enable_transition_bias', True),
            transition_transversion_ratio=evolution_params.get('transition_transversion_ratio', 2.0),
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
        
        start_time = time.time()
        
        # 创建临时基因组用于处理这个块
        temp_genome = Genome(genes_chunk)
        
        # 应用进化机制
        try:
            # 点突变
            mutations = point_mutation.apply_mutations(temp_genome, generations=1)
            chunk_stats['mutations'] = mutations
            
            # HGT（需要小心处理，避免基因数量变化导致的并行问题）
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
            with shared_progress.get_lock():
                shared_progress.value += len(genes_chunk)
        
        return temp_genome.genes, chunk_stats
    
    def evolve_one_generation_parallel(self, genome: Genome) -> Dict:
        """使用并行处理进化一代"""
        
        generation_start_time = time.time()
        
        # 准备进化参数
        evolution_params = {
            'mutation_rate': self.mutation_rate,
            'hgt_rate': self.hgt_rate,
            'recombination_rate': self.recombination_rate,
            'min_similarity_for_recombination': self.min_similarity_for_recombination,
            'enable_transition_bias': True,
            'transition_transversion_ratio': 2.0,
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
            'chunks_processed': len(gene_chunks),
            'parallel_processing_time': 0,
            'chunk_stats': []
        }
        
        # 设置共享进度计数器
        if self.enable_progress_sharing:
            manager = Manager()
            shared_progress = manager.Value('i', 0)
        else:
            shared_progress = None
        
        parallel_start_time = time.time()
        
        # 并行处理所有基因块
        with Pool(processes=self.num_processes) as pool:
            # 创建部分函数，固定evolution_params和shared_progress
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
            generation_stats['chunk_stats'].append(chunk_stats)
        
        # 更新基因组
        genome.genes = evolved_genes
        genome.generation += 1
        genome.total_mutations += generation_stats['total_mutations']
        genome.total_hgt_events += generation_stats['total_hgt_events']
        genome.total_recombination_events += generation_stats['total_recombination_events']
        
        # 记录最终统计
        generation_stats['final_stats'] = genome.get_statistics()
        generation_stats['total_processing_time'] = time.time() - generation_start_time
        
        return generation_stats
    
    def evolve_multiple_generations_parallel(self, genome: Genome, generations: int, 
                                           show_progress: bool = True) -> List[Dict]:
        """并行进化多代"""
        
        history = []
        start_time = time.time()
        
        if show_progress:
            # 根据代数确定显示频率
            if generations <= 50:
                display_freq = 5
            elif generations <= 100:
                display_freq = 10
            elif generations <= 1000:
                display_freq = 50
            else:
                display_freq = 100
            
            print(f"🚀 Starting PARALLEL evolution simulation: {generations:,} generations")
            print(f"   Parallel processes: {self.num_processes}")
            print(f"   Progress updates every {display_freq} generation(s)")
            print("=" * 80)
        else:
            display_freq = max(1, generations // 10)
        
        for gen in range(generations):
            gen_start_time = time.time()
            
            # 并行进化一代
            gen_stats = self.evolve_one_generation_parallel(genome)
            
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
                
                # 计算并行效率
                parallel_time = gen_stats['parallel_processing_time']
                total_time = gen_stats['total_processing_time']
                parallel_efficiency = (parallel_time / total_time) * 100 if total_time > 0 else 0
                
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
                parallel_info = (f"Parallel: {parallel_efficiency:.1f}% | "
                               f"Chunks: {gen_stats['chunks_processed']} | "
                               f"Processes: {self.num_processes}")
                
                print(f"\r[{bar}] {progress*100:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"{1/avg_time_per_gen:.1f} gen/s | ETA: {estimated_remaining/60:.1f}min | "
                      f"{genome_info} | {parallel_info}", end="", flush=True)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n\n🎉 PARALLEL evolution completed!")
            print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
            print(f"Average speed: {generations/total_time:.2f} generations/second")
            
            # 并行性能分析
            avg_parallel_efficiency = np.mean([
                (h['parallel_processing_time'] / h['total_processing_time']) * 100 
                for h in history if h['total_processing_time'] > 0
            ])
            
            print(f"Parallel efficiency: {avg_parallel_efficiency:.1f}%")
            print(f"Processes used: {self.num_processes}/{mp.cpu_count()}")
            print("=" * 80)
        
        self.evolution_history.extend(history)
        return history
    
    def simulate_evolution_parallel(self, 
                                  initial_genome: Genome, 
                                  generations: int,
                                  save_snapshots: bool = True,
                                  snapshot_interval: int = 100) -> Tuple[Genome, List[Dict]]:
        """完整的并行进化模拟"""
        
        print("🚀 PARALLEL PROKARYOTIC GENOME EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"📸 Snapshots: {'Enabled' if save_snapshots else 'Disabled'} (interval: {snapshot_interval})")
        print(f"⚡ Parallel processing: {self.num_processes} processes")
        print(f"🧬 Evolution mechanisms: Parallel point mutations, HGT, Homologous recombination")
        print("=" * 80)
        
        # 创建基因组副本
        evolving_genome = initial_genome.copy()
        simulation_start_time = time.time()
        
        # 记录初始状态
        snapshots = []
        if save_snapshots:
            initial_summary = self._get_evolution_summary(evolving_genome)
            initial_summary['snapshot_generation'] = 0
            snapshots.append(initial_summary)
        
        # 并行进化过程
        evolution_history = self.evolve_multiple_generations_parallel(
            evolving_genome, generations, show_progress=True
        )
        
        # 保存快照
        if save_snapshots:
            print(f"📸 Saving snapshots every {snapshot_interval} generations...")
            for i in range(0, len(evolution_history), snapshot_interval):
                if i < len(evolution_history):
                    snapshot = self._get_evolution_summary(evolving_genome)
                    snapshot['snapshot_generation'] = evolution_history[i]['generation']
                    snapshots.append(snapshot)
        
        # 最终总结
        total_simulation_time = time.time() - simulation_start_time
        
        print(f"\n🎉 PARALLEL SIMULATION COMPLETED!")
        print(f"🧬 Final genome: {evolving_genome.gene_count:,} genes, {evolving_genome.size:,} bp")
        print(f"📈 Changes: {evolving_genome.size - initial_genome.size:+,} bp, "
              f"{evolving_genome.gene_count - initial_genome.gene_count:+,} genes")
        print(f"⚡ Parallel performance: {self.num_processes} processes, "
              f"{generations/total_simulation_time:.2f} gen/s")
        
        return evolving_genome, snapshots
    
    def _get_evolution_summary(self, genome: Genome) -> Dict:
        """获取进化总结"""
        return {
            'genome_stats': genome.get_statistics(),
            'parallel_info': {
                'processes_used': self.num_processes,
                'parallel_processing': True,
                'cpu_cores_available': mp.cpu_count()
            }
        }
    
    def get_parallel_performance_analysis(self) -> Dict:
        """分析并行性能"""
        if not self.evolution_history:
            return {'error': 'No evolution history available'}
        
        parallel_times = [h['parallel_processing_time'] for h in self.evolution_history]
        total_times = [h['total_processing_time'] for h in self.evolution_history]
        
        parallel_efficiencies = [
            (p_time / t_time) * 100 if t_time > 0 else 0
            for p_time, t_time in zip(parallel_times, total_times)
        ]
        
        chunks_per_generation = [h['chunks_processed'] for h in self.evolution_history]
        
        return {
            'avg_parallel_efficiency': np.mean(parallel_efficiencies),
            'min_parallel_efficiency': np.min(parallel_efficiencies),
            'max_parallel_efficiency': np.max(parallel_efficiencies),
            'avg_parallel_time': np.mean(parallel_times),
            'avg_total_time': np.mean(total_times),
            'avg_chunks_per_generation': np.mean(chunks_per_generation),
            'processes_used': self.num_processes,
            'cpu_cores_available': mp.cpu_count(),
            'theoretical_speedup': self.num_processes,
            'actual_speedup': np.mean(parallel_efficiencies) / 100 * self.num_processes
        }


def evolve_genes_chunk_worker(chunk_args: Tuple[List[Gene], int], 
                            evolution_params: Dict,
                            shared_progress: Optional[Any] = None) -> Tuple[List[Gene], Dict]:
    """
    工作进程函数：处理基因块的进化
    这个函数需要在模块级别定义以支持multiprocessing
    """
    genes_chunk, process_id = chunk_args
    
    # 直接在工作进程中处理，避免循环导入
    from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
    from mechanisms.horizontal_transfer import HorizontalGeneTransfer
    from mechanisms.homologous_recombination import HomologousRecombination
    from core.genome import Genome
    
    # 创建本地进化机制实例
    point_mutation = OptimizedPointMutationEngine(
        mutation_rate=evolution_params['mutation_rate'],
        enable_transition_bias=evolution_params.get('enable_transition_bias', True),
        transition_transversion_ratio=evolution_params.get('transition_transversion_ratio', 2.0),
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
        
        # HGT（需要小心处理，避免基因数量变化导致的并行问题）
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