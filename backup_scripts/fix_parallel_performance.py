#!/usr/bin/env python3
"""
修复并行性能问题的补丁
直接优化 UnifiedEvolutionEngine 的并行处理

Version: 1.0.0
Author: ProGenomeEvoSimulator Team  
Date: 2025-09-27
"""

import time
import copy
import multiprocessing as mp
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from functools import partial

from core.genome import Genome, Gene


# 全局工作进程状态
_worker_initialized = False
_worker_engines = None


def init_parallel_worker(evolution_params: Dict):
    """
    工作进程初始化 - 只在进程启动时调用一次
    避免每次任务都重新创建对象
    """
    global _worker_initialized, _worker_engines
    
    if _worker_initialized:
        return
    
    try:
        # 导入必要的模块
        from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine
        from mechanisms.horizontal_transfer import HorizontalGeneTransfer
        from mechanisms.homologous_recombination import HomologousRecombination
        
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
        
        _worker_initialized = True
        
    except Exception as e:
        print(f"❌ Worker initialization failed: {e}")
        _worker_initialized = False
        _worker_engines = None


def optimized_evolve_chunk(chunk_data: Tuple[List[Gene], int]) -> Tuple[List[Gene], Dict]:
    """
    优化的基因块进化函数
    使用预初始化的对象，减少创建开销
    """
    global _worker_engines
    
    genes_chunk, chunk_id = chunk_data
    
    if not _worker_engines:
        # 如果工作进程未正确初始化，返回原始基因
        return genes_chunk, {
            'chunk_id': chunk_id,
            'genes_processed': len(genes_chunk),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0,
            'processing_time': 0,
            'error': 'Worker not initialized'
        }
    
    start_time = time.time()
    
    # 统计信息
    chunk_stats = {
        'chunk_id': chunk_id,
        'genes_processed': len(genes_chunk),
        'mutations': 0,
        'hgt_events': 0,
        'recombination_events': 0,
        'processing_time': 0
    }
    
    try:
        # 创建临时基因组
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
        
        evolved_genes = temp_genome.genes
        
    except Exception as e:
        print(f"❌ Chunk processing error: {e}")
        chunk_stats['error'] = str(e)
        evolved_genes = genes_chunk
    
    chunk_stats['processing_time'] = time.time() - start_time
    
    return evolved_genes, chunk_stats


def patch_unified_engine():
    """
    为 UnifiedEvolutionEngine 打补丁，优化并行性能
    """
    from core.unified_evolution_engine import UnifiedEvolutionEngine
    
    def optimized_calculate_chunk_size(self, total_genes: int) -> int:
        """
        优化的分块大小计算
        减少分块数量，增加每块大小以减少通信开销
        """
        num_processes = self.config['num_processes']
        
        # 目标：每个进程处理1-2个大块，而不是很多小块
        min_chunk_size = max(200, total_genes // (num_processes * 2))
        max_chunk_size = max(500, total_genes // num_processes)
        
        # 确保分块大小合理
        if total_genes < 1000:
            return max(100, total_genes // num_processes)
        else:
            return min(max_chunk_size, max(min_chunk_size, 300))
    
    def optimized_evolve_one_generation_parallel(self, genome: Genome) -> Dict:
        """
        优化的并行进化一代
        """
        generation_start_time = time.time()
        
        # 使用优化的分块策略
        chunk_size = self.optimized_calculate_chunk_size(genome.gene_count)
        gene_chunks = []
        
        for i in range(0, len(genome.genes), chunk_size):
            chunk = genome.genes[i:i + chunk_size]
            gene_chunks.append(chunk)
        
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'total_mutations': 0,
            'total_hgt_events': 0,
            'total_recombination_events': 0,
            'genes_lost': 0,
            'chunks_processed': len(gene_chunks),
            'chunk_size': chunk_size,
            'parallel_processing_time': 0,
            'processing_mode': 'optimized_parallel'
        }
        
        # 准备进化参数
        evolution_params = {
            'mutation_rate': self.config['mutation_rate'],
            'hgt_rate': self.config['hgt_rate'],
            'recombination_rate': self.config['recombination_rate'],
            'min_similarity_for_recombination': self.config['min_similarity_for_recombination'],
            'enable_transition_bias': True,
            'enable_hotspots': True
        }
        
        parallel_start_time = time.time()
        
        # 创建或重用进程池
        if not hasattr(self, '_process_pool') or self._process_pool is None:
            self._process_pool = Pool(
                processes=self.config['num_processes'],
                initializer=init_parallel_worker,
                initargs=(evolution_params,)
            )
        
        # 准备任务数据
        chunk_args = [(chunk, i) for i, chunk in enumerate(gene_chunks)]
        
        try:
            # 并行执行
            results = self._process_pool.map(optimized_evolve_chunk, chunk_args)
        except Exception as e:
            print(f"❌ Parallel execution failed: {e}")
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
    
    def cleanup_parallel_resources(self):
        """清理并行资源"""
        if hasattr(self, '_process_pool') and self._process_pool is not None:
            self._process_pool.close()
            self._process_pool.join()
            self._process_pool = None
    
    # 应用补丁
    UnifiedEvolutionEngine.optimized_calculate_chunk_size = optimized_calculate_chunk_size
    UnifiedEvolutionEngine.evolve_one_generation_parallel = optimized_evolve_one_generation_parallel
    UnifiedEvolutionEngine.cleanup_parallel_resources = cleanup_parallel_resources
    
    # 添加析构函数
    original_del = getattr(UnifiedEvolutionEngine, '__del__', None)
    
    def patched_del(self):
        self.cleanup_parallel_resources()
        if original_del:
            original_del(self)
    
    UnifiedEvolutionEngine.__del__ = patched_del
    
    print("✅ UnifiedEvolutionEngine parallel performance patch applied!")


def test_patched_performance():
    """测试补丁后的性能"""
    
    # 应用补丁
    patch_unified_engine()
    
    from core.unified_evolution_engine import UnifiedEvolutionEngine
    from core.genome import Genome, Gene
    
    # 创建测试基因组
    print("🧪 Testing patched parallel performance...")
    
    genes = []
    for i in range(1000):
        sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=500))
        gene = Gene(
            id=f"gene_{i:04d}",
            sequence=sequence,
            start_pos=i * 500,
            length=500,
            is_core=True,
            origin='ancestral'
        )
        genes.append(gene)
    
    test_genome = Genome(genes)
    generations = 10
    
    print(f"Test: {test_genome.gene_count:,} genes, {generations} generations")
    
    # 串行测试
    print("\n🔄 Serial (baseline)...")
    serial_genome = test_genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.005,
        enable_parallel=False,
        enable_gene_loss=False
    )
    
    serial_start = time.time()
    serial_engine.evolve_multiple_generations(serial_genome, generations, show_progress=False)
    serial_time = time.time() - serial_start
    
    print(f"   Time: {serial_time:.3f}s")
    
    # 并行测试（补丁后）
    print("\n⚡ Parallel (patched)...")
    parallel_genome = test_genome.copy()
    parallel_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-4,
        hgt_rate=0.005,
        enable_parallel=True,
        num_processes=4,
        enable_gene_loss=False
    )
    
    parallel_start = time.time()
    parallel_engine.evolve_multiple_generations(parallel_genome, generations, show_progress=False)
    parallel_time = time.time() - parallel_start
    
    # 清理资源
    parallel_engine.cleanup_parallel_resources()
    
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    efficiency = (speedup / 4) * 100
    
    print(f"   Time: {parallel_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Efficiency: {efficiency:.1f}%")
    
    # 结果分析
    print(f"\n📊 PATCH RESULTS:")
    if speedup > 2.0:
        print("   ✅ 优秀的并行性能！")
    elif speedup > 1.5:
        print("   ✅ 良好的并行性能")
    elif speedup > 1.2:
        print("   ⚠️  中等的并行性能")
    else:
        print("   ❌ 并行性能仍需改进")
    
    return {
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'efficiency': efficiency
    }


if __name__ == "__main__":
    print("🔧 PARALLEL PERFORMANCE FIX")
    print("=" * 60)
    
    results = test_patched_performance()
    
    print(f"\n💡 RECOMMENDATIONS:")
    if results['efficiency'] < 50:
        print("   1. 考虑使用更大的基因组进行测试")
        print("   2. 调整分块大小参数")
        print("   3. 检查是否有其他系统瓶颈")
    else:
        print("   ✅ 并行优化成功！可以应用到主代码中")