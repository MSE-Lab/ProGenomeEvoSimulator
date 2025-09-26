#!/usr/bin/env python3
"""
Critical Fixes for Genome Evolution Simulator
关键问题修复脚本
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from core.genome import Genome, Gene
import hashlib
import gc

class MemoryEfficientGene:
    """内存高效的基因类"""
    
    def __init__(self, id: str, sequence: str, start_pos: int = 0, 
                 is_core: bool = True, origin: str = "ancestral"):
        self.id = id
        self._sequence = bytearray(sequence.encode('ascii'))  # 使用bytearray节省内存
        self.start_pos = start_pos
        self.length = len(sequence)
        self.is_core = is_core
        self.origin = origin
        self.mutation_count = 0
        self.recombination_count = 0
        self._sequence_hash = None  # 缓存序列哈希
    
    @property
    def sequence(self) -> str:
        """获取序列字符串"""
        return self._sequence.decode('ascii')
    
    @sequence.setter
    def sequence(self, value: str):
        """设置序列并更新相关属性"""
        self._sequence = bytearray(value.encode('ascii'))
        self.length = len(value)
        self._sequence_hash = None  # 清除缓存的哈希
    
    def get_sequence_hash(self) -> str:
        """获取序列哈希用于快速比较"""
        if self._sequence_hash is None:
            self._sequence_hash = hashlib.md5(self._sequence).hexdigest()
        return self._sequence_hash
    
    def mutate_position(self, position: int, new_base: str):
        """在指定位置进行点突变"""
        if 0 <= position < len(self._sequence):
            self._sequence[position] = ord(new_base)
            self.mutation_count += 1
            self._sequence_hash = None  # 清除缓存的哈希

class SafeEvolutionEngine:
    """带有错误处理和恢复机制的进化引擎"""
    
    def __init__(self, checkpoint_interval: int = 100):
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []
        self.error_log = []
    
    def create_checkpoint(self, genome: Genome, generation: int):
        """创建检查点"""
        try:
            checkpoint = {
                'generation': generation,
                'genome_stats': genome.get_statistics(),
                'timestamp': time.time()
            }
            self.checkpoints.append(checkpoint)
            
            # 只保留最近的10个检查点
            if len(self.checkpoints) > 10:
                self.checkpoints.pop(0)
                
        except Exception as e:
            self.log_error(f"Checkpoint creation failed: {e}")
    
    def log_error(self, error_msg: str):
        """记录错误"""
        import time
        self.error_log.append({
            'timestamp': time.time(),
            'error': error_msg
        })
    
    def safe_evolve_generation(self, genome: Genome, mechanisms: List):
        """安全的单代进化，带错误处理"""
        try:
            generation_stats = {
                'generation': genome.generation + 1,
                'mutations': 0,
                'hgt_events': 0,
                'recombination_events': 0,
                'errors': []
            }
            
            # 应用各种进化机制
            for mechanism in mechanisms:
                try:
                    if hasattr(mechanism, 'apply_mutations'):
                        events = mechanism.apply_mutations(genome, 1)
                        generation_stats['mutations'] += events
                    elif hasattr(mechanism, 'apply_hgt'):
                        events = mechanism.apply_hgt(genome, 1)
                        generation_stats['hgt_events'] += events
                    elif hasattr(mechanism, 'apply_recombination'):
                        events = mechanism.apply_recombination(genome, 1)
                        generation_stats['recombination_events'] += events
                        
                except Exception as e:
                    error_msg = f"Mechanism {type(mechanism).__name__} failed: {e}"
                    self.log_error(error_msg)
                    generation_stats['errors'].append(error_msg)
            
            genome.generation += 1
            
            # 定期创建检查点
            if genome.generation % self.checkpoint_interval == 0:
                self.create_checkpoint(genome, genome.generation)
                gc.collect()  # 强制垃圾回收
            
            return generation_stats
            
        except Exception as e:
            self.log_error(f"Generation {genome.generation + 1} failed: {e}")
            raise

class FastHomologyFinder:
    """快速同源基因查找器"""
    
    def __init__(self, kmer_size: int = 15):
        self.kmer_size = kmer_size
        self.kmer_index = {}
        self._similarity_cache = {}
    
    def extract_kmers(self, sequence: str) -> Set[str]:
        """提取k-mer"""
        kmers = set()
        for i in range(len(sequence) - self.kmer_size + 1):
            kmers.add(sequence[i:i + self.kmer_size])
        return kmers
    
    def build_kmer_index(self, genome: Genome):
        """构建k-mer索引"""
        self.kmer_index.clear()
        
        for gene in genome.genes:
            kmers = self.extract_kmers(gene.sequence)
            for kmer in kmers:
                if kmer not in self.kmer_index:
                    self.kmer_index[kmer] = []
                self.kmer_index[kmer].append(gene)
    
    def find_candidate_homologs(self, query_gene: Gene) -> List[Gene]:
        """快速查找候选同源基因"""
        query_kmers = self.extract_kmers(query_gene.sequence)
        candidate_genes = []
        candidate_ids = set()
        
        for kmer in query_kmers:
            if kmer in self.kmer_index:
                for gene in self.kmer_index[kmer]:
                    if gene.id != query_gene.id and gene.id not in candidate_ids:
                        candidate_genes.append(gene)
                        candidate_ids.add(gene.id)
        
        return candidate_genes
    
    def calculate_kmer_similarity(self, gene1: Gene, gene2: Gene) -> float:
        """基于k-mer计算相似性"""
        cache_key = (gene1.id, gene2.id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        kmers1 = self.extract_kmers(gene1.sequence)
        kmers2 = self.extract_kmers(gene2.sequence)
        
        if not kmers1 or not kmers2:
            similarity = 0.0
        else:
            intersection = len(kmers1 & kmers2)
            union = len(kmers1 | kmers2)
            similarity = intersection / union if union > 0 else 0.0
        
        # 缓存结果
        self._similarity_cache[cache_key] = similarity
        self._similarity_cache[(gene2.id, gene1.id)] = similarity  # 对称性
        
        return similarity

class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.memory_warnings = []
    
    def check_memory_usage(self) -> Dict:
        """检查当前内存使用"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            status = {
                'memory_mb': memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'usage_ratio': memory_mb / self.max_memory_mb,
                'warning': memory_mb > self.max_memory_mb * 0.8
            }
            
            if status['warning']:
                warning_msg = f"High memory usage: {memory_mb:.1f}MB / {self.max_memory_mb}MB"
                self.memory_warnings.append(warning_msg)
                print(f"⚠️  {warning_msg}")
            
            return status
            
        except ImportError:
            return {'error': 'psutil not available for memory monitoring'}
    
    def suggest_memory_optimization(self, genome: Genome) -> List[str]:
        """建议内存优化措施"""
        suggestions = []
        
        # 检查基因组大小
        if genome.size > 10_000_000:  # 10MB
            suggestions.append("Consider using memory-mapped files for large genomes")
        
        # 检查基因数量
        if len(genome.genes) > 10000:
            suggestions.append("Consider processing genes in batches")
        
        # 检查序列长度分布
        gene_lengths = [gene.length for gene in genome.genes]
        if max(gene_lengths) > 50000:
            suggestions.append("Very long genes detected, consider sequence compression")
        
        return suggestions

def run_critical_fixes_demo():
    """演示关键修复功能"""
    print("🔧 CRITICAL FIXES DEMONSTRATION")
    print("=" * 50)
    
    # 1. 内存高效基因测试
    print("1. Testing memory-efficient gene storage...")
    original_gene = Gene(
        id="test_gene",
        sequence="ATCGATCGATCG" * 1000,  # 12KB sequence
        start_pos=0,
        length=12000
    )
    
    efficient_gene = MemoryEfficientGene(
        id="test_gene_efficient",
        sequence="ATCGATCGATCG" * 1000
    )
    
    print(f"   Original gene memory usage: ~{len(original_gene.sequence) * 4} bytes (string)")
    print(f"   Efficient gene memory usage: ~{len(efficient_gene._sequence)} bytes (bytearray)")
    print(f"   Memory savings: ~{((len(original_gene.sequence) * 4) - len(efficient_gene._sequence)) / (len(original_gene.sequence) * 4) * 100:.1f}%")
    
    # 2. 快速同源基因查找测试
    print("\n2. Testing fast homology finder...")
    from core.genome import create_initial_genome
    
    test_genome = create_initial_genome(gene_count=100, avg_gene_length=1000)
    homology_finder = FastHomologyFinder(kmer_size=10)
    
    print("   Building k-mer index...")
    homology_finder.build_kmer_index(test_genome)
    print(f"   Index built with {len(homology_finder.kmer_index)} unique k-mers")
    
    # 测试查找
    query_gene = test_genome.genes[0]
    candidates = homology_finder.find_candidate_homologs(query_gene)
    print(f"   Found {len(candidates)} candidate homologous genes")
    
    # 3. 内存监控测试
    print("\n3. Testing memory monitoring...")
    monitor = MemoryMonitor(max_memory_mb=512)
    memory_status = monitor.check_memory_usage()
    
    if 'memory_mb' in memory_status:
        print(f"   Current memory usage: {memory_status['memory_mb']:.1f}MB")
        print(f"   Usage ratio: {memory_status['usage_ratio']:.2f}")
    
    suggestions = monitor.suggest_memory_optimization(test_genome)
    if suggestions:
        print("   Memory optimization suggestions:")
        for suggestion in suggestions:
            print(f"     - {suggestion}")
    
    print("\n✅ Critical fixes demonstration completed!")

if __name__ == "__main__":
    run_critical_fixes_demo()