import random
import numpy as np
from typing import List, Tuple, Optional
from core.genome import Genome, Gene

class HomologousRecombination:
    """同源重组引擎"""
    
    def __init__(self, 
                 recombination_rate: float = 1e-6,
                 min_similarity: float = 0.7,
                 min_recombination_length: int = 50,
                 max_recombination_length: int = 1000,
                 recombination_length_mean: int = 300,
                 recombination_length_std: int = 100):
        
        self.recombination_rate = recombination_rate  # 每bp每代的重组率
        self.min_similarity = min_similarity  # 最小相似性阈值
        self.min_recombination_length = min_recombination_length
        self.max_recombination_length = max_recombination_length
        self.recombination_length_mean = recombination_length_mean
        self.recombination_length_std = recombination_length_std
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """计算两个序列的相似性"""
        if len(seq1) != len(seq2):
            # 对于长度不同的序列，取较短的长度进行比较
            min_len = min(len(seq1), len(seq2))
            seq1 = seq1[:min_len]
            seq2 = seq2[:min_len]
        
        if len(seq1) == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def find_homologous_gene_pairs(self, genome: Genome) -> List[Tuple[Gene, Gene]]:
        """找到基因组中的同源基因对"""
        homologous_pairs = []
        genes = genome.genes
        
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene1, gene2 = genes[i], genes[j]
                
                # 计算序列相似性
                similarity = self.calculate_sequence_similarity(gene1.sequence, gene2.sequence)
                
                if similarity >= self.min_similarity:
                    homologous_pairs.append((gene1, gene2))
        
        return homologous_pairs
    
    def calculate_recombination_events(self, genome: Genome, generations: int = 1) -> int:
        """计算重组事件数量"""
        # 基于基因组大小和重组率计算期望事件数
        expected_events = genome.size * self.recombination_rate * generations
        return np.random.poisson(expected_events)
    
    def generate_recombination_length(self) -> int:
        """生成重组长度（正态分布）"""
        length = int(np.random.normal(self.recombination_length_mean, self.recombination_length_std))
        return max(self.min_recombination_length, 
                  min(self.max_recombination_length, length))
    
    def perform_recombination(self, gene1: Gene, gene2: Gene) -> bool:
        """在两个同源基因间执行重组"""
        try:
            # 确定重组区域
            recomb_length = self.generate_recombination_length()
            
            # 选择较短基因的长度作为最大重组长度
            max_length = min(len(gene1.sequence), len(gene2.sequence))
            recomb_length = min(recomb_length, max_length - 1)
            
            if recomb_length < self.min_recombination_length:
                return False
            
            # 随机选择重组起始位置
            max_start = max_length - recomb_length
            if max_start <= 0:
                return False
            
            start_pos = random.randint(0, max_start)
            end_pos = start_pos + recomb_length
            
            # 执行重组（交换序列片段）
            seq1_list = list(gene1.sequence)
            seq2_list = list(gene2.sequence)
            
            # 交换片段
            temp_fragment = seq1_list[start_pos:end_pos]
            seq1_list[start_pos:end_pos] = seq2_list[start_pos:end_pos]
            seq2_list[start_pos:end_pos] = temp_fragment
            
            # 更新基因序列
            gene1.sequence = ''.join(seq1_list)
            gene2.sequence = ''.join(seq2_list)
            
            # 更新重组计数
            gene1.recombination_count += 1
            gene2.recombination_count += 1
            
            return True
            
        except Exception as e:
            print(f"Recombination failed: {e}")
            return False
    
    def apply_recombination(self, genome: Genome, generations: int = 1) -> int:
        """对基因组应用同源重组"""
        # 找到同源基因对
        homologous_pairs = self.find_homologous_gene_pairs(genome)
        
        if not homologous_pairs:
            return 0
        
        # 计算重组事件数量
        recombination_events = self.calculate_recombination_events(genome, generations)
        successful_recombinations = 0
        
        for _ in range(recombination_events):
            if homologous_pairs:
                # 随机选择一对同源基因
                gene1, gene2 = random.choice(homologous_pairs)
                
                if self.perform_recombination(gene1, gene2):
                    successful_recombinations += 1
                    genome.total_recombination_events += 1
        
        return successful_recombinations
    
    def get_recombination_statistics(self, genome: Genome) -> dict:
        """获取重组统计信息"""
        homologous_pairs = self.find_homologous_gene_pairs(genome)
        recombination_counts = [gene.recombination_count for gene in genome.genes]
        
        return {
            'total_recombination_events': genome.total_recombination_events,
            'homologous_gene_pairs': len(homologous_pairs),
            'genes_with_recombination': sum(1 for count in recombination_counts if count > 0),
            'avg_recombination_per_gene': np.mean(recombination_counts) if recombination_counts else 0,
            'max_recombination_per_gene': max(recombination_counts) if recombination_counts else 0,
            'recombination_potential': len(homologous_pairs) / len(genome.genes) if genome.genes else 0
        }