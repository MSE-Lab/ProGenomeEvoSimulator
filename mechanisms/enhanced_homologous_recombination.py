#!/usr/bin/env python3
"""
Enhanced Homologous Recombination - 基于分子进化理论的改进同源重组机制

改进特性：
1. 序列相似性依赖的重组频率
2. 重组热点识别 (Chi sites等)
3. 基因转换 vs 交叉重组
4. 重组长度的生物学约束
5. 功能域保护机制
6. 重组后的适应性评估
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from core.genome import Genome, Gene

@dataclass
class RecombinationEvent:
    """重组事件记录"""
    gene1_id: str
    gene2_id: str
    recombination_type: str  # 'crossover', 'gene_conversion', 'unequal_crossover'
    start_position: int
    end_position: int
    length: int
    sequence_similarity: float
    success: bool

class EnhancedHomologousRecombination:
    """增强的同源重组引擎"""
    
    def __init__(self, 
                 recombination_rate: float = 1e-6,
                 min_similarity: float = 0.85,  # 修正：更严格的相似度要求
                 enable_recombination_hotspots: bool = True,
                 enable_gene_conversion: bool = True,
                 enable_functional_protection: bool = True,
                 chi_site_enhancement: float = 5.0):  # 修正：更合理的Chi位点增强
        
        self.recombination_rate = recombination_rate
        self.min_similarity = min_similarity
        self.enable_recombination_hotspots = enable_recombination_hotspots
        self.enable_gene_conversion = enable_gene_conversion
        self.enable_functional_protection = enable_functional_protection
        self.chi_site_enhancement = chi_site_enhancement
        
        # 重组热点序列 (Chi sites 和其他已知热点)
        self.hotspot_motifs = {
            'chi_site': 'GCTGGTGG',           # E.coli Chi site
            'chi_like_1': 'GCTGGTGA',        # Chi-like sequences
            'chi_like_2': 'GCTGGTAG',
            'recombination_signal': 'CACGTG',  # 其他重组信号
            'palindrome': 'GAATTC',           # 回文序列
            'at_rich': 'AAATTT'               # AT富集区域
        }
        
        # 重组类型权重
        self.recombination_types = {
            'crossover': 0.6,           # 交叉重组 - 最常见
            'gene_conversion': 0.3,     # 基因转换 - 较常见
            'unequal_crossover': 0.1    # 不等交叉 - 较少见
        }
        
        # 重组长度参数 (基于实验数据)
        self.recombination_length_params = {
            'crossover': {'mean': 500, 'std': 200, 'min': 50, 'max': 2000},
            'gene_conversion': {'mean': 200, 'std': 100, 'min': 20, 'max': 800},
            'unequal_crossover': {'mean': 1000, 'std': 500, 'min': 100, 'max': 5000}
        }
        
        # 统计信息
        self.recombination_stats = {
            'total_attempts': 0,
            'successful_recombinations': 0,
            'blocked_by_protection': 0,
            'failed_similarity_check': 0,
            'hotspot_enhanced': 0,
            'recombination_by_type': {rtype: 0 for rtype in self.recombination_types},
            'events_history': []
        }
    
    def calculate_sequence_similarity_detailed(self, seq1: str, seq2: str) -> Tuple[float, Dict]:
        """详细的序列相似性计算"""
        if len(seq1) != len(seq2):
            # 对于长度不同的序列，使用局部比对
            min_len = min(len(seq1), len(seq2))
            max_len = max(len(seq1), len(seq2))
            
            # 滑动窗口寻找最佳比对
            best_similarity = 0.0
            best_offset = 0
            
            for offset in range(max_len - min_len + 1):
                if len(seq1) > len(seq2):
                    subseq1 = seq1[offset:offset + min_len]
                    subseq2 = seq2
                else:
                    subseq1 = seq1
                    subseq2 = seq2[offset:offset + min_len]
                
                matches = sum(1 for a, b in zip(subseq1, subseq2) if a == b)
                similarity = matches / min_len
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_offset = offset
            
            alignment_info = {
                'alignment_length': min_len,
                'best_offset': best_offset,
                'length_difference': abs(len(seq1) - len(seq2))
            }
            
            return best_similarity, alignment_info
        
        # 相同长度序列的直接比较
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        similarity = matches / len(seq1) if len(seq1) > 0 else 0.0
        
        # 计算详细的相似性信息
        transitions = 0
        transversions = 0
        gaps = 0
        
        transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        
        for a, b in zip(seq1, seq2):
            if a != b:
                if (a, b) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
        
        alignment_info = {
            'alignment_length': len(seq1),
            'matches': matches,
            'transitions': transitions,
            'transversions': transversions,
            'ti_tv_ratio': transitions / transversions if transversions > 0 else float('inf'),
            'length_difference': 0
        }
        
        return similarity, alignment_info
    
    def find_recombination_hotspots(self, sequence: str) -> List[Tuple[int, str, float]]:
        """寻找重组热点"""
        if not self.enable_recombination_hotspots:
            return []
        
        hotspots = []
        
        for motif_name, motif_seq in self.hotspot_motifs.items():
            motif_len = len(motif_seq)
            
            for i in range(len(sequence) - motif_len + 1):
                # 精确匹配
                if sequence[i:i + motif_len] == motif_seq:
                    enhancement = self.chi_site_enhancement if 'chi' in motif_name else 3.0
                    hotspots.append((i, motif_name, enhancement))
                
                # 允许1个错配的近似匹配
                elif motif_len >= 6:
                    mismatches = sum(1 for j in range(motif_len) 
                                   if sequence[i + j] != motif_seq[j])
                    if mismatches == 1:
                        enhancement = (self.chi_site_enhancement * 0.5 if 'chi' in motif_name else 1.5)
                        hotspots.append((i, f"{motif_name}_1mm", enhancement))
        
        return hotspots
    
    def find_homologous_gene_pairs_enhanced(self, genome: Genome) -> List[Tuple[Gene, Gene, float, Dict]]:
        """增强的同源基因对识别"""
        homologous_pairs = []
        genes = genome.genes
        
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene1, gene2 = genes[i], genes[j]
                
                # 计算详细的序列相似性
                similarity, alignment_info = self.calculate_sequence_similarity_detailed(
                    gene1.sequence, gene2.sequence
                )
                
                if similarity >= self.min_similarity:
                    homologous_pairs.append((gene1, gene2, similarity, alignment_info))
        
        return homologous_pairs
    
    def assess_functional_protection(self, gene1: Gene, gene2: Gene, 
                                   start_pos: int, end_pos: int) -> float:
        """评估功能域保护"""
        if not self.enable_functional_protection:
            return 1.0
        
        protection_score = 1.0
        
        # 检查是否影响重要功能域 (简化版本)
        # 在实际应用中，这里会使用蛋白质域数据库
        
        # 基于基因长度和位置推断功能重要性
        for gene in [gene1, gene2]:
            gene_length = len(gene.sequence)
            
            # 假设基因中部是重要的功能域
            functional_start = gene_length // 4
            functional_end = 3 * gene_length // 4
            
            # 检查重组区域是否与功能域重叠
            overlap_start = max(start_pos, functional_start)
            overlap_end = min(end_pos, functional_end)
            
            if overlap_end > overlap_start:
                overlap_length = overlap_end - overlap_start
                functional_length = functional_end - functional_start
                overlap_ratio = overlap_length / functional_length
                
                # 重叠比例越高，保护越强
                protection_penalty = overlap_ratio * 0.8
                protection_score *= (1.0 - protection_penalty)
        
        return max(0.1, protection_score)
    
    def select_recombination_type(self) -> str:
        """选择重组类型"""
        types = list(self.recombination_types.keys())
        weights = list(self.recombination_types.values())
        return np.random.choice(types, p=weights)
    
    def generate_recombination_length(self, recombination_type: str, 
                                    max_length: int) -> int:
        """生成重组长度"""
        params = self.recombination_length_params[recombination_type]
        
        # 使用截断正态分布
        length = int(np.random.normal(params['mean'], params['std']))
        length = max(params['min'], min(params['max'], length, max_length))
        
        return length
    
    def calculate_recombination_probability(self, gene1: Gene, gene2: Gene, 
                                         similarity: float) -> float:
        """计算重组概率"""
        base_prob = self.recombination_rate
        
        # 相似性依赖的概率调整
        similarity_factor = (similarity - self.min_similarity) / (1.0 - self.min_similarity)
        similarity_enhancement = 1.0 + similarity_factor * 5.0
        
        # 检查重组热点
        hotspot_enhancement = 1.0
        if self.enable_recombination_hotspots:
            hotspots1 = self.find_recombination_hotspots(gene1.sequence)
            hotspots2 = self.find_recombination_hotspots(gene2.sequence)
            
            if hotspots1 or hotspots2:
                max_enhancement = max([h[2] for h in hotspots1 + hotspots2] + [1.0])
                hotspot_enhancement = max_enhancement
                self.recombination_stats['hotspot_enhanced'] += 1
        
        # 基因长度因子 (较长的基因有更多重组机会)
        length_factor = min(len(gene1.sequence), len(gene2.sequence)) / 1000.0
        length_factor = min(2.0, max(0.5, length_factor))
        
        final_prob = base_prob * similarity_enhancement * hotspot_enhancement * length_factor
        return min(1.0, final_prob)
    
    def perform_enhanced_recombination(self, gene1: Gene, gene2: Gene, 
                                     recombination_type: str, similarity: float) -> bool:
        """执行增强的重组"""
        try:
            # 确定重组区域
            max_length = min(len(gene1.sequence), len(gene2.sequence))
            if max_length < 20:  # 太短无法重组
                return False
            
            recomb_length = self.generate_recombination_length(recombination_type, max_length - 10)
            
            # 选择重组起始位置
            max_start = max_length - recomb_length
            if max_start <= 0:
                return False
            
            # 偏好热点区域
            start_pos = random.randint(0, max_start)
            if self.enable_recombination_hotspots:
                hotspots1 = self.find_recombination_hotspots(gene1.sequence)
                hotspots2 = self.find_recombination_hotspots(gene2.sequence)
                
                all_hotspots = [(pos, enhancement) for pos, _, enhancement in hotspots1 + hotspots2]
                if all_hotspots:
                    # 30%概率选择热点附近
                    if random.random() < 0.3:
                        hotspot_pos, _ = random.choice(all_hotspots)
                        start_pos = max(0, min(max_start, hotspot_pos - recomb_length // 2))
            
            end_pos = start_pos + recomb_length
            
            # 功能保护检查
            protection_score = self.assess_functional_protection(gene1, gene2, start_pos, end_pos)
            if random.random() > protection_score:
                self.recombination_stats['blocked_by_protection'] += 1
                return False
            
            # 执行不同类型的重组
            success = False
            
            if recombination_type == 'crossover':
                success = self._perform_crossover(gene1, gene2, start_pos, end_pos)
            elif recombination_type == 'gene_conversion':
                success = self._perform_gene_conversion(gene1, gene2, start_pos, end_pos)
            elif recombination_type == 'unequal_crossover':
                success = self._perform_unequal_crossover(gene1, gene2, start_pos, end_pos)
            
            if success:
                # 记录重组事件
                event = RecombinationEvent(
                    gene1_id=gene1.id,
                    gene2_id=gene2.id,
                    recombination_type=recombination_type,
                    start_position=start_pos,
                    end_position=end_pos,
                    length=recomb_length,
                    sequence_similarity=similarity,
                    success=True
                )
                self.recombination_stats['events_history'].append(event)
                self.recombination_stats['recombination_by_type'][recombination_type] += 1
                
                # 更新基因重组计数
                gene1.recombination_count += 1
                gene2.recombination_count += 1
            
            return success
            
        except Exception as e:
            print(f"Enhanced recombination failed: {e}")
            return False
    
    def _perform_crossover(self, gene1: Gene, gene2: Gene, start: int, end: int) -> bool:
        """执行交叉重组"""
        seq1_list = list(gene1.sequence)
        seq2_list = list(gene2.sequence)
        
        # 交换片段
        temp_fragment = seq1_list[start:end]
        seq1_list[start:end] = seq2_list[start:end]
        seq2_list[start:end] = temp_fragment
        
        gene1.sequence = ''.join(seq1_list)
        gene2.sequence = ''.join(seq2_list)
        
        return True
    
    def _perform_gene_conversion(self, gene1: Gene, gene2: Gene, start: int, end: int) -> bool:
        """执行基因转换 (单向转移)"""
        seq1_list = list(gene1.sequence)
        seq2_list = list(gene2.sequence)
        
        # 随机选择转换方向
        if random.random() < 0.5:
            # gene2 -> gene1
            seq1_list[start:end] = seq2_list[start:end]
            gene1.sequence = ''.join(seq1_list)
        else:
            # gene1 -> gene2
            seq2_list[start:end] = seq1_list[start:end]
            gene2.sequence = ''.join(seq2_list)
        
        return True
    
    def _perform_unequal_crossover(self, gene1: Gene, gene2: Gene, start: int, end: int) -> bool:
        """执行不等交叉 (可能导致基因重复或缺失)"""
        # 简化版本 - 在实际应用中会更复杂
        seq1_list = list(gene1.sequence)
        seq2_list = list(gene2.sequence)
        
        # 创建不等长的交换
        fragment1 = seq1_list[start:end]
        fragment2 = seq2_list[start:end]
        
        # 随机调整片段长度
        if len(fragment1) > 10 and random.random() < 0.3:
            # 缩短片段1
            new_length = random.randint(len(fragment1) // 2, len(fragment1))
            fragment1 = fragment1[:new_length]
        
        if len(fragment2) > 10 and random.random() < 0.3:
            # 缩短片段2
            new_length = random.randint(len(fragment2) // 2, len(fragment2))
            fragment2 = fragment2[:new_length]
        
        # 执行交换
        seq1_list[start:end] = fragment2
        seq2_list[start:end] = fragment1
        
        gene1.sequence = ''.join(seq1_list)
        gene2.sequence = ''.join(seq2_list)
        
        return True
    
    def apply_enhanced_recombination(self, genome: Genome, generations: int = 1) -> int:
        """应用增强的同源重组"""
        # 找到同源基因对
        homologous_pairs = self.find_homologous_gene_pairs_enhanced(genome)
        
        if not homologous_pairs:
            return 0
        
        successful_recombinations = 0
        
        for gene1, gene2, similarity, alignment_info in homologous_pairs:
            self.recombination_stats['total_attempts'] += 1
            
            # 计算重组概率
            recomb_prob = self.calculate_recombination_probability(gene1, gene2, similarity)
            
            # 应用多代效应
            effective_prob = 1.0 - (1.0 - recomb_prob) ** generations
            
            if random.random() < effective_prob:
                # 选择重组类型
                recombination_type = self.select_recombination_type()
                
                # 执行重组
                if self.perform_enhanced_recombination(gene1, gene2, recombination_type, similarity):
                    successful_recombinations += 1
                    genome.total_recombination_events += 1
                    self.recombination_stats['successful_recombinations'] += 1
                else:
                    self.recombination_stats['failed_similarity_check'] += 1
        
        return successful_recombinations
    
    def get_enhanced_recombination_statistics(self, genome: Genome) -> Dict:
        """获取增强重组统计信息"""
        homologous_pairs = self.find_homologous_gene_pairs_enhanced(genome)
        recombination_counts = [gene.recombination_count for gene in genome.genes]
        
        base_stats = {
            'total_recombination_events': genome.total_recombination_events,
            'homologous_gene_pairs': len(homologous_pairs),
            'genes_with_recombination': sum(1 for count in recombination_counts if count > 0),
            'avg_recombination_per_gene': np.mean(recombination_counts) if recombination_counts else 0,
            'max_recombination_per_gene': max(recombination_counts) if recombination_counts else 0,
            'recombination_potential': len(homologous_pairs) / len(genome.genes) if genome.genes else 0
        }
        
        # 增强统计
        enhanced_stats = {
            'recombination_features': {
                'hotspots_enabled': self.enable_recombination_hotspots,
                'gene_conversion_enabled': self.enable_gene_conversion,
                'functional_protection_enabled': self.enable_functional_protection
            },
            'detailed_stats': self.recombination_stats.copy(),
            'success_rate': (self.recombination_stats['successful_recombinations'] / 
                           max(1, self.recombination_stats['total_attempts'])),
        }
        
        # 相似性分布分析
        if homologous_pairs:
            similarities = [similarity for _, _, similarity, _ in homologous_pairs]
            enhanced_stats['similarity_distribution'] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
        
        base_stats.update(enhanced_stats)
        return base_stats
    
    def print_recombination_analysis(self, genome: Genome):
        """打印重组分析结果"""
        stats = self.get_enhanced_recombination_statistics(genome)
        
        print("\n" + "=" * 60)
        print("🔄 ENHANCED RECOMBINATION ANALYSIS")
        print("=" * 60)
        
        print(f"📊 Recombination Statistics:")
        print(f"   Total recombination events: {stats['total_recombination_events']:,}")
        print(f"   Homologous gene pairs: {stats['homologous_gene_pairs']:,}")
        print(f"   Genes with recombination: {stats['genes_with_recombination']:,}")
        print(f"   Success rate: {stats['success_rate']:.3f}")
        
        detailed = stats['detailed_stats']
        print(f"\n🎯 Detailed Analysis:")
        print(f"   Total attempts: {detailed['total_attempts']:,}")
        print(f"   Successful recombinations: {detailed['successful_recombinations']:,}")
        print(f"   Blocked by protection: {detailed['blocked_by_protection']:,}")
        print(f"   Hotspot enhanced: {detailed['hotspot_enhanced']:,}")
        
        print(f"\n🔬 Recombination Types:")
        for rtype, count in detailed['recombination_by_type'].items():
            percentage = count / max(1, detailed['successful_recombinations']) * 100
            print(f"   {rtype}: {count} ({percentage:.1f}%)")
        
        if 'similarity_distribution' in stats:
            sim_dist = stats['similarity_distribution']
            print(f"\n📈 Similarity Distribution:")
            print(f"   Mean similarity: {sim_dist['mean']:.3f}")
            print(f"   Range: {sim_dist['min']:.3f} - {sim_dist['max']:.3f}")
        
        print("=" * 60)