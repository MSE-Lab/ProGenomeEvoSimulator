import numpy as np
from typing import List, Tuple, Dict, Optional
from core.genome import Genome, Gene
from dataclasses import dataclass

@dataclass
class OrthologPair:
    """同源基因对"""
    gene1: Gene
    gene2: Gene
    identity: float
    alignment_length: int
    gene1_coverage: float
    gene2_coverage: float

class ANICalculator:
    """平均核苷酸一致性计算器"""
    
    def __init__(self, 
                 ortholog_identity_threshold: float = 0.5,
                 min_alignment_length: int = 100,
                 fragment_length: int = 1000):
        
        self.ortholog_threshold = ortholog_identity_threshold
        self.min_alignment_length = min_alignment_length
        self.fragment_length = fragment_length
    
    def calculate_sequence_identity(self, seq1: str, seq2: str) -> Tuple[float, int]:
        """计算两个序列的一致性"""
        if not seq1 or not seq2:
            return 0.0, 0
        
        # 使用较短序列的长度进行比较
        min_length = min(len(seq1), len(seq2))
        if min_length < self.min_alignment_length:
            return 0.0, 0
        
        # 截取相同长度进行比较
        seq1_aligned = seq1[:min_length]
        seq2_aligned = seq2[:min_length]
        
        # 计算匹配数
        matches = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b)
        identity = matches / min_length
        
        return identity, min_length
    
    def find_orthologs(self, genome1: Genome, genome2: Genome) -> List[OrthologPair]:
        """找到两个基因组间的同源基因对"""
        ortholog_pairs = []
        
        # 简化的同源基因识别：基于基因ID和序列相似性
        for gene1 in genome1.genes:
            best_match = None
            best_identity = 0.0
            best_alignment_length = 0
            
            for gene2 in genome2.genes:
                # 如果基因ID相同（来自同一祖先），优先考虑
                if gene1.id == gene2.id or gene1.id.split('_')[0] == gene2.id.split('_')[0]:
                    identity, alignment_length = self.calculate_sequence_identity(
                        gene1.sequence, gene2.sequence
                    )
                    
                    if identity > best_identity and identity >= self.ortholog_threshold:
                        best_match = gene2
                        best_identity = identity
                        best_alignment_length = alignment_length
                
                # 如果没有ID匹配，基于序列相似性
                elif best_match is None:
                    identity, alignment_length = self.calculate_sequence_identity(
                        gene1.sequence, gene2.sequence
                    )
                    
                    if identity > best_identity and identity >= self.ortholog_threshold:
                        best_match = gene2
                        best_identity = identity
                        best_alignment_length = alignment_length
            
            # 如果找到合适的匹配
            if best_match and best_identity >= self.ortholog_threshold:
                ortholog_pair = OrthologPair(
                    gene1=gene1,
                    gene2=best_match,
                    identity=best_identity,
                    alignment_length=best_alignment_length,
                    gene1_coverage=best_alignment_length / len(gene1.sequence),
                    gene2_coverage=best_alignment_length / len(best_match.sequence)
                )
                ortholog_pairs.append(ortholog_pair)
        
        return ortholog_pairs
    
    def calculate_ani(self, genome1: Genome, genome2: Genome) -> Dict:
        """计算两个基因组间的ANI"""
        
        # 找到同源基因对
        ortholog_pairs = self.find_orthologs(genome1, genome2)
        
        if not ortholog_pairs:
            return {
                'ani': 0.0,
                'ortholog_count': 0,
                'total_genes_genome1': len(genome1.genes),
                'total_genes_genome2': len(genome2.genes),
                'ortholog_ratio': 0.0,
                'weighted_ani': 0.0,
                'identity_distribution': []
            }
        
        # 计算ANI
        identities = [pair.identity for pair in ortholog_pairs]
        alignment_lengths = [pair.alignment_length for pair in ortholog_pairs]
        
        # 简单平均ANI
        simple_ani = np.mean(identities)
        
        # 长度加权ANI
        total_alignment_length = sum(alignment_lengths)
        weighted_ani = sum(pair.identity * pair.alignment_length for pair in ortholog_pairs) / total_alignment_length
        
        # 统计信息
        ortholog_count = len(ortholog_pairs)
        total_genes = max(len(genome1.genes), len(genome2.genes))
        ortholog_ratio = ortholog_count / total_genes
        
        return {
            'ani': simple_ani,
            'weighted_ani': weighted_ani,
            'ortholog_count': ortholog_count,
            'total_genes_genome1': len(genome1.genes),
            'total_genes_genome2': len(genome2.genes),
            'ortholog_ratio': ortholog_ratio,
            'identity_distribution': identities,
            'alignment_lengths': alignment_lengths,
            'ortholog_pairs': ortholog_pairs
        }
    
    def analyze_ortholog_identity_distribution(self, ani_result: Dict) -> Dict:
        """分析同源基因一致性分布"""
        identities = ani_result['identity_distribution']
        
        if not identities:
            return {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'quartiles': [0.0, 0.0, 0.0],
                'bins': [],
                'counts': []
            }
        
        # 基本统计
        mean_identity = np.mean(identities)
        std_identity = np.std(identities)
        median_identity = np.median(identities)
        min_identity = np.min(identities)
        max_identity = np.max(identities)
        
        # 四分位数
        q25, q50, q75 = np.percentile(identities, [25, 50, 75])
        
        # 直方图分布
        bins = np.linspace(0, 1, 21)  # 20个区间
        counts, bin_edges = np.histogram(identities, bins=bins)
        
        return {
            'mean': mean_identity,
            'std': std_identity,
            'median': median_identity,
            'min': min_identity,
            'max': max_identity,
            'quartiles': [q25, q50, q75],
            'bins': bin_edges.tolist(),
            'counts': counts.tolist(),
            'sample_size': len(identities)
        }
    
    def compare_genomes_comprehensive(self, 
                                    ancestral_genome: Genome, 
                                    evolved_genome: Genome) -> Dict:
        """全面比较两个基因组"""
        
        # 计算ANI
        ani_result = self.calculate_ani(ancestral_genome, evolved_genome)
        
        # 分析一致性分布
        identity_distribution = self.analyze_ortholog_identity_distribution(ani_result)
        
        # 基因组组成变化
        ancestral_core_genes = sum(1 for gene in ancestral_genome.genes if gene.is_core)
        evolved_core_genes = sum(1 for gene in evolved_genome.genes if gene.is_core)
        evolved_hgt_genes = sum(1 for gene in evolved_genome.genes if not gene.is_core)
        
        # 基因组大小变化
        size_change = evolved_genome.size - ancestral_genome.size
        size_change_ratio = size_change / ancestral_genome.size if ancestral_genome.size > 0 else 0
        
        # 基因数量变化
        gene_count_change = evolved_genome.gene_count - ancestral_genome.gene_count
        gene_count_change_ratio = gene_count_change / ancestral_genome.gene_count if ancestral_genome.gene_count > 0 else 0
        
        return {
            'ani_analysis': ani_result,
            'identity_distribution': identity_distribution,
            'genome_composition': {
                'ancestral_core_genes': ancestral_core_genes,
                'evolved_core_genes': evolved_core_genes,
                'evolved_hgt_genes': evolved_hgt_genes,
                'core_gene_retention': evolved_core_genes / ancestral_core_genes if ancestral_core_genes > 0 else 0
            },
            'size_changes': {
                'ancestral_size': ancestral_genome.size,
                'evolved_size': evolved_genome.size,
                'size_change': size_change,
                'size_change_ratio': size_change_ratio
            },
            'gene_count_changes': {
                'ancestral_gene_count': ancestral_genome.gene_count,
                'evolved_gene_count': evolved_genome.gene_count,
                'gene_count_change': gene_count_change,
                'gene_count_change_ratio': gene_count_change_ratio
            }
        }