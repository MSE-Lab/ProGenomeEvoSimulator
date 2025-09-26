#!/usr/bin/env python3
"""
Conservation Analyzer - 保守基因分析器
用于计算和分析基因组中保守基因的比例和特征
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from core.genome import Genome, Gene

@dataclass
class ConservationResult:
    """保守性分析结果"""
    gene_id: str
    ancestral_gene_id: str
    identity: float
    is_conservative: bool
    conservation_category: str  # 'highly_conserved', 'moderately_conserved', 'poorly_conserved', 'non_conserved'
    structural_changes: Dict[str, int]  # 结构性变化统计

class ConservationAnalyzer:
    """保守基因分析器"""
    
    def __init__(self, 
                 conservation_threshold: float = 0.3,
                 high_conservation_threshold: float = 0.8,
                 moderate_conservation_threshold: float = 0.6):
        """
        初始化保守性分析器
        
        Args:
            conservation_threshold: 保守基因的最低一致性阈值 (默认0.3)
            high_conservation_threshold: 高度保守基因阈值 (默认0.8)
            moderate_conservation_threshold: 中度保守基因阈值 (默认0.6)
        """
        self.conservation_threshold = conservation_threshold
        self.high_conservation_threshold = high_conservation_threshold
        self.moderate_conservation_threshold = moderate_conservation_threshold
    
    def calculate_sequence_identity(self, seq1: str, seq2: str) -> Tuple[float, Dict[str, int]]:
        """
        计算两个序列的一致性和结构性差异
        
        Returns:
            Tuple[identity, structural_changes]
        """
        if not seq1 or not seq2:
            return 0.0, {'length_diff': abs(len(seq1) - len(seq2))}
        
        # 使用较短序列长度进行比较
        min_length = min(len(seq1), len(seq2))
        max_length = max(len(seq1), len(seq2))
        
        if min_length == 0:
            return 0.0, {'length_diff': max_length}
        
        # 截取相同长度进行比较
        seq1_aligned = seq1[:min_length]
        seq2_aligned = seq2[:min_length]
        
        # 计算匹配数和各种变化类型
        matches = 0
        transitions = 0  # A<->G, C<->T
        transversions = 0  # 其他变化
        
        transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        
        for i, (base1, base2) in enumerate(zip(seq1_aligned, seq2_aligned)):
            if base1 == base2:
                matches += 1
            else:
                if (base1, base2) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
        
        identity = matches / min_length
        
        structural_changes = {
            'length_diff': abs(len(seq1) - len(seq2)),
            'matches': matches,
            'transitions': transitions,
            'transversions': transversions,
            'total_changes': transitions + transversions,
            'alignment_length': min_length
        }
        
        return identity, structural_changes
    
    def categorize_conservation(self, identity: float) -> str:
        """根据一致性对保守程度进行分类"""
        if identity >= self.high_conservation_threshold:
            return 'highly_conserved'
        elif identity >= self.moderate_conservation_threshold:
            return 'moderately_conserved'
        elif identity >= self.conservation_threshold:
            return 'poorly_conserved'
        else:
            return 'non_conserved'
    
    def find_ancestral_gene(self, evolved_gene: Gene, ancestral_genome: Genome) -> Optional[Gene]:
        """
        为进化后的基因找到对应的祖先基因
        
        优先级:
        1. 基因ID完全匹配
        2. 基因ID前缀匹配 (去除后缀)
        3. 序列相似性最高的基因
        """
        # 1. 尝试ID完全匹配
        for ancestral_gene in ancestral_genome.genes:
            if ancestral_gene.id == evolved_gene.id:
                return ancestral_gene
        
        # 2. 尝试ID前缀匹配 (处理基因复制和重命名)
        evolved_base_id = evolved_gene.id.split('_')[0]
        for ancestral_gene in ancestral_genome.genes:
            ancestral_base_id = ancestral_gene.id.split('_')[0]
            if evolved_base_id == ancestral_base_id:
                return ancestral_gene
        
        # 3. 基于序列相似性查找最佳匹配
        best_match = None
        best_identity = 0.0
        
        for ancestral_gene in ancestral_genome.genes:
            identity, _ = self.calculate_sequence_identity(
                evolved_gene.sequence, ancestral_gene.sequence
            )
            if identity > best_identity:
                best_identity = identity
                best_match = ancestral_gene
        
        # 只有当相似性达到一定阈值时才认为是匹配
        if best_identity >= 0.1:  # 最低相似性阈值
            return best_match
        
        return None
    
    def analyze_gene_conservation(self, 
                                evolved_gene: Gene, 
                                ancestral_genome: Genome) -> ConservationResult:
        """分析单个基因的保守性"""
        
        # 查找对应的祖先基因
        ancestral_gene = self.find_ancestral_gene(evolved_gene, ancestral_genome)
        
        if ancestral_gene is None:
            # 如果找不到祖先基因，可能是HGT获得的新基因
            structural_changes = {
                'length_diff': 0,
                'matches': 0,
                'transitions': 0,
                'transversions': 0,
                'total_changes': 0,
                'alignment_length': 0,
                'origin_type': 1  # 1 = HGT_or_new_gene
            }
            
            return ConservationResult(
                gene_id=evolved_gene.id,
                ancestral_gene_id="NOT_FOUND",
                identity=0.0,
                is_conservative=False,
                conservation_category='non_conserved',
                structural_changes=structural_changes
            )
        
        # 计算序列一致性和结构性变化
        identity, structural_changes = self.calculate_sequence_identity(
            evolved_gene.sequence, ancestral_gene.sequence
        )
        
        # 判断是否保守
        is_conservative = identity >= self.conservation_threshold
        conservation_category = self.categorize_conservation(identity)
        
        return ConservationResult(
            gene_id=evolved_gene.id,
            ancestral_gene_id=ancestral_gene.id,
            identity=identity,
            is_conservative=is_conservative,
            conservation_category=conservation_category,
            structural_changes=structural_changes
        )
    
    def analyze_genome_conservation(self, 
                                  evolved_genome: Genome, 
                                  ancestral_genome: Genome) -> Dict[str, Any]:
        """分析整个基因组的保守性"""
        
        print(f"🔬 Analyzing genome conservation...")
        print(f"   Evolved genome: {evolved_genome.gene_count} genes")
        print(f"   Ancestral genome: {ancestral_genome.gene_count} genes")
        print(f"   Conservation threshold: {self.conservation_threshold}")
        
        # 分析每个基因的保守性
        conservation_results = []
        for gene in evolved_genome.genes:
            result = self.analyze_gene_conservation(gene, ancestral_genome)
            conservation_results.append(result)
        
        # 统计保守基因
        conservative_genes = [r for r in conservation_results if r.is_conservative]
        non_conservative_genes = [r for r in conservation_results if not r.is_conservative]
        
        # 按保守程度分类统计
        category_counts = {
            'highly_conserved': 0,
            'moderately_conserved': 0,
            'poorly_conserved': 0,
            'non_conserved': 0
        }
        
        for result in conservation_results:
            category_counts[result.conservation_category] += 1
        
        # 计算保守基因比例
        total_genes = len(conservation_results)
        conservative_ratio = len(conservative_genes) / total_genes if total_genes > 0 else 0
        
        # 分析结构性变化
        structural_analysis = self._analyze_structural_changes(conservation_results)
        
        # 分析进化机制对保守性的影响
        mechanism_impact = self._analyze_mechanism_impact(evolved_genome, conservation_results)
        
        return {
            'total_genes': total_genes,
            'conservative_genes': len(conservative_genes),
            'non_conservative_genes': len(non_conservative_genes),
            'conservative_ratio': conservative_ratio,
            'conservation_categories': category_counts,
            'conservation_results': conservation_results,
            'structural_analysis': structural_analysis,
            'mechanism_impact': mechanism_impact,
            'thresholds': {
                'conservation_threshold': self.conservation_threshold,
                'high_conservation_threshold': self.high_conservation_threshold,
                'moderate_conservation_threshold': self.moderate_conservation_threshold
            }
        }
    
    def _analyze_structural_changes(self, conservation_results: List[ConservationResult]) -> Dict[str, Any]:
        """分析结构性变化统计"""
        
        total_changes = 0
        total_transitions = 0
        total_transversions = 0
        total_length_changes = 0
        
        identity_distribution = []
        
        for result in conservation_results:
            if 'total_changes' in result.structural_changes:
                total_changes += result.structural_changes['total_changes']
            if 'transitions' in result.structural_changes:
                total_transitions += result.structural_changes['transitions']
            if 'transversions' in result.structural_changes:
                total_transversions += result.structural_changes['transversions']
            if 'length_diff' in result.structural_changes:
                total_length_changes += result.structural_changes['length_diff']
            
            identity_distribution.append(result.identity)
        
        # 计算Ti/Tv比率
        ti_tv_ratio = total_transitions / total_transversions if total_transversions > 0 else 0
        
        return {
            'total_sequence_changes': total_changes,
            'total_transitions': total_transitions,
            'total_transversions': total_transversions,
            'ti_tv_ratio': ti_tv_ratio,
            'total_length_changes': total_length_changes,
            'identity_distribution': {
                'mean': np.mean(identity_distribution),
                'std': np.std(identity_distribution),
                'median': np.median(identity_distribution),
                'min': np.min(identity_distribution),
                'max': np.max(identity_distribution)
            }
        }
    
    def _analyze_mechanism_impact(self, evolved_genome: Genome, 
                                conservation_results: List[ConservationResult]) -> Dict[str, Any]:
        """分析不同进化机制对保守性的影响"""
        
        # 统计不同来源基因的保守性
        core_gene_conservation = []
        hgt_gene_conservation = []
        
        for i, gene in enumerate(evolved_genome.genes):
            if i < len(conservation_results):
                result = conservation_results[i]
                if gene.is_core:
                    core_gene_conservation.append(result.identity)
                else:
                    hgt_gene_conservation.append(result.identity)
        
        # 统计突变和重组对保守性的影响
        high_mutation_genes = []  # 高突变基因
        high_recombination_genes = []  # 高重组基因
        
        for i, gene in enumerate(evolved_genome.genes):
            if i < len(conservation_results):
                result = conservation_results[i]
                
                # 高突变基因 (突变数 > 平均值)
                if hasattr(gene, 'mutation_count') and gene.mutation_count > 0:
                    high_mutation_genes.append(result.identity)
                
                # 高重组基因 (重组数 > 0)
                if hasattr(gene, 'recombination_count') and gene.recombination_count > 0:
                    high_recombination_genes.append(result.identity)
        
        return {
            'core_genes': {
                'count': len(core_gene_conservation),
                'mean_identity': np.mean(core_gene_conservation) if core_gene_conservation else 0,
                'conservative_ratio': sum(1 for x in core_gene_conservation if x >= self.conservation_threshold) / len(core_gene_conservation) if core_gene_conservation else 0
            },
            'hgt_genes': {
                'count': len(hgt_gene_conservation),
                'mean_identity': np.mean(hgt_gene_conservation) if hgt_gene_conservation else 0,
                'conservative_ratio': sum(1 for x in hgt_gene_conservation if x >= self.conservation_threshold) / len(hgt_gene_conservation) if hgt_gene_conservation else 0
            },
            'high_mutation_genes': {
                'count': len(high_mutation_genes),
                'mean_identity': np.mean(high_mutation_genes) if high_mutation_genes else 0,
                'conservative_ratio': sum(1 for x in high_mutation_genes if x >= self.conservation_threshold) / len(high_mutation_genes) if high_mutation_genes else 0
            },
            'high_recombination_genes': {
                'count': len(high_recombination_genes),
                'mean_identity': np.mean(high_recombination_genes) if high_recombination_genes else 0,
                'conservative_ratio': sum(1 for x in high_recombination_genes if x >= self.conservation_threshold) / len(high_recombination_genes) if high_recombination_genes else 0
            }
        }
    
    def print_conservation_summary(self, analysis_result: Dict[str, Any]):
        """打印保守性分析摘要"""
        
        print("\n" + "=" * 60)
        print("🧬 GENOME CONSERVATION ANALYSIS SUMMARY")
        print("=" * 60)
        
        # 基本统计
        print(f"📊 Basic Statistics:")
        print(f"   Total genes analyzed: {analysis_result['total_genes']:,}")
        print(f"   Conservative genes: {analysis_result['conservative_genes']:,}")
        print(f"   Non-conservative genes: {analysis_result['non_conservative_genes']:,}")
        print(f"   Conservative gene ratio: {analysis_result['conservative_ratio']:.3f} ({analysis_result['conservative_ratio']*100:.1f}%)")
        
        # 保守程度分类
        print(f"\n🎯 Conservation Categories:")
        categories = analysis_result['conservation_categories']
        total = analysis_result['total_genes']
        
        for category, count in categories.items():
            percentage = count / total * 100 if total > 0 else 0
            category_name = category.replace('_', ' ').title()
            print(f"   {category_name}: {count:,} ({percentage:.1f}%)")
        
        # 结构性变化分析
        structural = analysis_result['structural_analysis']
        print(f"\n🔬 Structural Changes:")
        print(f"   Total sequence changes: {structural['total_sequence_changes']:,}")
        print(f"   Transitions: {structural['total_transitions']:,}")
        print(f"   Transversions: {structural['total_transversions']:,}")
        print(f"   Ti/Tv ratio: {structural['ti_tv_ratio']:.2f}")
        print(f"   Total length changes: {structural['total_length_changes']:,} bp")
        
        # 一致性分布
        identity_dist = structural['identity_distribution']
        print(f"\n📈 Identity Distribution:")
        print(f"   Mean identity: {identity_dist['mean']:.3f}")
        print(f"   Standard deviation: {identity_dist['std']:.3f}")
        print(f"   Range: {identity_dist['min']:.3f} - {identity_dist['max']:.3f}")
        
        # 进化机制影响
        mechanism = analysis_result['mechanism_impact']
        print(f"\n⚙️  Evolution Mechanism Impact:")
        
        core_genes = mechanism['core_genes']
        hgt_genes = mechanism['hgt_genes']
        
        print(f"   Core genes: {core_genes['count']:,} genes, {core_genes['conservative_ratio']:.3f} conservative ratio")
        print(f"   HGT genes: {hgt_genes['count']:,} genes, {hgt_genes['conservative_ratio']:.3f} conservative ratio")
        
        if mechanism['high_mutation_genes']['count'] > 0:
            mut_genes = mechanism['high_mutation_genes']
            print(f"   High-mutation genes: {mut_genes['count']:,} genes, {mut_genes['conservative_ratio']:.3f} conservative ratio")
        
        if mechanism['high_recombination_genes']['count'] > 0:
            rec_genes = mechanism['high_recombination_genes']
            print(f"   High-recombination genes: {rec_genes['count']:,} genes, {rec_genes['conservative_ratio']:.3f} conservative ratio")
        
        print("=" * 60)
    
    def get_non_conservative_genes(self, analysis_result: Dict[str, Any]) -> List[ConservationResult]:
        """获取非保守基因列表"""
        return [r for r in analysis_result['conservation_results'] if not r.is_conservative]
    
    def get_highly_diverged_genes(self, analysis_result: Dict[str, Any], 
                                 divergence_threshold: float = 0.1) -> List[ConservationResult]:
        """获取高度分化的基因（一致性极低）"""
        return [r for r in analysis_result['conservation_results'] 
                if r.identity < divergence_threshold]