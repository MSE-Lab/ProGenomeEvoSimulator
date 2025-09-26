#!/usr/bin/env python3
"""
Molecular Evolution Engine - 基于分子进化理论的优化进化引擎

基于现代分子进化理论的关键概念：
1. 选择压力和基因功能重要性
2. 密码子使用偏好性
3. 同义/非同义突变的选择效应
4. 基因长度约束和功能域保护
5. 更真实的HGT和重组机制
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from core.genome import Genome, Gene
from mechanisms.point_mutation import PointMutationEngine
from mechanisms.horizontal_transfer import HorizontalGeneTransfer
from mechanisms.homologous_recombination import HomologousRecombination

@dataclass
class GeneFunction:
    """基因功能分类"""
    category: str  # 'essential', 'important', 'accessory', 'dispensable'
    selection_coefficient: float  # 选择系数 (负值表示有害突变的选择压力)
    conservation_level: float  # 保守程度 (0-1)
    codon_bias_strength: float  # 密码子偏好性强度
    functional_domains: List[Tuple[int, int]]  # 功能域位置 [(start, end), ...]

class MolecularEvolutionEngine:
    """基于分子进化理论的进化引擎"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-9,
                 hgt_rate: float = 0.001,
                 recombination_rate: float = 1e-6,
                 enable_selection: bool = True,
                 enable_codon_bias: bool = True,
                 enable_functional_constraints: bool = True):
        
        # 基础进化机制
        self.base_mutation_rate = mutation_rate
        self.hgt_rate = hgt_rate
        self.recombination_rate = recombination_rate
        
        # 分子进化特性开关
        self.enable_selection = enable_selection
        self.enable_codon_bias = enable_codon_bias
        self.enable_functional_constraints = enable_functional_constraints
        
        # 初始化进化机制
        self._setup_evolution_mechanisms()
        
        # 基因功能分类
        self.gene_functions = {}
        
        # 密码子表和偏好性
        self._setup_codon_usage()
        
        # 选择压力参数
        self.selection_parameters = {
            'essential_gene_protection': 0.9,  # 必需基因保护强度
            'synonymous_neutral_rate': 0.95,   # 同义突变中性比例
            'nonsynonymous_deleterious_rate': 0.7,  # 非同义突变有害比例
            'functional_domain_protection': 0.95,   # 功能域保护强度
        }
        
        # 统计信息
        self.evolution_stats = {
            'synonymous_mutations': 0,
            'nonsynonymous_mutations': 0,
            'selected_against_mutations': 0,
            'neutral_mutations': 0,
            'beneficial_mutations': 0,
            'functional_domain_mutations': 0,
        }
    
    def _setup_evolution_mechanisms(self):
        """初始化基础进化机制"""
        # 使用增强的点突变引擎
        self.point_mutation = PointMutationEngine(
            mutation_rate=self.base_mutation_rate,
            enable_transition_bias=True,
            transition_transversion_ratio=2.5,  # 更接近真实的Ti/Tv比例
            enable_hotspots=True,
            hotspot_multiplier=3.0,
            hotspot_motifs=['CG', 'GC', 'CCWGG', 'GCWGC']  # 包含限制酶位点
        )
        
        # HGT机制 - 添加选择性
        self.hgt = HorizontalGeneTransfer(self.hgt_rate)
        
        # 同源重组 - 提高相似性要求
        self.recombination = HomologousRecombination(
            recombination_rate=self.recombination_rate,
            min_similarity=0.8,  # 提高最小相似性要求
            min_recombination_length=100,
            max_recombination_length=500
        )
    
    def _setup_codon_usage(self):
        """设置密码子使用偏好性"""
        # 标准遗传密码表
        self.genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        # 原核生物典型的密码子偏好性 (基于E.coli)
        self.codon_preferences = {
            'F': {'TTT': 0.45, 'TTC': 0.55},
            'L': {'TTA': 0.13, 'TTG': 0.13, 'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 0.48},
            'S': {'TCT': 0.15, 'TCC': 0.15, 'TCA': 0.12, 'TCG': 0.15, 'AGT': 0.15, 'AGC': 0.28},
            'Y': {'TAT': 0.43, 'TAC': 0.57},
            'C': {'TGT': 0.45, 'TGC': 0.55},
            'W': {'TGG': 1.0},
            'P': {'CCT': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 0.49},
            'H': {'CAT': 0.42, 'CAC': 0.58},
            'Q': {'CAA': 0.35, 'CAG': 0.65},
            'R': {'CGT': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.10, 'AGA': 0.07, 'AGG': 0.04},
            'I': {'ATT': 0.49, 'ATC': 0.39, 'ATA': 0.12},
            'M': {'ATG': 1.0},
            'T': {'ACT': 0.19, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.24},
            'N': {'AAT': 0.45, 'AAC': 0.55},
            'K': {'AAA': 0.76, 'AAG': 0.24},
            'V': {'GTT': 0.26, 'GTC': 0.20, 'GTA': 0.17, 'GTG': 0.37},
            'A': {'GCT': 0.18, 'GCC': 0.26, 'GCA': 0.23, 'GCG': 0.33},
            'D': {'GAT': 0.62, 'GAC': 0.38},
            'E': {'GAA': 0.69, 'GAG': 0.31},
            'G': {'GGT': 0.35, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.15}
        }
    
    def classify_gene_function(self, gene: Gene) -> GeneFunction:
        """基于基因特征分类功能重要性"""
        # 简化的功能分类逻辑
        gene_length = len(gene.sequence)
        
        # 基于基因长度和ID模式进行功能分类
        if gene.origin == "hgt":
            # HGT基因通常是辅助功能
            category = "accessory"
            selection_coefficient = -0.1
            conservation_level = 0.3
            codon_bias_strength = 0.2
        elif gene_length < 300:
            # 短基因可能是调节基因或小蛋白
            category = "important"
            selection_coefficient = -0.3
            conservation_level = 0.6
            codon_bias_strength = 0.4
        elif gene_length > 2000:
            # 长基因可能是重要的结构或酶基因
            category = "essential"
            selection_coefficient = -0.8
            conservation_level = 0.9
            codon_bias_strength = 0.8
        else:
            # 中等长度基因
            category = "important"
            selection_coefficient = -0.5
            conservation_level = 0.7
            codon_bias_strength = 0.6
        
        # 预测功能域 (简化版本)
        functional_domains = []
        if gene_length > 500:
            # 假设长基因有保守的功能域
            domain_length = min(150, gene_length // 4)
            start_pos = gene_length // 4
            functional_domains.append((start_pos, start_pos + domain_length))
        
        return GeneFunction(
            category=category,
            selection_coefficient=selection_coefficient,
            conservation_level=conservation_level,
            codon_bias_strength=codon_bias_strength,
            functional_domains=functional_domains
        )
    
    def is_synonymous_mutation(self, original_codon: str, mutated_codon: str) -> bool:
        """判断是否为同义突变"""
        if len(original_codon) != 3 or len(mutated_codon) != 3:
            return False
        
        original_aa = self.genetic_code.get(original_codon, 'X')
        mutated_aa = self.genetic_code.get(mutated_codon, 'X')
        
        return original_aa == mutated_aa
    
    def calculate_selection_effect(self, gene: Gene, position: int, 
                                 original_base: str, mutated_base: str) -> float:
        """计算选择效应强度"""
        if not self.enable_selection:
            return 1.0  # 无选择压力
        
        # 获取基因功能分类
        gene_function = self.gene_functions.get(gene.id)
        if not gene_function:
            gene_function = self.classify_gene_function(gene)
            self.gene_functions[gene.id] = gene_function
        
        # 基础选择强度
        base_selection = 1.0
        
        # 检查是否在功能域内
        in_functional_domain = False
        if self.enable_functional_constraints:
            for domain_start, domain_end in gene_function.functional_domains:
                if domain_start <= position <= domain_end:
                    in_functional_domain = True
                    break
        
        # 检查密码子效应
        codon_position = position % 3
        codon_start = position - codon_position
        
        if codon_start + 2 < len(gene.sequence):
            original_codon = gene.sequence[codon_start:codon_start + 3]
            mutated_sequence = list(gene.sequence)
            mutated_sequence[position] = mutated_base
            mutated_codon = ''.join(mutated_sequence[codon_start:codon_start + 3])
            
            is_synonymous = self.is_synonymous_mutation(original_codon, mutated_codon)
            
            if is_synonymous:
                # 同义突变 - 大部分中性，但受密码子偏好性影响
                if self.enable_codon_bias:
                    # 检查密码子偏好性
                    aa = self.genetic_code.get(original_codon, 'X')
                    if aa in self.codon_preferences:
                        original_freq = self.codon_preferences[aa].get(original_codon, 0.1)
                        mutated_freq = self.codon_preferences[aa].get(mutated_codon, 0.1)
                        
                        # 偏好性差异影响选择
                        codon_effect = mutated_freq / original_freq
                        base_selection *= (1.0 + gene_function.codon_bias_strength * (codon_effect - 1.0))
                
                self.evolution_stats['synonymous_mutations'] += 1
            else:
                # 非同义突变 - 受强选择压力
                nonsynonymous_effect = 1.0 + gene_function.selection_coefficient
                base_selection *= nonsynonymous_effect
                self.evolution_stats['nonsynonymous_mutations'] += 1
        
        # 功能域保护
        if in_functional_domain:
            domain_protection = self.selection_parameters['functional_domain_protection']
            base_selection *= (1.0 - domain_protection)
            self.evolution_stats['functional_domain_mutations'] += 1
        
        # 基因重要性影响
        if gene_function.category == "essential":
            base_selection *= self.selection_parameters['essential_gene_protection']
        
        return max(0.01, base_selection)  # 确保选择效应不为零
    
    def apply_molecular_selection(self, genome: Genome, mutations: List[Tuple[Gene, int, str, str]]) -> int:
        """应用分子选择压力过滤突变"""
        if not self.enable_selection:
            # 无选择压力，应用所有突变
            for gene, position, original_base, mutated_base in mutations:
                gene.mutate_position(position, mutated_base)
            return len(mutations)
        
        successful_mutations = 0
        
        for gene, position, original_base, mutated_base in mutations:
            # 计算选择效应
            selection_effect = self.calculate_selection_effect(
                gene, position, original_base, mutated_base
            )
            
            # 基于选择效应决定是否接受突变
            if random.random() < selection_effect:
                gene.mutate_position(position, mutated_base)
                successful_mutations += 1
                
                # 统计突变类型
                if selection_effect > 0.95:
                    self.evolution_stats['neutral_mutations'] += 1
                elif selection_effect > 1.0:
                    self.evolution_stats['beneficial_mutations'] += 1
                else:
                    self.evolution_stats['selected_against_mutations'] += 1
        
        return successful_mutations
    
    def enhanced_hgt_selection(self, genome: Genome, donor_genes: List[Gene]) -> List[Gene]:
        """基于功能需求选择HGT基因"""
        if not self.enable_selection:
            return donor_genes
        
        # 简化的HGT选择逻辑
        selected_genes = []
        
        for donor_gene in donor_genes:
            # 基于基因长度和序列特征评估有用性
            gene_length = len(donor_gene.sequence)
            
            # 偏好中等长度的基因 (可能编码有用的酶)
            if 500 <= gene_length <= 1500:
                acceptance_prob = 0.8
            elif 300 <= gene_length <= 2000:
                acceptance_prob = 0.5
            else:
                acceptance_prob = 0.2
            
            # 检查是否与现有基因过于相似 (避免冗余)
            is_redundant = False
            for existing_gene in genome.genes:
                if self._calculate_sequence_similarity(donor_gene.sequence, existing_gene.sequence) > 0.9:
                    is_redundant = True
                    break
            
            if not is_redundant and random.random() < acceptance_prob:
                selected_genes.append(donor_gene)
        
        return selected_genes
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """计算序列相似性"""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / min_len
    
    def evolve_one_generation(self, genome: Genome) -> Dict:
        """进化一代 - 整合分子进化机制"""
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0,
            'selection_stats': {}
        }
        
        # 1. 收集潜在突变 (不立即应用)
        potential_mutations = []
        for gene in genome.genes:
            gene_mutations = self.point_mutation.calculate_mutations_per_gene(gene, generations=1)
            for position, mutation_rate in gene_mutations:
                if random.random() < mutation_rate:
                    original_base = gene.sequence[position]
                    mutated_base = self.point_mutation.get_mutated_base(original_base)
                    if mutated_base != original_base:
                        potential_mutations.append((gene, position, original_base, mutated_base))
        
        # 2. 应用分子选择压力
        successful_mutations = self.apply_molecular_selection(genome, potential_mutations)
        generation_stats['mutations'] = successful_mutations
        genome.total_mutations += successful_mutations
        
        # 3. 应用增强的HGT
        hgt_events = self.hgt.calculate_hgt_events(generations=1)
        if hgt_events > 0:
            donor_genes = [self.hgt.select_donor_gene() for _ in range(hgt_events)]
            selected_genes = self.enhanced_hgt_selection(genome, donor_genes)
            
            for donor_gene in selected_genes:
                if self.hgt.insert_gene(genome, donor_gene):
                    generation_stats['hgt_events'] += 1
                    genome.total_hgt_events += 1
        
        # 4. 应用同源重组
        recombination_events = self.recombination.apply_recombination(genome, generations=1)
        generation_stats['recombination_events'] = recombination_events
        
        # 更新代数
        genome.generation += 1
        
        # 记录选择统计
        generation_stats['selection_stats'] = self.evolution_stats.copy()
        generation_stats['final_stats'] = genome.get_statistics()
        
        return generation_stats
    
    def simulate_molecular_evolution(self, 
                                   initial_genome: Genome, 
                                   generations: int,
                                   save_snapshots: bool = True,
                                   snapshot_interval: int = 100) -> Tuple[Genome, List[Dict]]:
        """完整的分子进化模拟"""
        
        print("🧬 MOLECULAR EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"🔬 Molecular features: Selection pressure, Codon bias, Functional constraints")
        print(f"⚙️  Selection enabled: {self.enable_selection}")
        print(f"🧮 Codon bias enabled: {self.enable_codon_bias}")
        print(f"🛡️  Functional constraints: {self.enable_functional_constraints}")
        print("=" * 80)
        
        # 创建基因组副本
        evolving_genome = initial_genome.copy()
        simulation_start_time = time.time()
        
        # 初始化基因功能分类
        print("🔍 Classifying gene functions...")
        for gene in evolving_genome.genes:
            self.gene_functions[gene.id] = self.classify_gene_function(gene)
        
        # 记录初始状态
        snapshots = []
        if save_snapshots:
            initial_summary = self.get_evolution_summary(evolving_genome)
            initial_summary['snapshot_generation'] = 0
            snapshots.append(initial_summary)
        
        # 进化过程
        print(f"🚀 Starting molecular evolution...")
        history = []
        
        for gen in range(generations):
            gen_stats = self.evolve_one_generation(evolving_genome)
            history.append(gen_stats)
            
            # 显示进度
            if (gen + 1) % max(1, generations // 20) == 0:
                progress = (gen + 1) / generations * 100
                print(f"Progress: {progress:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"Genes: {evolving_genome.gene_count:,} | "
                      f"Mutations: {evolving_genome.total_mutations:,}")
            
            # 保存快照
            if save_snapshots and (gen + 1) % snapshot_interval == 0:
                snapshot = self.get_evolution_summary(evolving_genome)
                snapshot['snapshot_generation'] = gen + 1
                snapshots.append(snapshot)
        
        # 最终总结
        total_time = time.time() - simulation_start_time
        final_summary = self.get_evolution_summary(evolving_genome)
        
        print(f"\n🎉 MOLECULAR EVOLUTION COMPLETED!")
        print(f"🧬 Final genome: {evolving_genome.gene_count:,} genes, {evolving_genome.size:,} bp")
        print(f"📈 Changes: {evolving_genome.size - initial_genome.size:+,} bp, "
              f"{evolving_genome.gene_count - initial_genome.gene_count:+,} genes")
        print(f"⏱️  Total time: {total_time/60:.2f} minutes")
        
        # 显示分子进化统计
        self.print_molecular_evolution_summary()
        
        return evolving_genome, snapshots
    
    def get_evolution_summary(self, genome: Genome) -> Dict:
        """获取进化总结"""
        base_summary = {
            'genome_stats': genome.get_statistics(),
            'mutation_stats': self.point_mutation.get_mutation_statistics(genome),
            'hgt_stats': self.hgt.get_hgt_statistics(genome),
            'recombination_stats': self.recombination.get_recombination_statistics(genome),
        }
        
        # 添加分子进化特有的统计
        base_summary['molecular_evolution_stats'] = {
            'selection_enabled': self.enable_selection,
            'codon_bias_enabled': self.enable_codon_bias,
            'functional_constraints_enabled': self.enable_functional_constraints,
            'evolution_stats': self.evolution_stats.copy(),
            'gene_function_distribution': self._get_gene_function_distribution(genome)
        }
        
        return base_summary
    
    def _get_gene_function_distribution(self, genome: Genome) -> Dict:
        """获取基因功能分布统计"""
        distribution = {'essential': 0, 'important': 0, 'accessory': 0, 'dispensable': 0}
        
        for gene in genome.genes:
            gene_function = self.gene_functions.get(gene.id)
            if gene_function:
                distribution[gene_function.category] += 1
        
        return distribution
    
    def print_molecular_evolution_summary(self):
        """打印分子进化总结"""
        print("\n" + "=" * 60)
        print("🔬 MOLECULAR EVOLUTION SUMMARY")
        print("=" * 60)
        
        total_mutations = sum(self.evolution_stats.values())
        if total_mutations > 0:
            print(f"📊 Mutation Analysis:")
            print(f"   Synonymous mutations: {self.evolution_stats['synonymous_mutations']:,} "
                  f"({self.evolution_stats['synonymous_mutations']/total_mutations*100:.1f}%)")
            print(f"   Non-synonymous mutations: {self.evolution_stats['nonsynonymous_mutations']:,} "
                  f"({self.evolution_stats['nonsynonymous_mutations']/total_mutations*100:.1f}%)")
            print(f"   Neutral mutations: {self.evolution_stats['neutral_mutations']:,}")
            print(f"   Selected against: {self.evolution_stats['selected_against_mutations']:,}")
            print(f"   Beneficial mutations: {self.evolution_stats['beneficial_mutations']:,}")
            print(f"   Functional domain hits: {self.evolution_stats['functional_domain_mutations']:,}")
            
            # 计算dN/dS比率的近似值
            if self.evolution_stats['synonymous_mutations'] > 0:
                dn_ds_ratio = (self.evolution_stats['nonsynonymous_mutations'] / 
                             self.evolution_stats['synonymous_mutations'])
                print(f"   Approximate dN/dS ratio: {dn_ds_ratio:.3f}")
        
        print("=" * 60)