#!/usr/bin/env python3
"""
Gene Loss Mechanism
基因丢失机制 - 模拟原核生物进化中的基因删除事件
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.genome import Genome, Gene


class GeneLossEngine:
    """基因丢失引擎 - 模拟基因删除和基因组精简"""
    
    def __init__(self, 
                 loss_rate: float = 1e-7,  # 修正：更低的基础丢失率
                 core_gene_protection: float = 0.98,  # 修正：更强的核心基因保护
                 hgt_gene_loss_multiplier: float = 20.0,  # 修正：HGT基因更容易丢失
                 min_genome_size: int = 1200,  # 修正：更合理的最小基因组大小
                 min_core_genes: int = 1000,  # 修正：更多的必需核心基因
                 enable_size_pressure: bool = True,
                 optimal_genome_size: int = 3000):
        """
        初始化基因丢失引擎
        
        Args:
            loss_rate: 基础基因丢失率（每代每基因）
            core_gene_protection: 核心基因保护系数（0-1，越高越不容易丢失）
            hgt_gene_loss_multiplier: HGT基因丢失倍数（相对于核心基因）
            min_genome_size: 最小基因组大小（基因数）
            min_core_genes: 最小核心基因数
            enable_size_pressure: 是否启用基因组大小压力
            optimal_genome_size: 最优基因组大小
        """
        self.loss_rate = loss_rate
        self.core_gene_protection = core_gene_protection
        self.hgt_gene_loss_multiplier = hgt_gene_loss_multiplier
        self.min_genome_size = min_genome_size
        self.min_core_genes = min_core_genes
        self.enable_size_pressure = enable_size_pressure
        self.optimal_genome_size = optimal_genome_size
        
        # 统计信息
        self.loss_stats = {
            'total_genes_lost': 0,
            'core_genes_lost': 0,
            'hgt_genes_lost': 0,
            'size_pressure_losses': 0,
            'random_losses': 0
        }
        
        print(f"🗑️  Gene Loss Engine initialized:")
        print(f"   Base loss rate: {loss_rate}")
        print(f"   Core gene protection: {core_gene_protection*100:.1f}%")
        print(f"   HGT gene loss multiplier: {hgt_gene_loss_multiplier}x")
        print(f"   Min genome size: {min_genome_size} genes")
    
    def calculate_gene_loss_probability(self, gene: Gene, genome: Genome) -> float:
        """
        计算特定基因的丢失概率
        
        Args:
            gene: 要评估的基因
            genome: 当前基因组
            
        Returns:
            基因丢失概率（0-1）
        """
        base_prob = self.loss_rate
        
        # 1. 核心基因保护
        if gene.is_core:
            # 核心基因受到强保护
            base_prob *= (1.0 - self.core_gene_protection)
        else:
            # HGT基因更容易丢失
            if gene.origin == "hgt":
                base_prob *= self.hgt_gene_loss_multiplier
        
        # 2. 基因组大小压力
        if self.enable_size_pressure and genome.gene_count > self.optimal_genome_size:
            size_pressure = (genome.gene_count - self.optimal_genome_size) / self.optimal_genome_size
            # 基因组越大，丢失压力越大
            base_prob *= (1.0 + size_pressure * 2.0)
        
        # 3. 基因质量评估（基于突变负荷）
        if gene.mutation_count > 0:
            mutation_burden = gene.mutation_count / gene.length
            # 高突变负荷的基因更容易丢失
            base_prob *= (1.0 + mutation_burden * 5.0)
        
        # 确保概率在合理范围内
        return min(0.1, max(0.0, base_prob))  # 最大10%的丢失概率
    
    def can_lose_gene(self, gene: Gene, genome: Genome) -> bool:
        """
        检查基因是否可以被丢失（安全检查）
        
        Args:
            gene: 要检查的基因
            genome: 当前基因组
            
        Returns:
            是否可以安全丢失该基因
        """
        # 1. 检查最小基因组大小
        if genome.gene_count <= self.min_genome_size:
            return False
        
        # 2. 检查最小核心基因数
        if gene.is_core and genome.core_gene_count <= self.min_core_genes:
            return False
        
        # 3. 其他安全检查可以在这里添加
        # 例如：检查基因间依赖关系、必需基因标记等
        
        return True
    
    def select_genes_for_loss(self, genome: Genome) -> List[Gene]:
        """
        选择要丢失的基因
        
        Args:
            genome: 当前基因组
            
        Returns:
            要丢失的基因列表
        """
        genes_to_lose = []
        
        for gene in genome.genes:
            # 检查是否可以丢失
            if not self.can_lose_gene(gene, genome):
                continue
            
            # 计算丢失概率
            loss_prob = self.calculate_gene_loss_probability(gene, genome)
            
            # 随机决定是否丢失
            if random.random() < loss_prob:
                genes_to_lose.append(gene)
        
        return genes_to_lose
    
    def apply_gene_loss(self, genome: Genome, generations: int = 1) -> int:
        """
        应用基因丢失到基因组
        
        Args:
            genome: 要修改的基因组
            generations: 代数（用于调整丢失率）
            
        Returns:
            丢失的基因数量
        """
        total_lost = 0
        
        for _ in range(generations):
            # 选择要丢失的基因
            genes_to_lose = self.select_genes_for_loss(genome)
            
            if not genes_to_lose:
                continue
            
            # 执行基因丢失
            for gene in genes_to_lose:
                # 从基因组中移除基因
                genome.remove_gene(gene.id)
                total_lost += 1
                
                # 更新统计信息
                self.loss_stats['total_genes_lost'] += 1
                
                if gene.is_core:
                    self.loss_stats['core_genes_lost'] += 1
                else:
                    self.loss_stats['hgt_genes_lost'] += 1
                
                # 判断丢失原因
                if genome.gene_count > self.optimal_genome_size:
                    self.loss_stats['size_pressure_losses'] += 1
                else:
                    self.loss_stats['random_losses'] += 1
        
        return total_lost
    
    def get_loss_statistics(self, genome: Genome) -> Dict:
        """
        获取基因丢失统计信息
        
        Args:
            genome: 当前基因组
            
        Returns:
            统计信息字典
        """
        total_lost = self.loss_stats['total_genes_lost']
        
        # 计算丢失率
        core_loss_rate = 0
        hgt_loss_rate = 0
        
        if genome.generation > 0:
            # 估算每代丢失率
            core_loss_rate = self.loss_stats['core_genes_lost'] / genome.generation
            hgt_loss_rate = self.loss_stats['hgt_genes_lost'] / genome.generation
        
        return {
            'total_genes_lost': total_lost,
            'core_genes_lost': self.loss_stats['core_genes_lost'],
            'hgt_genes_lost': self.loss_stats['hgt_genes_lost'],
            'size_pressure_losses': self.loss_stats['size_pressure_losses'],
            'random_losses': self.loss_stats['random_losses'],
            
            # 比率统计
            'core_loss_percentage': (self.loss_stats['core_genes_lost'] / total_lost * 100) if total_lost > 0 else 0,
            'hgt_loss_percentage': (self.loss_stats['hgt_genes_lost'] / total_lost * 100) if total_lost > 0 else 0,
            
            # 丢失率
            'avg_core_loss_per_generation': core_loss_rate,
            'avg_hgt_loss_per_generation': hgt_loss_rate,
            'avg_total_loss_per_generation': total_lost / genome.generation if genome.generation > 0 else 0,
            
            # 当前基因组状态
            'current_genome_size': genome.gene_count,
            'current_core_genes': genome.core_gene_count,
            'current_hgt_genes': genome.hgt_gene_count,
            'size_pressure_active': genome.gene_count > self.optimal_genome_size,
            
            # 配置信息
            'loss_rate': self.loss_rate,
            'core_protection': self.core_gene_protection,
            'hgt_multiplier': self.hgt_gene_loss_multiplier,
            'min_genome_size': self.min_genome_size
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.loss_stats = {
            'total_genes_lost': 0,
            'core_genes_lost': 0,
            'hgt_genes_lost': 0,
            'size_pressure_losses': 0,
            'random_losses': 0
        }
    
    def print_loss_summary(self, genome: Genome):
        """打印基因丢失摘要"""
        stats = self.get_loss_statistics(genome)
        
        print("\n" + "=" * 50)
        print("🗑️  GENE LOSS SUMMARY")
        print("=" * 50)
        
        print(f"📊 Loss Statistics:")
        print(f"   Total genes lost: {stats['total_genes_lost']}")
        print(f"   Core genes lost: {stats['core_genes_lost']} ({stats['core_loss_percentage']:.1f}%)")
        print(f"   HGT genes lost: {stats['hgt_genes_lost']} ({stats['hgt_loss_percentage']:.1f}%)")
        print()
        
        print(f"📈 Loss Rates (per generation):")
        print(f"   Total loss rate: {stats['avg_total_loss_per_generation']:.3f}")
        print(f"   Core gene loss rate: {stats['avg_core_loss_per_generation']:.3f}")
        print(f"   HGT gene loss rate: {stats['avg_hgt_loss_per_generation']:.3f}")
        print()
        
        print(f"🧬 Current Genome Status:")
        print(f"   Total genes: {stats['current_genome_size']}")
        print(f"   Core genes: {stats['current_core_genes']}")
        print(f"   HGT genes: {stats['current_hgt_genes']}")
        print(f"   Size pressure: {'Active' if stats['size_pressure_active'] else 'Inactive'}")
        print()
        
        print(f"🎯 Loss Mechanisms:")
        print(f"   Size pressure losses: {stats['size_pressure_losses']}")
        print(f"   Random losses: {stats['random_losses']}")
        
        print("=" * 50)
    
    def analyze_loss_patterns(self, genome: Genome) -> Dict:
        """分析基因丢失模式"""
        stats = self.get_loss_statistics(genome)
        
        # 分析丢失偏好
        total_lost = stats['total_genes_lost']
        if total_lost == 0:
            return {'error': 'No genes lost yet'}
        
        # HGT基因丢失偏好分析
        hgt_loss_bias = stats['hgt_loss_percentage'] / 100.0
        expected_hgt_ratio = genome.hgt_gene_count / genome.gene_count if genome.gene_count > 0 else 0
        
        hgt_loss_enrichment = hgt_loss_bias / expected_hgt_ratio if expected_hgt_ratio > 0 else 0
        
        # 基因组大小变化趋势
        size_change_rate = -stats['avg_total_loss_per_generation']  # 负值表示缩小
        
        return {
            'hgt_loss_enrichment': hgt_loss_enrichment,  # >1表示HGT基因更容易丢失
            'genome_size_trend': size_change_rate,
            'loss_efficiency': total_lost / genome.generation if genome.generation > 0 else 0,
            'size_pressure_contribution': stats['size_pressure_losses'] / total_lost if total_lost > 0 else 0,
            'protection_effectiveness': 1.0 - (stats['core_genes_lost'] / (stats['core_genes_lost'] + stats['hgt_genes_lost'])) if total_lost > 0 else 1.0
        }