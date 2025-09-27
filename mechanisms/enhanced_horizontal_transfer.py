#!/usr/bin/env python3
"""
Enhanced Horizontal Gene Transfer - 基于分子进化理论的改进HGT机制

改进特性：
1. 基于生态位和功能需求的基因选择
2. 转移屏障和兼容性检查
3. 基因表达调控兼容性
4. 代谢网络整合考虑
5. 更真实的基因来源多样性
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from core.genome import Genome, Gene, generate_random_sequence

@dataclass
class GeneOrigin:
    """基因来源信息"""
    source_type: str  # 'plasmid', 'phage', 'transposon', 'chromosome'
    gc_content: float
    codon_usage_bias: float
    metabolic_category: str  # 'central', 'secondary', 'stress_response', 'virulence'
    transfer_frequency: float  # 转移频率

class EnhancedHorizontalGeneTransfer:
    """增强的横向基因转移引擎"""
    
    def __init__(self, 
                 hgt_rate: float = 1e-5,  # 修正：更真实的HGT率
                 gene_pool_size: int = 10000,
                 enable_transfer_barriers: bool = True,
                 enable_metabolic_integration: bool = True,
                 gc_content_tolerance: float = 0.10):  # 修正：更严格的GC含量容忍度
        
        self.hgt_rate = hgt_rate
        self.gene_pool_size = gene_pool_size
        self.enable_transfer_barriers = enable_transfer_barriers
        self.enable_metabolic_integration = enable_metabolic_integration
        self.gc_content_tolerance = gc_content_tolerance
        
        # 创建多样化的基因池
        self.external_gene_pools = self._create_diverse_gene_pools()
        
        # 转移机制权重
        self.transfer_mechanisms = {
            'conjugation': 0.4,      # 接合转移 - 主要来源
            'transformation': 0.3,   # 转化 - DNA摄取
            'transduction': 0.2,     # 转导 - 噬菌体介导
            'vesicle_transfer': 0.1  # 膜泡转移 - 新发现的机制
        }
        
        # 功能类别偏好性
        self.functional_preferences = {
            'antibiotic_resistance': 0.8,    # 抗生素抗性基因 - 高选择压力
            'metal_resistance': 0.7,         # 重金属抗性
            'stress_response': 0.6,          # 应激反应
            'secondary_metabolism': 0.5,     # 次级代谢
            'virulence_factors': 0.4,        # 毒力因子
            'central_metabolism': 0.2,       # 中心代谢 - 低转移率
            'housekeeping': 0.1              # 管家基因 - 很少转移
        }
        
        # 统计信息
        self.hgt_stats = {
            'successful_transfers': 0,
            'rejected_by_barriers': 0,
            'rejected_by_integration': 0,
            'transfer_by_mechanism': {mech: 0 for mech in self.transfer_mechanisms},
            'transfer_by_function': {func: 0 for func in self.functional_preferences}
        }
    
    def _create_diverse_gene_pools(self) -> Dict[str, List[Gene]]:
        """创建多样化的外部基因池"""
        gene_pools = {
            'plasmid': [],      # 质粒基因 - 高转移率
            'phage': [],        # 噬菌体基因 - 中等转移率
            'transposon': [],   # 转座子基因 - 中等转移率
            'chromosome': []    # 染色体基因 - 低转移率
        }
        
        # 不同来源的基因特征
        source_characteristics = {
            'plasmid': {
                'gc_range': (0.35, 0.65),
                'length_range': (300, 2000),
                'functions': ['antibiotic_resistance', 'metal_resistance', 'virulence_factors'],
                'transfer_freq': 0.8
            },
            'phage': {
                'gc_range': (0.30, 0.70),
                'length_range': (200, 1500),
                'functions': ['virulence_factors', 'stress_response', 'secondary_metabolism'],
                'transfer_freq': 0.6
            },
            'transposon': {
                'gc_range': (0.40, 0.60),
                'length_range': (500, 3000),
                'functions': ['antibiotic_resistance', 'secondary_metabolism'],
                'transfer_freq': 0.4
            },
            'chromosome': {
                'gc_range': (0.45, 0.55),
                'length_range': (600, 2500),
                'functions': ['central_metabolism', 'housekeeping', 'stress_response'],
                'transfer_freq': 0.2
            }
        }
        
        print(f"🧬 Creating diverse HGT gene pools...")
        
        for source_type, characteristics in source_characteristics.items():
            pool_size = self.gene_pool_size // 4  # 平均分配
            
            for i in range(pool_size):
                # 生成基因特征
                gc_content = random.uniform(*characteristics['gc_range'])
                length = random.randint(*characteristics['length_range'])
                function = random.choice(characteristics['functions'])
                
                # 生成序列
                sequence = self._generate_biased_sequence(length, gc_content)
                
                # 创建基因
                gene = Gene(
                    id=f"{source_type}_{function}_{i:04d}",
                    sequence=sequence,
                    start_pos=0,
                    length=length,
                    is_core=False,
                    origin="hgt"
                )
                
                # 添加来源信息
                gene.hgt_origin = GeneOrigin(
                    source_type=source_type,
                    gc_content=gc_content,
                    codon_usage_bias=random.uniform(0.1, 0.9),
                    metabolic_category=function,
                    transfer_frequency=characteristics['transfer_freq']
                )
                
                gene_pools[source_type].append(gene)
        
        print(f"✓ Created {sum(len(pool) for pool in gene_pools.values())} diverse HGT genes")
        return gene_pools
    
    def _generate_biased_sequence(self, length: int, gc_content: float) -> str:
        """生成具有特定GC含量的序列"""
        sequence = []
        for _ in range(length):
            if random.random() < gc_content:
                sequence.append(random.choice(['G', 'C']))
            else:
                sequence.append(random.choice(['A', 'T']))
        return ''.join(sequence)
    
    def calculate_genome_gc_content(self, genome: Genome) -> float:
        """计算基因组GC含量"""
        if genome.size == 0:
            return 0.5
        
        total_gc = 0
        total_bases = 0
        
        for gene in genome.genes:
            gc_count = gene.sequence.count('G') + gene.sequence.count('C')
            total_gc += gc_count
            total_bases += len(gene.sequence)
        
        return total_gc / total_bases if total_bases > 0 else 0.5
    
    def assess_transfer_barriers(self, donor_gene: Gene, recipient_genome: Genome) -> float:
        """评估转移屏障"""
        if not self.enable_transfer_barriers:
            return 1.0
        
        barrier_score = 1.0
        
        # 1. GC含量兼容性
        if hasattr(donor_gene, 'hgt_origin'):
            donor_gc = donor_gene.hgt_origin.gc_content
            recipient_gc = self.calculate_genome_gc_content(recipient_genome)
            
            gc_difference = abs(donor_gc - recipient_gc)
            if gc_difference > self.gc_content_tolerance:
                gc_penalty = 1.0 - (gc_difference - self.gc_content_tolerance) * 2
                barrier_score *= max(0.1, gc_penalty)
        
        # 2. 基因长度兼容性
        avg_gene_length = recipient_genome.size / recipient_genome.gene_count if recipient_genome.gene_count > 0 else 1000
        length_ratio = len(donor_gene.sequence) / avg_gene_length
        
        if length_ratio > 3.0 or length_ratio < 0.3:  # 过长或过短的基因
            barrier_score *= 0.5
        
        # 3. 序列复杂性检查
        donor_complexity = self._calculate_sequence_complexity(donor_gene.sequence)
        if donor_complexity < 0.3:  # 低复杂性序列 (重复序列)
            barrier_score *= 0.7
        
        return barrier_score
    
    def _calculate_sequence_complexity(self, sequence: str) -> float:
        """计算序列复杂性 (基于k-mer多样性)"""
        if len(sequence) < 6:
            return 1.0
        
        k = 3  # 使用三核苷酸
        kmers = set()
        
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i+k])
        
        max_possible_kmers = min(4**k, len(sequence) - k + 1)
        return len(kmers) / max_possible_kmers
    
    def assess_metabolic_integration(self, donor_gene: Gene, recipient_genome: Genome) -> float:
        """评估代谢网络整合可能性"""
        if not self.enable_metabolic_integration:
            return 1.0
        
        if not hasattr(donor_gene, 'hgt_origin'):
            return 0.5
        
        integration_score = 1.0
        donor_function = donor_gene.hgt_origin.metabolic_category
        
        # 基于功能类别的整合难度
        integration_difficulty = {
            'antibiotic_resistance': 0.9,    # 容易整合 - 独立功能
            'metal_resistance': 0.8,
            'stress_response': 0.7,
            'virulence_factors': 0.6,
            'secondary_metabolism': 0.4,     # 中等难度 - 可能需要辅助基因
            'central_metabolism': 0.2,       # 困难 - 可能干扰现有途径
            'housekeeping': 0.1              # 很困难 - 高度整合的功能
        }
        
        integration_score *= integration_difficulty.get(donor_function, 0.5)
        
        # 检查功能冗余
        existing_functions = set()
        for gene in recipient_genome.genes:
            if hasattr(gene, 'hgt_origin'):
                existing_functions.add(gene.hgt_origin.metabolic_category)
        
        if donor_function in existing_functions:
            integration_score *= 0.3  # 功能冗余降低整合可能性
        
        return integration_score
    
    def select_transfer_mechanism(self) -> str:
        """选择转移机制"""
        mechanisms = list(self.transfer_mechanisms.keys())
        weights = list(self.transfer_mechanisms.values())
        return np.random.choice(mechanisms, p=weights)
    
    def select_donor_gene_enhanced(self, recipient_genome: Genome) -> Optional[Gene]:
        """增强的供体基因选择"""
        # 选择基因池
        all_pools = []
        pool_weights = []
        
        for source_type, genes in self.external_gene_pools.items():
            if genes:
                all_pools.extend(genes)
                # 根据来源类型调整权重
                source_weight = genes[0].hgt_origin.transfer_frequency if hasattr(genes[0], 'hgt_origin') else 0.5
                pool_weights.extend([source_weight] * len(genes))
        
        if not all_pools:
            return None
        
        # 标准化权重
        total_weight = sum(pool_weights)
        if total_weight == 0:
            return random.choice(all_pools).copy()
        
        normalized_weights = [w / total_weight for w in pool_weights]
        
        # 选择基因
        selected_gene = np.random.choice(all_pools, p=normalized_weights).copy()
        
        # 应用转移屏障和整合评估
        barrier_score = self.assess_transfer_barriers(selected_gene, recipient_genome)
        integration_score = self.assess_metabolic_integration(selected_gene, recipient_genome)
        
        overall_success_prob = barrier_score * integration_score
        
        if random.random() < overall_success_prob:
            return selected_gene
        else:
            # 记录拒绝原因
            if barrier_score < 0.5:
                self.hgt_stats['rejected_by_barriers'] += 1
            elif integration_score < 0.5:
                self.hgt_stats['rejected_by_integration'] += 1
            return None
    
    def apply_enhanced_hgt(self, genome: Genome, generations: int = 1) -> int:
        """应用增强的HGT"""
        # 计算HGT事件数
        expected_events = self.hgt_rate * generations
        hgt_events = np.random.poisson(expected_events)
        
        successful_transfers = 0
        
        for _ in range(hgt_events):
            # 选择转移机制
            mechanism = self.select_transfer_mechanism()
            
            # 选择供体基因
            donor_gene = self.select_donor_gene_enhanced(genome)
            
            if donor_gene is None:
                continue
            
            # 尝试插入基因
            if self._insert_gene_with_integration(genome, donor_gene):
                successful_transfers += 1
                genome.total_hgt_events += 1
                
                # 更新统计
                self.hgt_stats['successful_transfers'] += 1
                self.hgt_stats['transfer_by_mechanism'][mechanism] += 1
                
                if hasattr(donor_gene, 'hgt_origin'):
                    function = donor_gene.hgt_origin.metabolic_category
                    self.hgt_stats['transfer_by_function'][function] += 1
        
        return successful_transfers
    
    def _insert_gene_with_integration(self, genome: Genome, donor_gene: Gene) -> bool:
        """带整合检查的基因插入"""
        try:
            # 选择插入位置 - 偏好基因组末端 (减少对现有基因的干扰)
            if genome.genes:
                # 70%概率插入末端，30%概率随机插入
                if random.random() < 0.7:
                    insert_position = len(genome.genes)
                else:
                    insert_position = random.randint(0, len(genome.genes))
                genome.genes.insert(insert_position, donor_gene)
            else:
                genome.genes.append(donor_gene)
            
            # 更新基因ID避免冲突
            donor_gene.id = f"hgt_{genome.generation}_{len(genome.genes):04d}"
            
            # 如果启用了代谢整合，可能需要调整基因表达
            if self.enable_metabolic_integration and hasattr(donor_gene, 'hgt_origin'):
                self._adjust_gene_expression(donor_gene, genome)
            
            return True
            
        except Exception as e:
            print(f"Enhanced HGT insertion failed: {e}")
            return False
    
    def _adjust_gene_expression(self, donor_gene: Gene, recipient_genome: Genome):
        """调整基因表达以适应新环境"""
        # 简化的表达调整 - 在实际实现中可能涉及启动子区域修改
        if hasattr(donor_gene, 'hgt_origin'):
            # 根据功能类别调整表达强度
            expression_adjustments = {
                'antibiotic_resistance': 1.2,  # 提高表达
                'stress_response': 1.1,
                'secondary_metabolism': 0.9,
                'central_metabolism': 0.8,     # 降低表达避免干扰
                'housekeeping': 0.7
            }
            
            function = donor_gene.hgt_origin.metabolic_category
            adjustment = expression_adjustments.get(function, 1.0)
            
            # 这里可以添加更复杂的表达调控逻辑
            # 例如修改启动子序列、添加调控元件等
    
    def get_enhanced_hgt_statistics(self, genome: Genome) -> Dict:
        """获取增强HGT统计信息"""
        hgt_genes = [gene for gene in genome.genes if gene.origin == "hgt"]
        
        # 基础统计
        base_stats = {
            'total_hgt_events': genome.total_hgt_events,
            'current_hgt_genes': len(hgt_genes),
            'hgt_gene_ratio': len(hgt_genes) / len(genome.genes) if genome.genes else 0,
            'avg_hgt_gene_length': np.mean([gene.length for gene in hgt_genes]) if hgt_genes else 0,
            'hgt_contribution_to_genome_size': sum(gene.length for gene in hgt_genes) / genome.size if genome.size > 0 else 0
        }
        
        # 增强统计
        enhanced_stats = {
            'transfer_barriers_enabled': self.enable_transfer_barriers,
            'metabolic_integration_enabled': self.enable_metabolic_integration,
            'hgt_detailed_stats': self.hgt_stats.copy()
        }
        
        # 功能分布分析
        if hgt_genes:
            function_distribution = {}
            source_distribution = {}
            gc_content_distribution = []
            
            for gene in hgt_genes:
                if hasattr(gene, 'hgt_origin'):
                    func = gene.hgt_origin.metabolic_category
                    source = gene.hgt_origin.source_type
                    
                    function_distribution[func] = function_distribution.get(func, 0) + 1
                    source_distribution[source] = source_distribution.get(source, 0) + 1
                    gc_content_distribution.append(gene.hgt_origin.gc_content)
            
            enhanced_stats.update({
                'function_distribution': function_distribution,
                'source_distribution': source_distribution,
                'avg_hgt_gc_content': np.mean(gc_content_distribution) if gc_content_distribution else 0,
                'hgt_gc_diversity': np.std(gc_content_distribution) if gc_content_distribution else 0
            })
        
        # 合并统计
        base_stats.update(enhanced_stats)
        return base_stats
    
    def print_hgt_analysis(self, genome: Genome):
        """打印HGT分析结果"""
        stats = self.get_enhanced_hgt_statistics(genome)
        
        print("\n" + "=" * 60)
        print("🧬 ENHANCED HGT ANALYSIS")
        print("=" * 60)
        
        print(f"📊 Transfer Statistics:")
        print(f"   Total HGT events: {stats['total_hgt_events']:,}")
        print(f"   Current HGT genes: {stats['current_hgt_genes']:,}")
        print(f"   HGT gene ratio: {stats['hgt_gene_ratio']:.3f}")
        print(f"   Successful transfers: {self.hgt_stats['successful_transfers']:,}")
        print(f"   Rejected by barriers: {self.hgt_stats['rejected_by_barriers']:,}")
        print(f"   Rejected by integration: {self.hgt_stats['rejected_by_integration']:,}")
        
        if 'function_distribution' in stats:
            print(f"\n🔬 Functional Distribution:")
            for func, count in stats['function_distribution'].items():
                percentage = count / stats['current_hgt_genes'] * 100
                print(f"   {func}: {count} ({percentage:.1f}%)")
        
        if 'source_distribution' in stats:
            print(f"\n📡 Source Distribution:")
            for source, count in stats['source_distribution'].items():
                percentage = count / stats['current_hgt_genes'] * 100
                print(f"   {source}: {count} ({percentage:.1f}%)")
        
        print("=" * 60)