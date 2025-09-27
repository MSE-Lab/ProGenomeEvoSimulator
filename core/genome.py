"""
Genome Data Structures and Utilities
基因组数据结构和工具函数

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import uuid

__version__ = "1.0.0"

@dataclass
class Gene:
    """基因类"""
    id: str
    sequence: str
    start_pos: int
    length: int
    is_core: bool = True  # 是否为核心基因（非HGT获得）
    origin: str = "ancestral"  # 基因来源：ancestral, hgt
    mutation_count: int = 0  # 累积突变数
    recombination_count: int = 0  # 重组事件数
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if self.length == 0:
            self.length = len(self.sequence)
    
    def mutate_position(self, position: int, new_base: str):
        """在指定位置进行点突变"""
        if 0 <= position < len(self.sequence):
            seq_list = list(self.sequence)
            seq_list[position] = new_base
            self.sequence = ''.join(seq_list)
            self.mutation_count += 1
    
    def apply_recombination(self, donor_sequence: str, start: int, end: int):
        """应用同源重组"""
        if 0 <= start < end <= len(self.sequence):
            seq_list = list(self.sequence)
            seq_list[start:end] = list(donor_sequence[start:end])
            self.sequence = ''.join(seq_list)
            self.recombination_count += 1
    
    def copy(self):
        """创建基因副本"""
        return Gene(
            id=self.id + "_copy",
            sequence=self.sequence,
            start_pos=self.start_pos,
            length=self.length,
            is_core=self.is_core,
            origin=self.origin,
            mutation_count=self.mutation_count,
            recombination_count=self.recombination_count
        )

class Genome:
    """基因组类"""
    
    def __init__(self, genes: List[Gene] = None):
        self.genes = genes or []
        self.generation = 0
        self.total_mutations = 0
        self.total_hgt_events = 0
        self.total_recombination_events = 0
        
    @property
    def size(self) -> int:
        """基因组大小（总碱基数）"""
        return sum(gene.length for gene in self.genes)
    
    @property
    def gene_count(self) -> int:
        """基因数量"""
        return len(self.genes)
    
    @property
    def core_gene_count(self) -> int:
        """核心基因数量"""
        return sum(1 for gene in self.genes if gene.is_core)
    
    @property
    def hgt_gene_count(self) -> int:
        """HGT获得基因数量"""
        return sum(1 for gene in self.genes if not gene.is_core)
    
    def get_gene_by_id(self, gene_id: str) -> Optional[Gene]:
        """根据ID获取基因"""
        for gene in self.genes:
            if gene.id == gene_id:
                return gene
        return None
    
    def add_gene(self, gene: Gene):
        """添加基因"""
        self.genes.append(gene)
        if gene.origin == "hgt":
            self.total_hgt_events += 1
    
    def remove_gene(self, gene_id: str):
        """移除基因"""
        self.genes = [gene for gene in self.genes if gene.id != gene_id]
    
    def get_statistics(self) -> Dict:
        """获取基因组统计信息"""
        return {
            'generation': self.generation,
            'total_size': self.size,
            'gene_count': self.gene_count,
            'core_genes': self.core_gene_count,
            'hgt_genes': self.hgt_gene_count,
            'total_mutations': self.total_mutations,
            'total_hgt_events': self.total_hgt_events,
            'total_recombination_events': self.total_recombination_events,
            'avg_gene_length': self.size / self.gene_count if self.gene_count > 0 else 0
        }
    
    def copy(self):
        """创建基因组副本"""
        new_genes = [gene.copy() for gene in self.genes]
        new_genome = Genome(new_genes)
        new_genome.generation = self.generation
        new_genome.total_mutations = self.total_mutations
        new_genome.total_hgt_events = self.total_hgt_events
        new_genome.total_recombination_events = self.total_recombination_events
        return new_genome

# 遗传密码表 - 标准密码子表
GENETIC_CODE = {
    # 起始密码子
    'ATG': 'M',  # 甲硫氨酸 - 起始密码子
    
    # 终止密码子
    'TAA': '*',  # 终止密码子 (琥珀)
    'TAG': '*',  # 终止密码子 (琥珀)
    'TGA': '*',  # 终止密码子 (蛋白石)
    
    # 其他密码子
    'TTT': 'F', 'TTC': 'F',  # 苯丙氨酸
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',  # 亮氨酸
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',  # 丝氨酸
    'TAT': 'Y', 'TAC': 'Y',  # 酪氨酸
    'TGT': 'C', 'TGC': 'C',  # 半胱氨酸
    'TGG': 'W',  # 色氨酸
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # 脯氨酸
    'CAT': 'H', 'CAC': 'H',  # 组氨酸
    'CAA': 'Q', 'CAG': 'Q',  # 谷氨酰胺
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',  # 精氨酸
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',  # 异亮氨酸
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # 苏氨酸
    'AAT': 'N', 'AAC': 'N',  # 天冬酰胺
    'AAA': 'K', 'AAG': 'K',  # 赖氨酸
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',  # 缬氨酸
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # 丙氨酸
    'GAT': 'D', 'GAC': 'D',  # 天冬氨酸
    'GAA': 'E', 'GAG': 'E',  # 谷氨酸
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',  # 甘氨酸
}

# 起始密码子
START_CODONS = ['ATG']

# 终止密码子
STOP_CODONS = ['TAA', 'TAG', 'TGA']

# 所有有效密码子（除了起始和终止密码子）
CODING_CODONS = [codon for codon in GENETIC_CODE.keys() 
                 if codon not in START_CODONS and codon not in STOP_CODONS]


def generate_random_codon(exclude_start_stop: bool = True) -> str:
    """
    生成随机密码子
    
    Args:
        exclude_start_stop: 是否排除起始和终止密码子
    
    Returns:
        随机密码子序列
    """
    if exclude_start_stop:
        return random.choice(CODING_CODONS)
    else:
        return random.choice(list(GENETIC_CODE.keys()))


def generate_biologically_correct_gene(target_length: int, min_length: int = 150) -> str:
    """
    生成生物学上正确的基因序列
    
    要求:
    1. 长度必须是3的倍数（密码子）
    2. 以起始密码子开始（ATG）
    3. 以终止密码子结束（TAA/TAG/TGA）
    4. 中间序列由有效密码子组成
    5. 确保最小功能长度（至少50个密码子 = 150bp）
    
    Args:
        target_length: 目标基因长度
        min_length: 最小基因长度
    
    Returns:
        生物学上正确的基因序列
    """
    # 确保最小长度符合生物学要求（至少50个密码子）
    absolute_min_length = 150  # 50个密码子的最小功能基因
    min_length = max(min_length, absolute_min_length)
    
    # 确保长度是3的倍数且不小于最小长度
    if target_length < min_length:
        target_length = min_length
    
    # 调整到最近的3的倍数
    target_length = ((target_length + 2) // 3) * 3
    
    # 确保至少有起始密码子 + 终止密码子 + 至少48个编码密码子 = 150bp
    if target_length < absolute_min_length:
        target_length = absolute_min_length
    
    # 计算需要的密码子数量
    total_codons = target_length // 3
    
    # 生物学检查：确保有足够的编码密码子
    if total_codons < 50:  # 少于50个密码子的基因在原核生物中极其罕见
        total_codons = 50
        target_length = 150
    
    # 构建基因序列
    sequence_parts = []
    
    # 1. 起始密码子（在原核生物中99%是ATG）
    sequence_parts.append('ATG')  # 使用最常见的起始密码子
    
    # 2. 编码区域（中间的密码子）
    coding_codons_needed = total_codons - 2  # 减去起始和终止密码子
    
    # 确保至少有48个编码密码子
    if coding_codons_needed < 48:
        coding_codons_needed = 48
        total_codons = 50
    
    for _ in range(coding_codons_needed):
        sequence_parts.append(generate_random_codon(exclude_start_stop=True))
    
    # 3. 终止密码子（使用生物学上合理的分布）
    # TAA (琥珀): ~60%, TAG (琥珀): ~20%, TGA (蛋白石): ~20%
    stop_codon_weights = {'TAA': 0.6, 'TAG': 0.2, 'TGA': 0.2}
    stop_codons = list(stop_codon_weights.keys())
    weights = list(stop_codon_weights.values())
    selected_stop = np.random.choice(stop_codons, p=weights)
    sequence_parts.append(selected_stop)
    
    # 组合成完整序列
    gene_sequence = ''.join(sequence_parts)
    
    # 最终验证
    if len(gene_sequence) % 3 != 0 or len(gene_sequence) < absolute_min_length:
        # 如果出现问题，递归重新生成
        return generate_biologically_correct_gene(target_length, min_length)
    
    return gene_sequence


def validate_gene_sequence(sequence: str) -> Dict[str, any]:
    """
    验证基因序列的生物学正确性
    
    Args:
        sequence: 基因序列
    
    Returns:
        验证结果字典
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'length': len(sequence),
        'codon_count': len(sequence) // 3,
        'has_start_codon': False,
        'has_stop_codon': False,
        'invalid_codons': []
    }
    
    # 检查长度是否是3的倍数
    if len(sequence) % 3 != 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"序列长度 {len(sequence)} 不是3的倍数")
    
    # 检查最小长度
    if len(sequence) < 9:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"序列长度 {len(sequence)} 小于最小基因长度 9bp")
    
    if len(sequence) >= 3:
        # 检查起始密码子
        start_codon = sequence[:3]
        if start_codon in START_CODONS:
            validation_result['has_start_codon'] = True
        else:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"起始密码子 '{start_codon}' 无效，应为 {START_CODONS}")
        
        # 检查终止密码子
        if len(sequence) >= 6:
            stop_codon = sequence[-3:]
            if stop_codon in STOP_CODONS:
                validation_result['has_stop_codon'] = True
            else:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"终止密码子 '{stop_codon}' 无效，应为 {STOP_CODONS}")
    
    # 检查所有密码子的有效性
    for i in range(0, len(sequence), 3):
        if i + 3 <= len(sequence):
            codon = sequence[i:i+3]
            if codon not in GENETIC_CODE:
                validation_result['is_valid'] = False
                validation_result['invalid_codons'].append((i//3, codon))
    
    return validation_result


def generate_random_sequence(length: int, gc_content: float = 0.5) -> str:
    """
    生成随机DNA序列（保留用于兼容性，但建议使用generate_biologically_correct_gene）
    """
    sequence = []
    for _ in range(length):
        if random.random() < gc_content:
            sequence.append(random.choice(['G', 'C']))
        else:
            sequence.append(random.choice(['A', 'T']))
    return ''.join(sequence)

def generate_realistic_gene_length(target_mean: int = 1000, min_length: int = 150) -> int:
    """
    Generate realistic prokaryotic gene length using gamma distribution
    确保长度是3的倍数（密码子要求）
    
    Prokaryotic genes typically:
    - Most genes around 1000bp
    - Few very short genes (150-300bp) 
    - Few very long genes (>3000bp)
    - Minimum functional gene length ~150bp (50 codons)
    - Length must be multiple of 3 (codon requirement)
    """
    # Use gamma distribution parameters that give realistic prokaryotic gene length distribution
    # Shape parameter (alpha): controls the skewness
    # Scale parameter (beta): controls the spread
    
    # Calculate gamma parameters to achieve target mean
    # For prokaryotic genes: shape=2.5 gives good right-skewed distribution
    shape = 2.5
    scale = target_mean / shape
    
    # Generate length from gamma distribution
    length = int(np.random.gamma(shape, scale))
    
    # Ensure minimum length and reasonable maximum
    length = max(min_length, min(length, 8000))  # Cap at 8kb for very long genes
    
    # 确保长度是3的倍数（密码子要求）
    length = ((length + 2) // 3) * 3
    
    # 确保最小长度（至少包含起始密码子 + 1个编码密码子 + 终止密码子 = 9bp）
    if length < 9:
        length = 9
    
    return length

def create_initial_genome(gene_count: int = 3000, 
                         avg_gene_length: int = 1000,
                         min_gene_length: int = 150,
                         use_biological_sequences: bool = True) -> Genome:
    """
    Create initial genome with realistic prokaryotic gene length distribution
    现在生成生物学上正确的基因序列
    
    Args:
        gene_count: Number of genes in the genome
        avg_gene_length: Target average gene length (bp, will be adjusted to multiple of 3)
        min_gene_length: Minimum gene length (bp, will be adjusted to multiple of 3)
        use_biological_sequences: Whether to generate biologically correct gene sequences
    """
    genes = []
    current_pos = 0
    
    # 确保参数是3的倍数
    avg_gene_length = ((avg_gene_length + 2) // 3) * 3
    min_gene_length = max(9, ((min_gene_length + 2) // 3) * 3)  # 最小9bp（3个密码子）
    
    print(f"🧬 Generating {gene_count:,} biologically correct genes...")
    print(f"📊 Parameters:")
    print(f"   Target average length: {avg_gene_length} bp (adjusted to codon multiple)")
    print(f"   Minimum length: {min_gene_length} bp (adjusted to codon multiple)")
    print(f"   Biological sequences: {'Enabled' if use_biological_sequences else 'Disabled'}")
    print(f"   Features: Start codons, stop codons, valid codon sequences")
    
    gene_lengths = []
    validation_stats = {
        'valid_genes': 0,
        'invalid_genes': 0,
        'total_codons': 0,
        'start_codon_distribution': {},
        'stop_codon_distribution': {}
    }
    
    for i in range(gene_count):
        # Use realistic gene length distribution (ensures multiple of 3)
        target_length = generate_realistic_gene_length(avg_gene_length, min_gene_length)
        gene_lengths.append(target_length)
        
        if use_biological_sequences:
            # Generate biologically correct gene sequence
            sequence = generate_biologically_correct_gene(target_length, min_gene_length)
            
            # Validate the generated sequence
            validation = validate_gene_sequence(sequence)
            if validation['is_valid']:
                validation_stats['valid_genes'] += 1
                validation_stats['total_codons'] += validation['codon_count']
                
                # Track start and stop codon usage
                start_codon = sequence[:3]
                stop_codon = sequence[-3:]
                
                validation_stats['start_codon_distribution'][start_codon] = \
                    validation_stats['start_codon_distribution'].get(start_codon, 0) + 1
                validation_stats['stop_codon_distribution'][stop_codon] = \
                    validation_stats['stop_codon_distribution'].get(stop_codon, 0) + 1
            else:
                validation_stats['invalid_genes'] += 1
                print(f"⚠️  Warning: Generated invalid gene {i}: {validation['errors']}")
        else:
            # Generate random sequence (for compatibility)
            sequence = generate_random_sequence(target_length)
        
        gene = Gene(
            id=f"gene_{i:04d}",
            sequence=sequence,
            start_pos=current_pos,
            length=len(sequence),  # Use actual sequence length
            is_core=True,
            origin="ancestral"
        )
        genes.append(gene)
        current_pos += len(sequence)
    
    # Print distribution statistics
    gene_lengths = np.array(gene_lengths)
    actual_lengths = np.array([len(gene.sequence) for gene in genes])
    
    print(f"\n✓ Generated genome statistics:")
    print(f"  📏 Size and Length:")
    print(f"     Total size: {current_pos:,} bp")
    print(f"     Target avg length: {gene_lengths.mean():.1f} bp")
    print(f"     Actual avg length: {actual_lengths.mean():.1f} bp")
    print(f"     Length range: {actual_lengths.min()}-{actual_lengths.max()} bp")
    
    print(f"  📊 Length Distribution:")
    print(f"     Genes <500bp: {np.sum(actual_lengths < 500):,} ({np.sum(actual_lengths < 500)/len(actual_lengths)*100:.1f}%)")
    print(f"     Genes 500-1500bp: {np.sum((actual_lengths >= 500) & (actual_lengths <= 1500)):,} ({np.sum((actual_lengths >= 500) & (actual_lengths <= 1500))/len(actual_lengths)*100:.1f}%)")
    print(f"     Genes >1500bp: {np.sum(actual_lengths > 1500):,} ({np.sum(actual_lengths > 1500)/len(actual_lengths)*100:.1f}%)")
    
    if use_biological_sequences:
        print(f"  🧬 Biological Validation:")
        print(f"     Valid genes: {validation_stats['valid_genes']:,}/{gene_count:,} ({validation_stats['valid_genes']/gene_count*100:.1f}%)")
        print(f"     Total codons: {validation_stats['total_codons']:,}")
        print(f"     Avg codons per gene: {validation_stats['total_codons']/validation_stats['valid_genes']:.1f}")
        
        print(f"  🎯 Codon Usage:")
        print(f"     Start codons: {dict(validation_stats['start_codon_distribution'])}")
        print(f"     Stop codons: {dict(validation_stats['stop_codon_distribution'])}")
        
        # Verify all genes have correct length (multiple of 3)
        non_codon_genes = [gene for gene in genes if len(gene.sequence) % 3 != 0]
        if non_codon_genes:
            print(f"  ⚠️  Warning: {len(non_codon_genes)} genes have non-codon lengths!")
        else:
            print(f"  ✅ All genes have codon-compatible lengths (multiples of 3)")
    
    return Genome(genes)