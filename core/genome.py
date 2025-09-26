import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import uuid

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

def generate_random_sequence(length: int, gc_content: float = 0.5) -> str:
    """生成随机DNA序列"""
    sequence = []
    for _ in range(length):
        if random.random() < gc_content:
            sequence.append(random.choice(['G', 'C']))
        else:
            sequence.append(random.choice(['A', 'T']))
    return ''.join(sequence)

def generate_realistic_gene_length(target_mean: int = 1000, min_length: int = 100) -> int:
    """
    Generate realistic prokaryotic gene length using gamma distribution
    
    Prokaryotic genes typically:
    - Most genes around 1000bp
    - Few very short genes (100-300bp)
    - Few very long genes (>3000bp)
    - Minimum functional gene length ~100bp
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
    
    return length

def create_initial_genome(gene_count: int = 3000, 
                         avg_gene_length: int = 1000,
                         min_gene_length: int = 100) -> Genome:
    """
    Create initial genome with realistic prokaryotic gene length distribution
    
    Args:
        gene_count: Number of genes in the genome
        avg_gene_length: Target average gene length (bp)
        min_gene_length: Minimum gene length (bp)
    """
    genes = []
    current_pos = 0
    
    print(f"Generating {gene_count:,} genes with realistic length distribution...")
    print(f"Target average length: {avg_gene_length} bp, minimum: {min_gene_length} bp")
    
    gene_lengths = []
    for i in range(gene_count):
        # Use realistic gene length distribution
        length = generate_realistic_gene_length(avg_gene_length, min_gene_length)
        gene_lengths.append(length)
        
        sequence = generate_random_sequence(length)
        
        gene = Gene(
            id=f"gene_{i:04d}",
            sequence=sequence,
            start_pos=current_pos,
            length=length,
            is_core=True,
            origin="ancestral"
        )
        genes.append(gene)
        current_pos += length
    
    # Print distribution statistics
    gene_lengths = np.array(gene_lengths)
    print(f"✓ Generated genome statistics:")
    print(f"  - Total size: {current_pos:,} bp")
    print(f"  - Actual average gene length: {gene_lengths.mean():.1f} bp")
    print(f"  - Gene length range: {gene_lengths.min()}-{gene_lengths.max()} bp")
    print(f"  - Genes <500bp: {np.sum(gene_lengths < 500):,} ({np.sum(gene_lengths < 500)/len(gene_lengths)*100:.1f}%)")
    print(f"  - Genes 500-1500bp: {np.sum((gene_lengths >= 500) & (gene_lengths <= 1500)):,} ({np.sum((gene_lengths >= 500) & (gene_lengths <= 1500))/len(gene_lengths)*100:.1f}%)")
    print(f"  - Genes >1500bp: {np.sum(gene_lengths > 1500):,} ({np.sum(gene_lengths > 1500)/len(gene_lengths)*100:.1f}%)")
    
    return Genome(genes)