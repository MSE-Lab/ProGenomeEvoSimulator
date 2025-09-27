import random
import numpy as np
from typing import List
from core.genome import Genome, Gene, generate_random_sequence

class HorizontalGeneTransfer:
    """横向基因转移引擎"""
    
    def __init__(self, hgt_rate: float = 0.001, gene_pool_size: int = 10000):
        self.hgt_rate = hgt_rate  # 每代每基因组的HGT概率
        self.gene_pool_size = gene_pool_size
        self.external_gene_pool = self._create_gene_pool()
    
    def _create_gene_pool(self) -> List[Gene]:
        """创建外部基因池"""
        gene_pool = []
        
        for i in range(self.gene_pool_size):
            # 基因长度变化范围更大
            length = np.random.choice([
                np.random.normal(800, 150),   # 短基因
                np.random.normal(1200, 200),  # 中等基因
                np.random.normal(2000, 300)   # 长基因
            ])
            length = max(200, int(length))
            
            sequence = generate_random_sequence(length)
            
            gene = Gene(
                id=f"hgt_pool_{i:05d}",
                sequence=sequence,
                start_pos=0,
                length=length,
                is_core=False,
                origin="hgt"
            )
            gene_pool.append(gene)
        
        return gene_pool
    
    def calculate_hgt_events(self, generations: int = 1) -> int:
        """计算HGT事件数量"""
        # 使用泊松分布模拟HGT事件
        expected_events = self.hgt_rate * generations
        return np.random.poisson(expected_events)
    
    def select_donor_gene(self) -> Gene:
        """从基因池中随机选择供体基因"""
        donor = random.choice(self.external_gene_pool)
        # 创建副本避免修改原始基因池
        return donor.copy()
    
    def insert_gene(self, genome: Genome, donor_gene: Gene) -> bool:
        """将供体基因插入到基因组中"""
        try:
            # 随机选择插入位置
            if genome.genes:
                # 修复索引越界问题：插入位置应该在0到len(genome.genes)之间（包括len）
                # insert()方法允许在末尾插入，所以这里是正确的
                insert_position = random.randint(0, len(genome.genes))
                genome.genes.insert(insert_position, donor_gene)
            else:
                genome.genes.append(donor_gene)
            
            # 更新基因ID以避免冲突
            donor_gene.id = f"hgt_{genome.generation}_{len(genome.genes):04d}"
            
            return True
        except Exception as e:
            print(f"HGT insertion failed: {e}")
            return False
    
    def apply_hgt(self, genome: Genome, generations: int = 1) -> int:
        """对基因组应用横向基因转移"""
        hgt_events = self.calculate_hgt_events(generations)
        successful_transfers = 0
        
        for _ in range(hgt_events):
            donor_gene = self.select_donor_gene()
            if self.insert_gene(genome, donor_gene):
                successful_transfers += 1
                genome.total_hgt_events += 1
        
        return successful_transfers
    
    def get_hgt_statistics(self, genome: Genome) -> dict:
        """获取HGT统计信息"""
        hgt_genes = [gene for gene in genome.genes if gene.origin == "hgt"]
        
        return {
            'total_hgt_events': genome.total_hgt_events,
            'current_hgt_genes': len(hgt_genes),
            'hgt_gene_ratio': len(hgt_genes) / len(genome.genes) if genome.genes else 0,
            'avg_hgt_gene_length': np.mean([gene.length for gene in hgt_genes]) if hgt_genes else 0,
            'hgt_contribution_to_genome_size': sum(gene.length for gene in hgt_genes) / genome.size if genome.size > 0 else 0
        }