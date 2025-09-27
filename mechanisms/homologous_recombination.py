import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from core.genome import Genome, Gene

class HomologousRecombination:
    """
    同源重组引擎 - 重新设计版本
    
    新机制：模拟外源同源基因替换效应
    - 随机选择基因组中的基因
    - 一次性引入多个点突变
    - 模拟外源同源基因的序列差异效果
    """
    
    def __init__(self, 
                 recombination_rate: float = 1e-4,
                 mutations_per_event: Tuple[int, int] = (5, 15),
                 enable_debug: bool = False):
        
        self.recombination_rate = recombination_rate  # 每基因每代的重组概率
        self.mutations_per_event = mutations_per_event  # 每次重组事件的突变数量范围 (min, max)
        self.enable_debug = enable_debug
        
        # 统计信息
        self.total_recombination_events = 0
        self.total_mutations_from_recombination = 0
        self.genes_affected_by_recombination = set()
        
        if self.enable_debug:
            print(f"🔧 HomologousRecombination initialized (NEW DESIGN):")
            print(f"   Recombination rate: {recombination_rate} (per gene per generation)")
            print(f"   Mutations per event: {mutations_per_event[0]}-{mutations_per_event[1]}")
            print(f"   Debug mode: Enabled")
    
    def calculate_recombination_events(self, genome: Genome, generations: int = 1) -> int:
        """计算本代需要发生的重组事件数量"""
        # 基于基因数量和重组率计算期望事件数
        expected_events = len(genome.genes) * self.recombination_rate * generations
        
        if self.enable_debug:
            print(f"🎲 Calculating recombination events:")
            print(f"   Genes: {len(genome.genes)}, Rate: {self.recombination_rate}, Generations: {generations}")
            print(f"   Expected events: {expected_events:.3f}")
        
        # 使用泊松分布生成实际事件数
        actual_events = np.random.poisson(expected_events)
        
        if self.enable_debug:
            print(f"   Actual events to perform: {actual_events}")
        
        return actual_events
    
    def select_mutation_positions(self, gene: Gene, num_mutations: int) -> List[int]:
        """在基因中选择突变位点"""
        sequence_length = len(gene.sequence)
        
        if num_mutations >= sequence_length:
            # 如果突变数量超过序列长度，选择所有位点
            return list(range(sequence_length))
        
        # 随机选择不重复的位点
        positions = random.sample(range(sequence_length), num_mutations)
        return sorted(positions)
    
    def generate_point_mutation(self, original_base: str) -> str:
        """生成点突变，返回新的碱基"""
        bases = ['A', 'T', 'G', 'C']
        # 移除原始碱基，从剩余的碱基中随机选择
        available_bases = [base for base in bases if base != original_base]
        return random.choice(available_bases)
    
    def perform_homologous_recombination(self, gene: Gene) -> int:
        """
        对单个基因执行同源重组
        返回引入的突变数量
        """
        # 确定本次重组事件的突变数量
        min_mutations, max_mutations = self.mutations_per_event
        num_mutations = random.randint(min_mutations, max_mutations)
        
        # 选择突变位点
        mutation_positions = self.select_mutation_positions(gene, num_mutations)
        
        if not mutation_positions:
            return 0
        
        # 执行突变
        sequence_list = list(gene.sequence)
        mutations_applied = 0
        
        for pos in mutation_positions:
            if pos < len(sequence_list):
                original_base = sequence_list[pos]
                new_base = self.generate_point_mutation(original_base)
                sequence_list[pos] = new_base
                mutations_applied += 1
        
        # 更新基因序列
        gene.sequence = ''.join(sequence_list)
        
        # 更新基因的重组计数
        gene.recombination_count += 1
        
        if self.enable_debug:
            print(f"   🧬 Gene {gene.id}: Applied {mutations_applied} mutations at positions {mutation_positions[:5]}{'...' if len(mutation_positions) > 5 else ''}")
        
        return mutations_applied
    
    def apply_recombination(self, genome: Genome, generations: int = 1) -> int:
        """对基因组应用同源重组"""
        if self.enable_debug:
            print(f"\n🧬 Applying homologous recombination to genome (generation {genome.generation})...")
        
        if not genome.genes:
            if self.enable_debug:
                print("⚠️  No genes in genome to recombine")
            return 0
        
        # 计算重组事件数量
        recombination_events = self.calculate_recombination_events(genome, generations)
        
        if recombination_events == 0:
            if self.enable_debug:
                print("🎲 No recombination events to perform this generation")
            return 0
        
        successful_recombinations = 0
        total_mutations_this_generation = 0
        
        if self.enable_debug:
            print(f"🎯 Performing {recombination_events} recombination events on {len(genome.genes)} genes...")
        
        for i in range(recombination_events):
            # 随机选择一个基因进行重组
            target_gene = random.choice(genome.genes)
            
            if self.enable_debug and i < 3:  # 只显示前3次的详细信息
                print(f"   Event {i+1}: Targeting Gene {target_gene.id} (length: {len(target_gene.sequence)}bp)")
            
            # 执行重组（多点突变）
            mutations_applied = self.perform_homologous_recombination(target_gene)
            
            if mutations_applied > 0:
                successful_recombinations += 1
                total_mutations_this_generation += mutations_applied
                
                # 更新统计信息
                genome.total_recombination_events += 1
                self.total_recombination_events += 1
                self.total_mutations_from_recombination += mutations_applied
                self.genes_affected_by_recombination.add(target_gene.id)
        
        if self.enable_debug:
            print(f"📊 Recombination summary:")
            print(f"   Events performed: {successful_recombinations}/{recombination_events}")
            print(f"   Total mutations introduced: {total_mutations_this_generation}")
            print(f"   Average mutations per event: {total_mutations_this_generation/successful_recombinations:.1f}" if successful_recombinations > 0 else "   Average mutations per event: 0")
        
        return successful_recombinations
    
    def get_recombination_statistics(self, genome: Genome) -> Dict[str, float]:
        """获取重组统计信息"""
        recombination_counts = [gene.recombination_count for gene in genome.genes]
        
        stats = {
            'total_recombination_events': float(genome.total_recombination_events),
            'total_mutations_from_recombination': float(self.total_mutations_from_recombination),
            'genes_affected_by_recombination': float(len(self.genes_affected_by_recombination)),
            'genes_with_recombination': float(sum(1 for count in recombination_counts if count > 0)),
            'avg_recombination_per_gene': float(np.mean(recombination_counts)) if recombination_counts else 0.0,
            'max_recombination_per_gene': float(max(recombination_counts)) if recombination_counts else 0.0,
            'avg_mutations_per_recombination': float(self.total_mutations_from_recombination / self.total_recombination_events) if self.total_recombination_events > 0 else 0.0,
            'recombination_rate': float(self.recombination_rate),
            'mutations_per_event_range': f"{self.mutations_per_event[0]}-{self.mutations_per_event[1]}"
        }
        
        return stats
    
    def reset_statistics(self):
        """重置统计计数器"""
        self.total_recombination_events = 0
        self.total_mutations_from_recombination = 0
        self.genes_affected_by_recombination.clear()