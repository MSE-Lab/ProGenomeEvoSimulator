#!/usr/bin/env python3
"""
Enhanced Evolution Engine with Gene Loss
集成基因丢失机制的增强进化引擎
"""

import time
import copy
from typing import Dict, List, Optional, Any
from core.evolution_engine_optimized import OptimizedEvolutionEngine
from core.genome import Genome
from mechanisms.gene_loss import GeneLossEngine


class EvolutionEngineWithGeneLoss(OptimizedEvolutionEngine):
    """集成基因丢失机制的进化引擎"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-6,
                 hgt_rate: float = 1e-8,
                 recombination_rate: float = 1e-9,
                 # 基因丢失参数
                 enable_gene_loss: bool = True,
                 loss_rate: float = 1e-6,
                 core_gene_protection: float = 0.95,
                 hgt_gene_loss_multiplier: float = 10.0,
                 min_genome_size: int = 1000,
                 min_core_genes: int = 800,
                 optimal_genome_size: int = 3000):
        """
        初始化集成基因丢失的进化引擎
        
        Args:
            enable_gene_loss: 是否启用基因丢失机制
            loss_rate: 基因丢失率
            core_gene_protection: 核心基因保护系数
            hgt_gene_loss_multiplier: HGT基因丢失倍数
            min_genome_size: 最小基因组大小
            min_core_genes: 最小核心基因数
            optimal_genome_size: 最优基因组大小
        """
        super().__init__(mutation_rate, hgt_rate, recombination_rate)
        
        self.enable_gene_loss = enable_gene_loss
        
        # 初始化基因丢失引擎
        if self.enable_gene_loss:
            self.gene_loss = GeneLossEngine(
                loss_rate=loss_rate,
                core_gene_protection=core_gene_protection,
                hgt_gene_loss_multiplier=hgt_gene_loss_multiplier,
                min_genome_size=min_genome_size,
                min_core_genes=min_core_genes,
                optimal_genome_size=optimal_genome_size
            )
        else:
            self.gene_loss = None
        
        print(f"🧬 Enhanced Evolution Engine with Gene Loss initialized:")
        print(f"   Gene loss: {'Enabled' if enable_gene_loss else 'Disabled'}")
        if enable_gene_loss:
            print(f"   Loss rate: {loss_rate}")
            print(f"   Core protection: {core_gene_protection*100:.1f}%")
            print(f"   Min genome size: {min_genome_size} genes")
    
    def evolve_one_generation(self, genome: Genome) -> Dict:
        """进化一代，包含基因丢失"""
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0,
            'genes_lost': 0
        }
        
        # 1. 应用传统进化机制（点突变、HGT、重组）
        mutations = self.point_mutation.apply_mutations(genome, generations=1)
        generation_stats['mutations'] = mutations
        
        hgt_events = self.hgt.apply_hgt(genome, generations=1)
        generation_stats['hgt_events'] = hgt_events
        
        recombination_events = self.recombination.apply_recombination(genome, generations=1)
        generation_stats['recombination_events'] = recombination_events
        
        # 2. 应用基因丢失机制
        if self.enable_gene_loss and self.gene_loss:
            genes_lost = self.gene_loss.apply_gene_loss(genome, generations=1)
            generation_stats['genes_lost'] = genes_lost
        
        # 更新代数
        genome.generation += 1
        
        # 记录最终统计
        generation_stats['final_stats'] = genome.get_statistics()
        
        return generation_stats
    
    def evolve_multiple_generations(self, genome: Genome, generations: int, show_progress: bool = True) -> List[Dict]:
        """进化多代，包含基因丢失"""
        history = []
        start_time = time.time()
        
        if show_progress:
            # 确定显示频率
            if generations <= 50:
                display_freq = 5
            elif generations <= 100:
                display_freq = 10
            elif generations <= 1000:
                display_freq = 50
            else:
                display_freq = 100
            
            print(f"Starting enhanced evolution simulation: {generations:,} generations")
            print(f"Gene loss: {'Enabled' if self.enable_gene_loss else 'Disabled'}")
            print(f"Progress updates every {display_freq} generation(s)")
            print("=" * 80)
        else:
            display_freq = max(1, generations // 10)
        
        for gen in range(generations):
            gen_start_time = time.time()
            gen_stats = self.evolve_one_generation(genome)
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            
            history.append(gen_stats)
            
            # 显示进度
            if show_progress and ((gen + 1) % display_freq == 0 or gen == 0):
                elapsed_total = time.time() - start_time
                avg_time_per_gen = elapsed_total / (gen + 1)
                remaining_gens = generations - (gen + 1)
                estimated_remaining = remaining_gens * avg_time_per_gen
                
                # 进度条
                progress = (gen + 1) / generations
                bar_width = 30
                filled_width = int(bar_width * progress)
                bar = '█' * filled_width + '░' * (bar_width - filled_width)
                
                # 基因组信息（包含基因丢失）
                genome_info = (f"Genes: {genome.gene_count:,} | "
                             f"Events: {genome.total_mutations:,}mut "
                             f"{genome.total_hgt_events:,}HGT "
                             f"{genome.total_recombination_events:,}rec")
                
                # 基因丢失信息
                if self.enable_gene_loss and self.gene_loss:
                    loss_stats = self.gene_loss.get_loss_statistics(genome)
                    loss_info = f" {loss_stats['total_genes_lost']:,}lost"
                    genome_info += loss_info
                
                print(f"\r[{bar}] {progress*100:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"{1/avg_time_per_gen:.1f} gen/s | ETA: {estimated_remaining/60:.1f}min | "
                      f"{genome_info}", end="", flush=True)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n\n🚀 Enhanced evolution completed!")
            print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
            print(f"Average speed: {generations/total_time:.2f} generations/second")
            
            # 显示基因丢失统计
            if self.enable_gene_loss and self.gene_loss:
                loss_stats = self.gene_loss.get_loss_statistics(genome)
                print(f"Gene loss stats: {loss_stats['total_genes_lost']} genes lost "
                      f"({loss_stats['avg_total_loss_per_generation']:.3f} per generation)")
            
            print("=" * 80)
        
        self.evolution_history.extend(history)
        return history
    
    def simulate_evolution(self, 
                          initial_genome: Genome, 
                          generations: int,
                          save_snapshots: bool = True,
                          snapshot_interval: int = 100) -> tuple:
        """完整的进化模拟，包含基因丢失"""
        
        print("🧬 ENHANCED PROKARYOTIC GENOME EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"📸 Snapshots: {'Enabled' if save_snapshots else 'Disabled'} (interval: {snapshot_interval})")
        print(f"🗑️  Gene loss: {'Enabled' if self.enable_gene_loss else 'Disabled'}")
        print(f"⚙️  Evolution mechanisms: Point mutations, HGT, Homologous recombination" + 
              (", Gene loss" if self.enable_gene_loss else ""))
        print("=" * 80)
        
        # 创建基因组副本
        evolving_genome = initial_genome.copy()
        simulation_start_time = time.time()
        
        # 记录初始状态
        snapshots = []
        if save_snapshots:
            initial_summary = self.get_evolution_summary(evolving_genome)
            initial_summary['snapshot_generation'] = 0
            snapshots.append(initial_summary)
        
        # 进化过程
        evolution_history = self.evolve_multiple_generations(evolving_genome, generations, show_progress=True)
        
        # 保存快照
        if save_snapshots:
            print(f"📸 Saving snapshots every {snapshot_interval} generations...")
            for i in range(0, len(evolution_history), snapshot_interval):
                if i < len(evolution_history):
                    snapshot = self.get_evolution_summary(evolving_genome)
                    snapshot['snapshot_generation'] = evolution_history[i]['generation']
                    snapshots.append(snapshot)
        
        # 最终总结
        total_simulation_time = time.time() - simulation_start_time
        final_summary = self.get_evolution_summary(evolving_genome)
        
        print(f"\n🎉 ENHANCED SIMULATION COMPLETED!")
        print(f"🧬 Final genome: {evolving_genome.gene_count:,} genes, {evolving_genome.size:,} bp")
        print(f"📈 Changes: {evolving_genome.size - initial_genome.size:+,} bp, "
              f"{evolving_genome.gene_count - initial_genome.gene_count:+,} genes")
        
        # 显示基因丢失摘要
        if self.enable_gene_loss and self.gene_loss:
            self.gene_loss.print_loss_summary(evolving_genome)
        
        return evolving_genome, snapshots
    
    def get_evolution_summary(self, genome: Genome) -> Dict:
        """获取进化总结，包含基因丢失统计"""
        summary = super().get_evolution_summary(genome)
        
        # 添加基因丢失统计
        if self.enable_gene_loss and self.gene_loss:
            loss_stats = self.gene_loss.get_loss_statistics(genome)
            summary['gene_loss_stats'] = loss_stats
            
            # 添加基因丢失分析
            loss_patterns = self.gene_loss.analyze_loss_patterns(genome)
            summary['gene_loss_patterns'] = loss_patterns
        
        return summary
    
    def print_evolution_summary(self, evolved_genome: Genome):
        """打印进化总结，包括基因丢失分析"""
        super().print_evolution_summary(evolved_genome)
        
        if self.enable_gene_loss and self.gene_loss:
            print("\n" + "=" * 60)
            print("🗑️  GENE LOSS ANALYSIS")
            print("=" * 60)
            
            # 基因丢失统计
            loss_stats = self.gene_loss.get_loss_statistics(evolved_genome)
            print(f"📊 Loss Statistics:")
            print(f"   Total genes lost: {loss_stats['total_genes_lost']}")
            print(f"   Core genes lost: {loss_stats['core_genes_lost']}")
            print(f"   HGT genes lost: {loss_stats['hgt_genes_lost']}")
            print(f"   Average loss per generation: {loss_stats['avg_total_loss_per_generation']:.3f}")
            
            # 基因丢失模式分析
            loss_patterns = self.gene_loss.analyze_loss_patterns(evolved_genome)
            if 'error' not in loss_patterns:
                print(f"\n📈 Loss Patterns:")
                print(f"   HGT loss enrichment: {loss_patterns['hgt_loss_enrichment']:.2f}x")
                print(f"   Genome size trend: {loss_patterns['genome_size_trend']:+.3f} genes/gen")
                print(f"   Protection effectiveness: {loss_patterns['protection_effectiveness']:.3f}")
            
            print("=" * 60)
    
    def get_comprehensive_statistics(self, genome: Genome) -> Dict:
        """获取包含基因丢失的综合统计信息"""
        stats = {
            'genome_stats': genome.get_statistics(),
            'mutation_stats': self.point_mutation.get_mutation_statistics(genome),
            'hgt_stats': self.hgt.get_hgt_statistics(genome),
            'recombination_stats': self.recombination.get_recombination_statistics(genome)
        }
        
        # 添加基因丢失统计
        if self.enable_gene_loss and self.gene_loss:
            stats['gene_loss_stats'] = self.gene_loss.get_loss_statistics(genome)
            stats['gene_loss_patterns'] = self.gene_loss.analyze_loss_patterns(genome)
        
        # 计算综合进化效率
        stats['evolution_efficiency'] = {
            'mutations_per_generation': genome.total_mutations / max(1, genome.generation),
            'hgt_per_generation': genome.total_hgt_events / max(1, genome.generation),
            'recombination_per_generation': genome.total_recombination_events / max(1, genome.generation)
        }
        
        if self.enable_gene_loss and self.gene_loss:
            loss_stats = self.gene_loss.get_loss_statistics(genome)
            stats['evolution_efficiency']['loss_per_generation'] = loss_stats['avg_total_loss_per_generation']
        
        return stats