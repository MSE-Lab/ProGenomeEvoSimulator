import numpy as np
import time
from typing import Dict, List, Tuple
from core.genome import Genome
from mechanisms.point_mutation import PointMutationEngine
from mechanisms.horizontal_transfer import HorizontalGeneTransfer
from mechanisms.homologous_recombination import HomologousRecombination

class EvolutionEngine:
    """进化引擎 - 协调所有进化机制"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-9,
                 hgt_rate: float = 0.001,
                 recombination_rate: float = 1e-6,
                 min_similarity_for_recombination: float = 0.7):
        
        # 初始化各种进化机制
        self.point_mutation = PointMutationEngine(mutation_rate)
        self.hgt = HorizontalGeneTransfer(hgt_rate)
        self.recombination = HomologousRecombination(
            recombination_rate, 
            min_similarity_for_recombination
        )
        
        # 进化历史记录
        self.evolution_history = []
        
    def evolve_one_generation(self, genome: Genome) -> Dict:
        """进化一代"""
        generation_stats = {
            'generation': genome.generation + 1,
            'initial_stats': genome.get_statistics(),
            'mutations': 0,
            'hgt_events': 0,
            'recombination_events': 0
        }
        
        # 1. 应用点突变
        mutations = self.point_mutation.apply_mutations(genome, generations=1)
        generation_stats['mutations'] = mutations
        
        # 2. 应用横向基因转移
        hgt_events = self.hgt.apply_hgt(genome, generations=1)
        generation_stats['hgt_events'] = hgt_events
        
        # 3. 应用同源重组
        recombination_events = self.recombination.apply_recombination(genome, generations=1)
        generation_stats['recombination_events'] = recombination_events
        
        # 更新代数
        genome.generation += 1
        
        # 记录最终统计
        generation_stats['final_stats'] = genome.get_statistics()
        
        return generation_stats
    
    def evolve_multiple_generations(self, genome: Genome, generations: int, show_progress: bool = True) -> List[Dict]:
        """Evolve multiple generations with detailed progress tracking"""
        history = []
        start_time = time.time()
        
        if show_progress:
            # Determine display frequency based on total generations
            if generations <= 50:
                display_freq = 5   # Every 5 generations for small runs
            elif generations <= 100:
                display_freq = 10  # Every 10 generations
            elif generations <= 1000:
                display_freq = 50  # Every 50 generations
            else:
                display_freq = 100  # Every 100 generations
            
            print(f"Starting evolution simulation: {generations:,} generations")
            print(f"Progress updates every {display_freq} generation(s)")
            print("=" * 70)
        else:
            display_freq = max(1, generations // 10)  # Still calculate for internal use
        
        for gen in range(generations):
            gen_start_time = time.time()
            gen_stats = self.evolve_one_generation(genome)
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            
            history.append(gen_stats)
            
            # Display progress at specified intervals
            if show_progress and ((gen + 1) % display_freq == 0 or gen == 0):
                elapsed_total = time.time() - start_time
                avg_time_per_gen = elapsed_total / (gen + 1)
                remaining_gens = generations - (gen + 1)
                estimated_remaining = remaining_gens * avg_time_per_gen
                
                # Progress bar
                progress = (gen + 1) / generations
                bar_width = 30
                filled_width = int(bar_width * progress)
                bar = '█' * filled_width + '░' * (bar_width - filled_width)
                
                # Include genome stats in the same line to avoid line breaks
                genome_info = f"Genes: {genome.gene_count:,} | Events: {genome.total_mutations:,}mut {genome.total_hgt_events:,}HGT {genome.total_recombination_events:,}rec"
                
                # Update progress bar on same line with all info
                print(f"\r[{bar}] {progress*100:.1f}% | Gen {gen + 1:,}/{generations:,} | "
                      f"{1/avg_time_per_gen:.1f} gen/s | "
                      f"ETA: {estimated_remaining/60:.1f}min | {genome_info}", end="", flush=True)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n\nEvolution completed!")  # Extra newline to ensure clean separation
            print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
            print(f"Average speed: {generations/total_time:.2f} generations/second")
            print("=" * 80)
        
        self.evolution_history.extend(history)
        return history
    
    def get_evolution_summary(self, genome: Genome) -> Dict:
        """获取进化总结"""
        mutation_stats = self.point_mutation.get_mutation_statistics(genome)
        hgt_stats = self.hgt.get_hgt_statistics(genome)
        recombination_stats = self.recombination.get_recombination_statistics(genome)
        
        return {
            'genome_stats': genome.get_statistics(),
            'mutation_stats': mutation_stats,
            'hgt_stats': hgt_stats,
            'recombination_stats': recombination_stats,
            'evolution_efficiency': {
                'mutations_per_generation': genome.total_mutations / max(1, genome.generation),
                'hgt_per_generation': genome.total_hgt_events / max(1, genome.generation),
                'recombination_per_generation': genome.total_recombination_events / max(1, genome.generation)
            }
        }
    
    def simulate_evolution(self, 
                          initial_genome: Genome, 
                          generations: int,
                          save_snapshots: bool = True,
                          snapshot_interval: int = 100) -> Tuple[Genome, List[Dict]]:
        """Complete evolution simulation with enhanced progress tracking"""
        
        print("🧬 PROKARYOTIC GENOME EVOLUTION SIMULATION")
        print("=" * 80)
        print(f"📊 Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
        print(f"🎯 Target generations: {generations:,}")
        print(f"📸 Snapshots: {'Enabled' if save_snapshots else 'Disabled'} (interval: {snapshot_interval})")
        print(f"⚙️  Evolution mechanisms: Point mutations, HGT, Homologous recombination")
        print("=" * 80)
        
        # Create genome copy to avoid modifying original
        evolving_genome = initial_genome.copy()
        simulation_start_time = time.time()
        
        # Record initial state
        snapshots = []
        if save_snapshots:
            initial_summary = self.get_evolution_summary(evolving_genome)
            initial_summary['snapshot_generation'] = 0
            snapshots.append(initial_summary)
        
        # Evolution process with enhanced tracking
        evolution_history = self.evolve_multiple_generations(evolving_genome, generations, show_progress=True)
        
        # Save snapshots during evolution
        if save_snapshots:
            print(f"📸 Saving snapshots every {snapshot_interval} generations...")
            for i in range(0, len(evolution_history), snapshot_interval):
                if i < len(evolution_history):
                    snapshot = self.get_evolution_summary(evolving_genome)
                    snapshot['snapshot_generation'] = evolution_history[i]['generation']
                    snapshots.append(snapshot)
        
        # Final summary (brief, since evolve_multiple_generations already showed completion)
        total_simulation_time = time.time() - simulation_start_time
        final_summary = self.get_evolution_summary(evolving_genome)
        
        print(f"\n🎉 SIMULATION COMPLETED!")
        print(f"🧬 Final genome: {evolving_genome.gene_count:,} genes, {evolving_genome.size:,} bp")
        print(f"📈 Changes: {evolving_genome.size - initial_genome.size:+,} bp, {evolving_genome.gene_count - initial_genome.gene_count:+,} genes")
        
        return evolving_genome, snapshots