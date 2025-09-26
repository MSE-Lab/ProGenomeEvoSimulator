#!/usr/bin/env python3
"""
原核生物基因组进化模拟器
主程序入口
"""

import numpy as np
import matplotlib.pyplot as plt
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine
from analysis.ani_calculator import ANICalculator

def run_evolution_simulation(generations: int = 1000,
                           initial_gene_count: int = 3000,
                           mutation_rate: float = 1e-8,
                           hgt_rate: float = 0.001,
                           recombination_rate: float = 1e-6):
    """运行进化模拟"""
    
    print("=" * 60)
    print("Prokaryotic Genome Evolution Simulator")
    print("=" * 60)
    
    # 1. Create initial genome with realistic gene length distribution
    print("1. Creating initial genome with realistic gene length distribution...")
    initial_genome = create_initial_genome(
        gene_count=initial_gene_count,
        avg_gene_length=1000,
        min_gene_length=100
    )
    
    print(f"   ✓ Genome created with realistic prokaryotic gene length distribution")
    print()
    
    # 2. Initialize evolution engine
    print("2. Initializing evolution engine...")
    evolution_engine = EvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate
    )
    print(f"   Point mutation rate: {mutation_rate}")
    print(f"   Horizontal gene transfer rate: {hgt_rate}")
    print(f"   Homologous recombination rate: {recombination_rate}")
    print()
    
    # 3. Run evolution simulation
    print("3. Starting evolution simulation...")
    evolved_genome, snapshots = evolution_engine.simulate_evolution(
        initial_genome=initial_genome,
        generations=generations,
        save_snapshots=True,
        snapshot_interval=100
    )
    print()
    
    # 4. Calculate ANI and ortholog analysis
    print("4. Analyzing evolution results...")
    ani_calculator = ANICalculator()
    
    comprehensive_analysis = ani_calculator.compare_genomes_comprehensive(
        ancestral_genome=initial_genome,
        evolved_genome=evolved_genome
    )
    
    # 5. Output results
    print_analysis_results(comprehensive_analysis, generations)
    
    # 6. Visualize results
    visualize_results(comprehensive_analysis, snapshots, generations)
    
    return initial_genome, evolved_genome, comprehensive_analysis, snapshots

def print_analysis_results(analysis: dict, generations: int):
    """Print analysis results"""
    
    print("=" * 60)
    print("Evolution Analysis Results")
    print("=" * 60)
    
    # ANI results
    ani_data = analysis['ani_analysis']
    print(f"Average Nucleotide Identity (ANI): {ani_data['ani']:.4f}")
    print(f"Weighted ANI: {ani_data['weighted_ani']:.4f}")
    print(f"Orthologous gene pairs: {ani_data['ortholog_count']}")
    print(f"Ortholog ratio: {ani_data['ortholog_ratio']:.4f}")
    print()
    
    # Identity distribution
    identity_dist = analysis['identity_distribution']
    print("Orthologous Gene Identity Distribution:")
    print(f"  Mean: {identity_dist['mean']:.4f}")
    print(f"  Standard deviation: {identity_dist['std']:.4f}")
    print(f"  Median: {identity_dist['median']:.4f}")
    print(f"  Range: {identity_dist['min']:.4f} - {identity_dist['max']:.4f}")
    print()
    
    # Genome composition changes
    composition = analysis['genome_composition']
    print("Genome Composition Changes:")
    print(f"  Ancestral core genes: {composition['ancestral_core_genes']}")
    print(f"  Evolved core genes: {composition['evolved_core_genes']}")
    print(f"  HGT acquired genes: {composition['evolved_hgt_genes']}")
    print(f"  Core gene retention rate: {composition['core_gene_retention']:.4f}")
    print()
    
    # Genome size changes
    size_changes = analysis['size_changes']
    print("Genome Size Changes:")
    print(f"  Initial size: {size_changes['ancestral_size']:,} bp")
    print(f"  Final size: {size_changes['evolved_size']:,} bp")
    print(f"  Size change: {size_changes['size_change']:+,} bp ({size_changes['size_change_ratio']:+.2%})")
    print()
    
    # Gene count changes
    gene_changes = analysis['gene_count_changes']
    print("Gene Count Changes:")
    print(f"  Initial gene count: {gene_changes['ancestral_gene_count']}")
    print(f"  Final gene count: {gene_changes['evolved_gene_count']}")
    print(f"  Gene count change: {gene_changes['gene_count_change']:+d} ({gene_changes['gene_count_change_ratio']:+.2%})")
    print()

def visualize_results(analysis: dict, snapshots: list, generations: int):
    """Visualize results"""
    
    try:
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Genome Evolution Simulation Results ({generations} generations)', fontsize=16)
        
        # 1. Orthologous gene identity distribution histogram
        identity_dist = analysis['identity_distribution']
        if identity_dist['counts']:
            axes[0, 0].hist(analysis['ani_analysis']['identity_distribution'], 
                           bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(identity_dist['mean'], color='red', linestyle='--', 
                              label=f'Mean: {identity_dist["mean"]:.3f}')
            axes[0, 0].set_xlabel('Sequence Identity')
            axes[0, 0].set_ylabel('Number of Gene Pairs')
            axes[0, 0].set_title('Orthologous Gene Identity Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Genome size changes
        if len(snapshots) > 1:
            generations_list = [s.get('snapshot_generation', 0) for s in snapshots]
            sizes = [s['genome_stats']['total_size'] for s in snapshots]
            axes[0, 1].plot(generations_list, sizes, 'b-o', markersize=4)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Genome Size (bp)')
            axes[0, 1].set_title('Genome Size Changes')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gene count changes
        if len(snapshots) > 1:
            gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
            core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
            hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
            
            axes[1, 0].plot(generations_list, gene_counts, 'g-o', label='Total genes', markersize=4)
            axes[1, 0].plot(generations_list, core_genes, 'b-s', label='Core genes', markersize=4)
            axes[1, 0].plot(generations_list, hgt_genes, 'r-^', label='HGT genes', markersize=4)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Gene Count')
            axes[1, 0].set_title('Gene Count Changes')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Evolution event accumulation
        if len(snapshots) > 1:
            mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
            hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
            recombinations = [s['genome_stats']['total_recombination_events'] for s in snapshots]
            
            axes[1, 1].plot(generations_list, mutations, 'purple', label='Point mutations', linewidth=2)
            axes[1, 1].plot(generations_list, hgt_events, 'orange', label='HGT events', linewidth=2)
            axes[1, 1].plot(generations_list, recombinations, 'green', label='Recombination events', linewidth=2)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Cumulative Events')
            axes[1, 1].set_title('Evolution Event Accumulation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolution_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization results saved as 'evolution_simulation_results.png'")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        print("Skipping visualization step...")

def main():
    """Main function"""
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run simulation
    initial_genome, evolved_genome, analysis, snapshots = run_evolution_simulation(
        generations=1000,           # Evolution generations
        initial_gene_count=3000,    # Initial gene count
        mutation_rate=1e-8,         # Point mutation rate (slightly increased for observable effects)
        hgt_rate=0.002,            # HGT rate (slightly increased for observable effects)
        recombination_rate=1e-5     # Recombination rate (slightly increased for observable effects)
    )
    
    print("\nSimulation completed!")
    print("You can adjust parameters and re-run the simulation, or analyze the result data.")

if __name__ == "__main__":
    main()