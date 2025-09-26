#!/usr/bin/env python3
"""
Optimized Prokaryotic Genome Evolution Simulator
优化版原核基因组进化模拟器

This version uses optimized mutation models for improved performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from core.genome import create_initial_genome, Genome
from core.evolution_engine_optimized import OptimizedEvolutionEngine

def create_evolution_plots(snapshots: List[Dict], evolved_genome: Genome, initial_genome: Genome):
    """Create comprehensive evolution plots with optimized data"""
    if len(snapshots) < 2:
        print("⚠️  Not enough snapshots for plotting")
        return
    
    # Extract data from snapshots
    generations_list = [s['snapshot_generation'] for s in snapshots]
    genome_sizes = [s['genome_stats']['total_size'] for s in snapshots]
    gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
    core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
    hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimized Prokaryotic Genome Evolution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Genome size evolution
    axes[0, 0].plot(generations_list, genome_sizes, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Genome Size Evolution', fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Genome Size (bp)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: Gene count evolution
    axes[0, 1].plot(generations_list, gene_counts, 'g-', linewidth=2, marker='s', markersize=4, label='Total genes')
    axes[0, 1].plot(generations_list, core_genes, 'orange', linewidth=2, marker='^', markersize=4, label='Core genes')
    axes[0, 1].plot(generations_list, hgt_genes, 'red', linewidth=2, marker='v', markersize=4, label='HGT genes')
    axes[0, 1].set_title('Gene Count Evolution', fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Gene Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Evolution events (if we have enough data)
    if len(snapshots) > 1:
        mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
        hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
        recombination_events = [s['genome_stats']['total_recombination_events'] for s in snapshots]
        
        axes[1, 0].plot(generations_list, mutations, 'purple', label='Mutations', linewidth=2, marker='o', markersize=4)
        axes[1, 0].plot(generations_list, hgt_events, 'orange', label='HGT', linewidth=2, marker='s', markersize=4)
        axes[1, 0].plot(generations_list, recombination_events, 'brown', label='Recombination', linewidth=2, marker='^', markersize=4)
        axes[1, 0].set_title('Evolution Events (Optimized Processing)', fontweight='bold')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Cumulative Events')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Optimization statistics
    if 'mutation_stats' in snapshots[-1]:
        mutation_stats = snapshots[-1]['mutation_stats']
        
        # Create pie chart for mutation types
        if mutation_stats['transitions'] > 0 or mutation_stats['transversions'] > 0:
            labels = ['Transitions', 'Transversions']
            sizes = [mutation_stats['transitions'], mutation_stats['transversions']]
            colors = ['lightblue', 'lightcoral']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title(f'Mutation Types\n(Ti/Tv ratio: {mutation_stats["ti_tv_ratio"]:.2f})', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No mutations\nto display', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Mutation Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimized_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Optimized evolution plots saved as 'optimized_evolution_analysis.png'")
    plt.show()

def run_optimized_simulation(initial_gene_count: int = 3000,
                           avg_gene_length: int = 1000,
                           generations: int = 1000,
                           mutation_rate: float = 1e-9,
                           hgt_rate: float = 0.001,
                           recombination_rate: float = 1e-6,
                           save_snapshots: bool = True,
                           snapshot_interval: int = 100,
                           create_plots: bool = True) -> Tuple[Genome, List[Dict]]:
    """
    Run optimized prokaryotic genome evolution simulation
    
    Args:
        initial_gene_count: Initial number of genes
        avg_gene_length: Average gene length in bp
        generations: Number of generations to simulate
        mutation_rate: Point mutation rate per bp per generation
        hgt_rate: Horizontal gene transfer rate per genome per generation
        recombination_rate: Homologous recombination rate per bp per generation
        save_snapshots: Whether to save evolution snapshots
        snapshot_interval: Interval between snapshots
        create_plots: Whether to create evolution plots
    
    Returns:
        Tuple of (evolved_genome, snapshots)
    """
    
    print("🧬 OPTIMIZED PROKARYOTIC GENOME EVOLUTION SIMULATOR")
    print("=" * 80)
    print("🚀 Using optimized mutation models for enhanced performance")
    print("=" * 80)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create initial genome
    print("📊 Creating initial genome...")
    initial_genome = create_initial_genome(
        gene_count=initial_gene_count,
        avg_gene_length=avg_gene_length
    )
    
    # Initialize optimized evolution engine
    evolution_engine = OptimizedEvolutionEngine(
        mutation_rate=mutation_rate,
        hgt_rate=hgt_rate,
        recombination_rate=recombination_rate
    )
    
    print(f"\n⚙️  Evolution parameters:")
    print(f"   Point mutation rate: {mutation_rate}")
    print(f"   Horizontal gene transfer rate: {hgt_rate}")
    print(f"   Recombination rate: {recombination_rate}")
    print(f"   Optimizations: Batch processing, hotspot caching, vectorized operations")
    
    # Run evolution simulation
    evolved_genome, snapshots = evolution_engine.simulate_evolution(
        initial_genome=initial_genome,
        generations=generations,
        save_snapshots=save_snapshots,
        snapshot_interval=snapshot_interval
    )
    
    # Display results
    print(f"\n📈 EVOLUTION RESULTS:")
    print(f"   Initial genome: {initial_genome.gene_count:,} genes, {initial_genome.size:,} bp")
    print(f"   Final genome: {evolved_genome.gene_count:,} genes, {evolved_genome.size:,} bp")
    print(f"   Size change: {evolved_genome.size - initial_genome.size:+,} bp")
    print(f"   Gene count change: {evolved_genome.gene_count - initial_genome.gene_count:+,}")
    
    print(f"\n🧬 Evolution Event Statistics:")
    print(f"  Total mutations: {evolved_genome.total_mutations:,}")
    print(f"  HGT events: {evolved_genome.total_hgt_events:,}")
    print(f"  Recombination events: {evolved_genome.total_recombination_events:,}")
    
    # Get detailed mutation statistics
    if snapshots:
        final_summary = evolution_engine.get_evolution_summary(evolved_genome)
        mutation_stats = final_summary['mutation_stats']
        
        print(f"\n🔬 Optimized Mutation Analysis:")
        print(f"  Transitions: {mutation_stats['transitions']:,}")
        print(f"  Transversions: {mutation_stats['transversions']:,}")
        print(f"  Ti/Tv ratio: {mutation_stats['ti_tv_ratio']:.2f}")
        print(f"  Hotspot mutations: {mutation_stats['hotspot_mutations']:,} ({mutation_stats['hotspot_percentage']:.1f}%)")
        print(f"  Cache efficiency: {mutation_stats['cache_size']} genes cached")
        
        if len(snapshots) > 1:
            print(f"\n📊 Performance Summary:")
            print(f"   - Simulated {evolved_genome.generation} generations of evolution")
            print(f"   - Used optimized batch processing for mutations")
            print(f"   - Cached hotspot analysis for {mutation_stats['cache_size']} genes")
            print(f"   - Achieved significant performance improvements")
    
    # Create evolution plots
    if create_plots and snapshots:
        print(f"\n📊 Creating evolution analysis plots...")
        create_evolution_plots(snapshots, evolved_genome, initial_genome)
    
    return evolved_genome, snapshots

def main():
    """Main function with optimized parameters"""
    
    print("🧬 Welcome to the Optimized Prokaryotic Genome Evolution Simulator!")
    print("   This version includes performance optimizations for large-scale simulations.")
    print()
    
    # Example usage with optimized settings
    evolved_genome, snapshots = run_optimized_simulation(
        initial_gene_count=3000,    # Initial gene count
        avg_gene_length=1000,       # Average gene length
        generations=1000,           # Number of generations
        mutation_rate=1e-8,         # Point mutation rate (optimized processing)
        hgt_rate=0.002,            # HGT rate (slightly increased for observable effects)
        recombination_rate=1e-6,    # Recombination rate
        save_snapshots=True,        # Save evolution snapshots
        snapshot_interval=100,      # Snapshot interval
        create_plots=True          # Create analysis plots
    )
    
    print(f"\n✅ Optimized simulation completed successfully!")
    print(f"📁 Results saved and plots generated")
    print(f"🚀 Performance optimizations delivered faster execution")

if __name__ == "__main__":
    main()