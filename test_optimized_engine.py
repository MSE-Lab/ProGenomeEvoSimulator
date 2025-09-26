#!/usr/bin/env python3
"""
Test Optimized Evolution Engine
"""

import time
import numpy as np
from core.genome import create_initial_genome
from core.evolution_engine_optimized import OptimizedEvolutionEngine

def test_optimized_engine():
    """Test the optimized evolution engine"""
    print("🚀 TESTING OPTIMIZED EVOLUTION ENGINE")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create test genome
    print("Creating test genome...")
    genome = create_initial_genome(gene_count=500, avg_gene_length=1000)
    print(f"Test genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    
    # Create optimized engine
    engine = OptimizedEvolutionEngine(
        mutation_rate=1e-7,  # Higher rate for visible effects
        hgt_rate=0.01,       # Higher rate for visible effects
        recombination_rate=1e-5
    )
    
    print(f"\nRunning 100 generations with optimized engine...")
    
    start_time = time.time()
    
    # Run evolution
    evolved_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=100,
        save_snapshots=True,
        snapshot_interval=25
    )
    
    total_time = time.time() - start_time
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Speed: {100/total_time:.2f} generations/second")
    print(f"   Initial genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    print(f"   Final genome: {evolved_genome.gene_count:,} genes, {evolved_genome.size:,} bp")
    print(f"   Changes: {evolved_genome.size - genome.size:+,} bp, {evolved_genome.gene_count - genome.gene_count:+,} genes")
    print(f"   Total mutations: {evolved_genome.total_mutations:,}")
    print(f"   Total HGT events: {evolved_genome.total_hgt_events:,}")
    print(f"   Total recombination: {evolved_genome.total_recombination_events:,}")
    
    # Get detailed statistics
    summary = engine.get_evolution_summary(evolved_genome)
    mutation_stats = summary['mutation_stats']
    
    print(f"\n🧬 MUTATION STATISTICS:")
    print(f"   Transitions: {mutation_stats['transitions']:,}")
    print(f"   Transversions: {mutation_stats['transversions']:,}")
    print(f"   Ti/Tv ratio: {mutation_stats['ti_tv_ratio']:.2f}")
    print(f"   Hotspot mutations: {mutation_stats['hotspot_mutations']:,} ({mutation_stats['hotspot_percentage']:.1f}%)")
    print(f"   Cache efficiency: {mutation_stats['cache_size']} genes cached")
    
    print(f"\n✅ Optimized engine test completed successfully!")
    
    return total_time, evolved_genome

if __name__ == "__main__":
    test_optimized_engine()