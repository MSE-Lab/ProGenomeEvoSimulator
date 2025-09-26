#!/usr/bin/env python3
"""
Longer test to see more obvious progress changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Longer Progress Test (to see more obvious changes)")
    print("=" * 60)
    
    # Create larger genome for slower evolution
    genes = []
    for i in range(50):  # More genes
        sequence = generate_random_sequence(800)  # Longer genes
        gene = Gene(
            id=f"long_gene_{i:03d}",
            sequence=sequence,
            start_pos=i * 800,
            length=800
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Larger test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Create engine with higher rates for more visible changes
    engine = EvolutionEngine(
        mutation_rate=1e-6,  # Higher mutation rate
        hgt_rate=0.1,        # Higher HGT rate
        recombination_rate=1e-4  # Higher recombination rate
    )
    
    print("\n" + "="*60)
    print("🚀 Running longer simulation (should see progress changes)")
    print("="*60)
    
    # Run longer simulation
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=200,  # More generations
        save_snapshots=False
    )
    
    print("\n" + "="*60)
    print("✅ LONGER TEST COMPLETED!")
    print(f"Initial: {genome.gene_count} genes, {genome.size:,} bp")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print(f"Changes: {final_genome.size - genome.size:+,} bp, {final_genome.gene_count - genome.gene_count:+,} genes")
    print("="*60)

if __name__ == "__main__":
    main()