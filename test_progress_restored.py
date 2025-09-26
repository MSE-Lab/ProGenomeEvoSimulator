#!/usr/bin/env python3
"""
Test to verify that progress bar is restored and working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Testing Restored Progress Bar")
    print("=" * 50)
    
    # Create test genome
    genes = []
    for i in range(8):
        sequence = generate_random_sequence(200)
        gene = Gene(
            id=f"test_gene_{i}",
            sequence=sequence,
            start_pos=i * 200,
            length=200
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Create engine
    engine = EvolutionEngine(
        mutation_rate=1e-7,
        hgt_rate=0.05,
        recombination_rate=1e-5
    )
    
    print("\n" + "="*50)
    print("🚀 Testing progress bar restoration...")
    print("Should show ONE progress bar with real-time updates")
    print("="*50)
    
    # Test with enough generations to see progress updates
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=25,  # Enough to see progress updates every 5 generations
        save_snapshots=False
    )
    
    print("\n✅ Test completed!")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print("If you saw a progress bar with real-time updates, the fix worked! 🎉")

if __name__ == "__main__":
    main()