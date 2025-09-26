#!/usr/bin/env python3
"""
Quick test to verify progress display fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Quick Progress Display Test")
    print("=" * 40)
    
    # Create minimal genome
    genes = []
    for i in range(5):
        sequence = generate_random_sequence(100)
        gene = Gene(
            id=f"gene_{i}",
            sequence=sequence,
            start_pos=i * 100,
            length=100
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Test genome: {genome.gene_count} genes, {genome.size} bp")
    
    # Create engine
    engine = EvolutionEngine(
        mutation_rate=1e-7,
        hgt_rate=0.1,
        recombination_rate=1e-4
    )
    
    print("\n" + "="*40)
    print("🚀 Testing simulate_evolution method...")
    print("(Should show only ONE progress section)")
    print("="*40)
    
    # Test the method that was causing duplicate progress
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=10,
        save_snapshots=False
    )
    
    print("\n✅ Test completed!")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size} bp")

if __name__ == "__main__":
    main()