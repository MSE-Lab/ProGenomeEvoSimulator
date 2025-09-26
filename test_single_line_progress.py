#!/usr/bin/env python3
"""
Test to verify progress bar updates on the same line
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Testing Single-Line Progress Bar Updates")
    print("=" * 60)
    
    # Create test genome
    genes = []
    for i in range(20):
        sequence = generate_random_sequence(500)
        gene = Gene(
            id=f"test_gene_{i:02d}",
            sequence=sequence,
            start_pos=i * 500,
            length=500
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Create engine
    engine = EvolutionEngine(
        mutation_rate=1e-6,
        hgt_rate=0.05,
        recombination_rate=1e-5
    )
    
    print("\n" + "="*60)
    print("🚀 Testing single-line progress updates...")
    print("Should see ONE progress bar that updates in place")
    print("="*60)
    
    # Test with moderate generations to see updates
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=100,  # Enough to see multiple updates
        save_snapshots=False
    )
    
    print("\n" + "="*60)
    print("✅ SINGLE-LINE PROGRESS TEST COMPLETED!")
    print(f"Initial: {genome.gene_count} genes, {genome.size:,} bp")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print("="*60)
    print("SUCCESS: Progress bar should update on the same line!")
    print("FAILURE: If you saw multiple progress bar lines, it needs more fixing")
    print("="*60)

if __name__ == "__main__":
    main()