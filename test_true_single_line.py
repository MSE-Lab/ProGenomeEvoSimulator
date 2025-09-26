#!/usr/bin/env python3
"""
Test for true single-line progress bar updates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Testing TRUE Single-Line Progress Bar")
    print("=" * 60)
    
    # Create test genome
    genes = []
    for i in range(8):
        sequence = generate_random_sequence(300)
        gene = Gene(
            id=f"test_{i}",
            sequence=sequence,
            start_pos=i * 300,
            length=300
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Create engine
    engine = EvolutionEngine(
        mutation_rate=1e-7,
        hgt_rate=0.01,
        recombination_rate=1e-6
    )
    
    print("\n" + "="*60)
    print("🚀 Testing: Should see ONE line that updates in place")
    print("="*60)
    
    # Test directly with evolve_multiple_generations
    final_genome = genome.copy()
    history = engine.evolve_multiple_generations(final_genome, 40, show_progress=True)
    
    print(f"\n✅ Test completed!")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print("\nSUCCESS: If you saw only ONE progress line that updated in place! 🎉")
    print("FAILURE: If you saw multiple progress lines printed.")

if __name__ == "__main__":
    main()