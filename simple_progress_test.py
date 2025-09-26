#!/usr/bin/env python3
"""
Simple test for single-line progress bar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 Simple Single-Line Progress Test")
    print("=" * 50)
    
    # Create minimal genome
    genes = []
    for i in range(5):
        sequence = generate_random_sequence(200)
        gene = Gene(
            id=f"gene_{i}",
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
        hgt_rate=0.02,
        recombination_rate=1e-6
    )
    
    print("\n" + "="*50)
    print("🚀 Testing progress bar (should update on same line)")
    print("="*50)
    
    # Test with just the evolve_multiple_generations method directly
    print("Testing evolve_multiple_generations directly:")
    final_genome = genome.copy()
    history = engine.evolve_multiple_generations(final_genome, 50, show_progress=True)
    
    print(f"\n✅ Direct test completed!")
    print(f"Final: {final_genome.gene_count} genes, {final_genome.size:,} bp")

if __name__ == "__main__":
    main()