#!/usr/bin/env python3
"""
Test script to verify that the progress display issue has been fixed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def test_progress_display():
    """Test that only one progress display appears"""
    print("🧪 Testing Progress Display Fix")
    print("=" * 50)
    
    # Create a small test genome
    print("Creating test genome...")
    genes = []
    for i in range(10):
        sequence = generate_random_sequence(500)
        gene = Gene(
            id=f"gene_{i:03d}",
            sequence=sequence,
            start_pos=i * 500,
            length=500,
            is_core=True,
            origin="ancestral"
        )
        genes.append(gene)
    
    genome = Genome(genes)
    print(f"Initial genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Create evolution engine
    engine = EvolutionEngine(
        mutation_rate=1e-8,
        hgt_rate=0.01,
        recombination_rate=1e-5
    )
    
    print("\n" + "="*50)
    print("🚀 Starting evolution test (should show only ONE progress display)")
    print("="*50)
    
    # Run a short simulation using simulate_evolution
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=20,
        save_snapshots=False
    )
    
    print("\n" + "="*50)
    print("✅ Test completed!")
    print(f"Final genome: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print("If you saw only ONE progress section above, the fix worked! 🎉")
    print("="*50)

if __name__ == "__main__":
    test_progress_display()