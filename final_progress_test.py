#!/usr/bin/env python3
"""
Final test to verify single progress display
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import Genome, Gene, generate_random_sequence
from core.evolution_engine import EvolutionEngine

def main():
    print("🧪 FINAL Progress Display Test")
    print("=" * 60)
    
    # Create test genome
    genes = []
    for i in range(6):
        sequence = generate_random_sequence(300)
        gene = Gene(
            id=f"final_test_gene_{i}",
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
        hgt_rate=0.02,
        recombination_rate=1e-5
    )
    
    print("\n" + "="*60)
    print("🚀 TESTING: Should show EXACTLY ONE progress section")
    print("="*60)
    
    # Test with moderate number of generations
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=30,
        save_snapshots=False
    )
    
    print("\n" + "="*60)
    print("✅ FINAL TEST COMPLETED!")
    print(f"Result: {final_genome.gene_count} genes, {final_genome.size:,} bp")
    print("="*60)
    print("SUCCESS CRITERIA:")
    print("✓ Should see only ONE progress bar section")
    print("✓ Should see real-time progress updates")
    print("✓ Should see final completion message")
    print("✗ Should NOT see duplicate progress displays")
    print("="*60)

if __name__ == "__main__":
    main()