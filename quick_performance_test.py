#!/usr/bin/env python3
"""
Quick Performance Test for Mutation Model Optimization
"""

import time
import numpy as np
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine

def test_mutation_performance():
    """Test current mutation model performance"""
    print("🔬 MUTATION MODEL PERFORMANCE TEST")
    print("=" * 50)
    
    # Create small test genome
    print("Creating test genome...")
    genome = create_initial_genome(gene_count=100, avg_gene_length=1000)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Test with different mutation rates
    rates = [1e-9, 1e-8, 1e-7, 1e-6]
    
    for rate in rates:
        print(f"\nTesting mutation rate: {rate}")
        engine = EvolutionEngine(mutation_rate=rate, hgt_rate=0, recombination_rate=0)
        
        start_time = time.time()
        
        # Run 10 generations
        test_genome = genome.copy()
        for i in range(10):
            engine.evolve_one_generation(test_genome)
        
        elapsed = time.time() - start_time
        mutations_per_sec = test_genome.total_mutations / elapsed if elapsed > 0 else 0
        
        print(f"  Time: {elapsed:.3f}s, Mutations: {test_genome.total_mutations}, Rate: {mutations_per_sec:.1f} mut/s")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_mutation_performance()