#!/usr/bin/env python3
"""
Quick Optimization Test - Fast verification of mutation model improvements
"""

import time
import numpy as np
from core.genome import create_initial_genome
from mechanisms.point_mutation import PointMutationEngine
from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine

def quick_comparison():
    """Quick performance comparison"""
    print("⚡ QUICK OPTIMIZATION VERIFICATION")
    print("=" * 45)
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Small test for quick results
    genome = create_initial_genome(gene_count=200, avg_gene_length=1000)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # Test parameters
    mutation_rate = 1e-6  # High rate for visible effects
    generations = 20
    
    print(f"Testing {generations} generations at {mutation_rate} mutation rate...")
    
    # Test original engine
    print("\n🔬 Original Engine:")
    original_engine = PointMutationEngine(mutation_rate=mutation_rate)
    test_genome1 = genome.copy()
    
    start_time = time.time()
    total_mutations1 = 0
    for _ in range(generations):
        mutations = original_engine.apply_mutations(test_genome1, 1)
        total_mutations1 += mutations
    time1 = time.time() - start_time
    
    print(f"   Time: {time1:.3f}s")
    print(f"   Mutations: {total_mutations1:,}")
    print(f"   Speed: {generations/time1:.1f} gen/s")
    
    # Test optimized engine
    print("\n🚀 Optimized Engine:")
    optimized_engine = OptimizedPointMutationEngine(mutation_rate=mutation_rate)
    test_genome2 = genome.copy()
    
    start_time = time.time()
    total_mutations2 = 0
    for _ in range(generations):
        mutations = optimized_engine.apply_mutations(test_genome2, 1)
        total_mutations2 += mutations
    time2 = time.time() - start_time
    
    print(f"   Time: {time2:.3f}s")
    print(f"   Mutations: {total_mutations2:,}")
    print(f"   Speed: {generations/time2:.1f} gen/s")
    print(f"   Cache size: {len(optimized_engine._hotspot_cache)} genes")
    
    # Calculate improvement
    if time1 > 0 and time2 > 0:
        speedup = time1 / time2
        print(f"\n💡 Results:")
        print(f"   Speedup: {speedup:.2f}x faster")
        print(f"   Time saved: {time1 - time2:.3f}s")
        
        # Verify correctness
        mutation_diff = abs(total_mutations1 - total_mutations2)
        avg_mutations = (total_mutations1 + total_mutations2) / 2
        if avg_mutations > 0:
            relative_diff = mutation_diff / avg_mutations * 100
            print(f"   Accuracy: {100-relative_diff:.1f}% (mutation counts similar)")
        
        if speedup > 1.5:
            print("   🟢 SUCCESS: Significant optimization achieved!")
        elif speedup > 1.1:
            print("   🟡 GOOD: Noticeable improvement")
        else:
            print("   🟠 MINIMAL: Limited improvement")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    quick_comparison()