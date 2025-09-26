#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized Mutation Engine
"""

import time
import numpy as np
from core.genome import create_initial_genome, Genome
from mechanisms.point_mutation import PointMutationEngine
from mechanisms.point_mutation_optimized import OptimizedPointMutationEngine

def benchmark_mutation_engine(engine, genome: Genome, generations: int, name: str):
    """Benchmark a mutation engine"""
    print(f"\n🔬 Testing {name}...")
    
    # Create a copy to avoid modifying original
    test_genome = genome.copy()
    
    start_time = time.time()
    total_mutations = 0
    
    for gen in range(generations):
        mutations = engine.apply_mutations(test_genome, generations=1)
        total_mutations += mutations
    
    elapsed_time = time.time() - start_time
    
    # Calculate performance metrics
    mutations_per_second = total_mutations / elapsed_time if elapsed_time > 0 else 0
    generations_per_second = generations / elapsed_time if elapsed_time > 0 else 0
    
    print(f"  ⏱️  Time: {elapsed_time:.3f}s")
    print(f"  🧬 Total mutations: {total_mutations:,}")
    print(f"  ⚡ Speed: {generations_per_second:.2f} gen/s, {mutations_per_second:.1f} mut/s")
    
    return {
        'name': name,
        'time': elapsed_time,
        'mutations': total_mutations,
        'gen_per_sec': generations_per_second,
        'mut_per_sec': mutations_per_second,
        'final_genome': test_genome
    }

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🚀 MUTATION ENGINE PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test parameters
    test_configs = [
        {'genes': 100, 'gens': 50, 'rate': 1e-7},
        {'genes': 500, 'gens': 20, 'rate': 1e-7},
        {'genes': 1000, 'gens': 10, 'rate': 1e-7},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n📊 Test Configuration:")
        print(f"   Genes: {config['genes']:,}, Generations: {config['gens']}, Rate: {config['rate']}")
        print("-" * 60)
        
        # Create test genome
        genome = create_initial_genome(
            gene_count=config['genes'], 
            avg_gene_length=1000
        )
        
        # Initialize engines
        original_engine = PointMutationEngine(
            mutation_rate=config['rate'],
            enable_transition_bias=True,
            enable_hotspots=True
        )
        
        optimized_engine = OptimizedPointMutationEngine(
            mutation_rate=config['rate'],
            enable_transition_bias=True,
            enable_hotspots=True
        )
        
        # Benchmark both engines
        original_result = benchmark_mutation_engine(
            original_engine, genome, config['gens'], "Original Engine"
        )
        
        optimized_result = benchmark_mutation_engine(
            optimized_engine, genome, config['gens'], "Optimized Engine"
        )
        
        # Calculate improvement
        if original_result['time'] > 0:
            speedup = original_result['time'] / optimized_result['time']
            print(f"\n💡 Performance Improvement:")
            print(f"   Speedup: {speedup:.2f}x faster")
            print(f"   Time saved: {original_result['time'] - optimized_result['time']:.3f}s")
        
        # Verify correctness (mutation counts should be similar)
        mutation_diff = abs(original_result['mutations'] - optimized_result['mutations'])
        mutation_avg = (original_result['mutations'] + optimized_result['mutations']) / 2
        relative_diff = mutation_diff / mutation_avg * 100 if mutation_avg > 0 else 0
        
        print(f"   Mutation count difference: {relative_diff:.1f}% (should be <10%)")
        
        results.append({
            'config': config,
            'original': original_result,
            'optimized': optimized_result,
            'speedup': speedup if original_result['time'] > 0 else 0
        })
    
    # Summary
    print(f"\n🎯 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    total_speedup = 0
    valid_tests = 0
    
    for i, result in enumerate(results):
        config = result['config']
        speedup = result['speedup']
        
        print(f"Test {i+1} ({config['genes']} genes, {config['gens']} gens): {speedup:.2f}x speedup")
        
        if speedup > 0:
            total_speedup += speedup
            valid_tests += 1
    
    if valid_tests > 0:
        avg_speedup = total_speedup / valid_tests
        print(f"\n🏆 Average speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 2.0:
            print("🟢 EXCELLENT: Significant performance improvement!")
        elif avg_speedup > 1.5:
            print("🟡 GOOD: Noticeable performance improvement")
        elif avg_speedup > 1.1:
            print("🟠 MODERATE: Some performance improvement")
        else:
            print("🔴 MINIMAL: Limited performance improvement")
    
    print("\n✅ Performance comparison completed!")
    return results

def test_memory_usage():
    """Test memory usage of both engines"""
    print(f"\n🧠 MEMORY USAGE COMPARISON")
    print("-" * 40)
    
    # Create test genome
    genome = create_initial_genome(gene_count=1000, avg_gene_length=1000)
    
    # Test original engine
    original_engine = PointMutationEngine(mutation_rate=1e-7, enable_hotspots=True)
    
    # Test optimized engine
    optimized_engine = OptimizedPointMutationEngine(mutation_rate=1e-7, enable_hotspots=True)
    
    # Run a few generations to populate caches
    for _ in range(5):
        original_engine.apply_mutations(genome.copy(), 1)
        optimized_engine.apply_mutations(genome.copy(), 1)
    
    # Check cache sizes
    print(f"Optimized engine cache size: {len(optimized_engine._hotspot_cache)} genes cached")
    print(f"Memory optimization: Hotspot positions cached for reuse")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    results = run_performance_comparison()
    test_memory_usage()
    
    print(f"\n📝 Next Steps:")
    print(f"   - If performance is good, update evolution_engine.py to use optimized version")
    print(f"   - Run larger scale tests with benchmark_10k.py")
    print(f"   - Consider additional optimizations if needed")