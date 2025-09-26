#!/usr/bin/env python3
"""
Quick Speed Test - Automatic Performance Assessment
快速性能测试 - 自动评估演化速度
"""

import numpy as np
import time
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine

def run_speed_assessment():
    """Run automatic speed assessment"""
    
    print("⚡ AUTOMATIC SPEED ASSESSMENT")
    print("=" * 60)
    print("Testing evolution speed with realistic parameters...")
    print("This will help estimate time needed for 10,000 generations")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(42)
    
    # Test parameters
    test_generations = 50  # Small test first
    
    # Create realistic genome (3000 genes)
    print("📊 Creating test genome...")
    genome = create_initial_genome(
        gene_count=3000,
        avg_gene_length=1000
    )
    
    print(f"✅ Test genome: {genome.gene_count:,} genes, {genome.size:,} bp")
    print()
    
    # Set up evolution engine with realistic parameters
    engine = EvolutionEngine(
        mutation_rate=1e-9,      # Realistic rate
        hgt_rate=0.001,         # Realistic rate  
        recombination_rate=1e-6  # Realistic rate
    )
    
    print(f"🧬 Testing with {test_generations} generations...")
    print("   (Using realistic mutation rates)")
    print()
    
    # Run test
    start_time = time.time()
    
    try:
        evolved_genome, snapshots = engine.simulate_evolution(
            initial_genome=genome,
            generations=test_generations,
            save_snapshots=False  # Skip snapshots for speed
        )
        
        elapsed_time = time.time() - start_time
        
        # Calculate performance metrics
        rate = test_generations / elapsed_time
        
        print(f"\n📈 PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"⏱️  Test completed in: {elapsed_time:.2f} seconds")
        print(f"⚡ Speed: {rate:.2f} generations/second")
        print(f"📊 Time per generation: {elapsed_time/test_generations:.3f} seconds")
        print()
        
        # Extrapolate to different scales
        print("🔮 TIME ESTIMATES:")
        scales = [100, 500, 1000, 5000, 10000, 50000, 100000]
        
        for scale in scales:
            estimated_time = scale / rate
            if estimated_time < 60:
                time_str = f"{estimated_time:.1f} seconds"
            elif estimated_time < 3600:
                time_str = f"{estimated_time/60:.1f} minutes"
            elif estimated_time < 86400:
                time_str = f"{estimated_time/3600:.1f} hours"
            else:
                time_str = f"{estimated_time/86400:.1f} days"
            
            print(f"   {scale:6,} generations: {time_str}")
        
        print()
        
        # Performance assessment
        print("💡 PERFORMANCE ASSESSMENT:")
        if rate >= 10:
            print("   🟢 EXCELLENT: Very fast simulation speed")
            print("   ✅ 10,000 generations should complete quickly")
        elif rate >= 1:
            print("   🟡 GOOD: Reasonable simulation speed")
            print("   ✅ 10,000 generations feasible for overnight runs")
        elif rate >= 0.1:
            print("   🟠 MODERATE: Slower simulation speed")
            print("   ⚠️  10,000 generations will take several hours")
            print("   💡 Consider optimization for large-scale runs")
        else:
            print("   🔴 SLOW: Very slow simulation speed")
            print("   ⚠️  10,000 generations will take very long")
            print("   💡 Optimization strongly recommended")
        
        print()
        
        # Optimization suggestions
        if rate < 1:
            print("🚀 OPTIMIZATION SUGGESTIONS:")
            print("   1. Reduce mutation rates for testing")
            print("   2. Use smaller genomes for initial experiments")
            print("   3. Implement parallel processing")
            print("   4. Consider batch processing mutations")
            print("   5. Use more efficient data structures")
        
        # Evolution results
        print(f"\n🧬 EVOLUTION RESULTS (after {test_generations} generations):")
        print(f"   Genome size change: {evolved_genome.size - genome.size:+,} bp")
        print(f"   Gene count change: {evolved_genome.gene_count - genome.gene_count:+,}")
        print(f"   Total mutations: {evolved_genome.total_mutations:,}")
        print(f"   Total HGT events: {evolved_genome.total_hgt_events:,}")
        print(f"   Total recombination: {evolved_genome.total_recombination_events:,}")
        
        print("=" * 60)
        
        return rate, elapsed_time
        
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n⏹️  Test interrupted after {elapsed_time:.2f} seconds")
        return None, elapsed_time
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return None, 0

def main():
    """Main function"""
    
    print("🧬 PROKARYOTIC GENOME EVOLUTION SIMULATOR")
    print("   Performance Assessment Tool")
    print()
    
    rate, elapsed = run_speed_assessment()
    
    if rate:
        print(f"\n✅ Speed assessment completed!")
        print(f"   Your system can process ~{rate:.1f} generations/second")
        
        # Specific recommendation for 10k generations
        time_10k = 10000 / rate
        if time_10k < 3600:
            print(f"   10,000 generations estimated: {time_10k/60:.1f} minutes")
            print("   ✅ Suitable for interactive use")
        elif time_10k < 86400:
            print(f"   10,000 generations estimated: {time_10k/3600:.1f} hours")
            print("   ✅ Suitable for overnight runs")
        else:
            print(f"   10,000 generations estimated: {time_10k/86400:.1f} days")
            print("   ⚠️  Consider optimization or smaller scale tests")
    
    print(f"\n📝 Next steps:")
    print(f"   - Run 'python benchmark_10k.py' for full 10k test")
    print(f"   - Run 'python demo.py' to see detailed progress display")
    print(f"   - Modify parameters in main.py for your research needs")

if __name__ == "__main__":
    main()