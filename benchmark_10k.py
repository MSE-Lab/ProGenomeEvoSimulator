#!/usr/bin/env python3
"""
10,000 Generation Benchmark Test
用于测试10000代演化所需的时间
"""

import numpy as np
import time
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine
from analysis.ani_calculator import ANICalculator

def run_10k_benchmark():
    """Run 10,000 generation benchmark"""
    
    print("🚀 10,000 GENERATION BENCHMARK TEST")
    print("=" * 80)
    print("This test will help estimate the time needed for large-scale simulations")
    print("You can stop the test at any time with Ctrl+C to get time estimates")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create realistic initial genome (3000 genes as requested)
    print("📊 Creating initial genome (3000 genes)...")
    initial_genome = create_initial_genome(
        gene_count=3000,
        avg_gene_length=1000,
        gene_length_std=200
    )
    
    print(f"✅ Initial genome created:")
    print(f"   - Genes: {initial_genome.gene_count:,}")
    print(f"   - Size: {initial_genome.size:,} bp")
    print(f"   - Avg gene length: {initial_genome.size/initial_genome.gene_count:.1f} bp")
    print()
    
    # Set up evolution engine with realistic parameters
    print("⚙️  Setting up evolution engine...")
    evolution_engine = EvolutionEngine(
        mutation_rate=1e-9,      # Realistic mutation rate
        hgt_rate=0.001,         # Realistic HGT rate
        recombination_rate=1e-6  # Realistic recombination rate
    )
    
    print(f"   - Point mutation rate: 1e-9 per bp per generation")
    print(f"   - HGT rate: 0.001 per genome per generation")
    print(f"   - Recombination rate: 1e-6 per bp per generation")
    print()
    
    # Start benchmark
    benchmark_start = time.time()
    
    try:
        # Run 10,000 generation evolution
        print("🧬 Starting 10,000 generation evolution...")
        print("   (Press Ctrl+C to stop and get time estimates)")
        print()
        
        evolved_genome, snapshots = evolution_engine.simulate_evolution(
            initial_genome=initial_genome,
            generations=10000,
            save_snapshots=True,
            snapshot_interval=500  # Save snapshots every 500 generations
        )
        
        # If completed successfully
        total_time = time.time() - benchmark_start
        
        print("\n🎉 BENCHMARK COMPLETED!")
        print("=" * 80)
        print(f"⏱️  Total time for 10,000 generations: {total_time/3600:.2f} hours")
        print(f"⚡ Average speed: {10000/total_time:.2f} generations/second")
        print(f"📊 Time per 1000 generations: {total_time/10:.1f} minutes")
        
        # Calculate ANI for final comparison
        print("\n📈 Calculating final ANI...")
        ani_calculator = ANICalculator()
        analysis = ani_calculator.compare_genomes_comprehensive(
            ancestral_genome=initial_genome,
            evolved_genome=evolved_genome
        )
        
        ani_data = analysis['ani_analysis']
        print(f"🧬 Final ANI: {ani_data['ani']:.4f}")
        print(f"🔗 Orthologous genes: {ani_data['ortholog_count']}")
        
        # Performance summary
        print("\n📋 PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Genome size change: {evolved_genome.size - initial_genome.size:+,} bp")
        print(f"Gene count change: {evolved_genome.gene_count - initial_genome.gene_count:+,}")
        print(f"Total mutations: {evolved_genome.total_mutations:,}")
        print(f"Total HGT events: {evolved_genome.total_hgt_events:,}")
        print(f"Total recombination: {evolved_genome.total_recombination_events:,}")
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        elapsed_time = time.time() - benchmark_start
        completed_gens = evolution_engine.evolution_history[-1]['generation'] if evolution_engine.evolution_history else 0
        
        print(f"\n\n⏹️  BENCHMARK INTERRUPTED")
        print("=" * 80)
        print(f"⏱️  Elapsed time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
        print(f"📊 Completed generations: {completed_gens:,}")
        
        if completed_gens > 0:
            rate = completed_gens / elapsed_time
            estimated_total_time = 10000 / rate
            
            print(f"⚡ Current speed: {rate:.2f} generations/second")
            print(f"📈 Estimated time for 10,000 generations: {estimated_total_time/3600:.2f} hours")
            print(f"📈 Estimated time for 1,000 generations: {estimated_total_time/10/60:.1f} minutes")
            
            # Extrapolate for different scales
            print(f"\n🔮 TIME ESTIMATES FOR DIFFERENT SCALES:")
            print(f"   - 1,000 generations: {estimated_total_time/10/60:.1f} minutes")
            print(f"   - 5,000 generations: {estimated_total_time/2/60:.1f} minutes")
            print(f"   - 10,000 generations: {estimated_total_time/60:.1f} minutes ({estimated_total_time/3600:.2f} hours)")
            print(f"   - 50,000 generations: {estimated_total_time*5/3600:.1f} hours")
            print(f"   - 100,000 generations: {estimated_total_time*10/3600:.1f} hours ({estimated_total_time*10/3600/24:.1f} days)")
        
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        if completed_gens > 0 and rate < 1.0:
            print("   - Current speed is slow, consider optimization")
            print("   - Try reducing mutation/HGT/recombination rates for testing")
            print("   - Consider implementing parallel processing")
            print("   - Use smaller genome for initial testing")
        
        print("=" * 80)

def quick_speed_test():
    """Quick speed test with 100 generations"""
    
    print("⚡ QUICK SPEED TEST (100 generations)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Small genome for quick test
    genome = create_initial_genome(gene_count=3000, avg_gene_length=1000)
    engine = EvolutionEngine(mutation_rate=1e-9, hgt_rate=0.001, recombination_rate=1e-6)
    
    start_time = time.time()
    evolved_genome, _ = engine.simulate_evolution(genome, generations=100, save_snapshots=False)
    elapsed = time.time() - start_time
    
    rate = 100 / elapsed
    estimated_10k = 10000 / rate
    
    print(f"\n📊 QUICK TEST RESULTS:")
    print(f"   - 100 generations completed in: {elapsed:.2f} seconds")
    print(f"   - Speed: {rate:.2f} generations/second")
    print(f"   - Estimated time for 10,000 generations: {estimated_10k/3600:.2f} hours")
    print("=" * 50)

def main():
    """Main function"""
    
    print("Choose test mode:")
    print("1. Quick speed test (100 generations)")
    print("2. Full 10,000 generation benchmark")
    print("3. Both tests")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        quick_speed_test()
    elif choice == "2":
        run_10k_benchmark()
    elif choice == "3":
        quick_speed_test()
        print("\n" + "="*80 + "\n")
        input("Press Enter to continue with full benchmark...")
        run_10k_benchmark()
    else:
        print("Invalid choice, running quick test...")
        quick_speed_test()

if __name__ == "__main__":
    main()