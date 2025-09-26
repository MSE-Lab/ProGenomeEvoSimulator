#!/usr/bin/env python3
"""
Test Progress Display
测试进度显示功能
"""

import numpy as np
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine

def test_progress_display():
    """Test the progress display functionality"""
    
    print("🧪 TESTING PROGRESS DISPLAY")
    print("=" * 50)
    
    # Set random seed
    np.random.seed(42)
    
    # Create small genome for quick test
    print("Creating small test genome...")
    genome = create_initial_genome(gene_count=50, avg_gene_length=200)
    print(f"Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    print()
    
    # Create evolution engine
    engine = EvolutionEngine(
        mutation_rate=1e-6,      # Higher rate for visible effects
        hgt_rate=0.01,          # Higher rate for visible effects
        recombination_rate=1e-4  # Higher rate for visible effects
    )
    
    # Test with 25 generations (should show progress every 5 generations)
    print("Testing with 25 generations...")
    print("(Progress should display every 5 generations)")
    print()
    
    try:
        evolved_genome, snapshots = engine.simulate_evolution(
            initial_genome=genome,
            generations=25,
            save_snapshots=False
        )
        
        print("\n✅ Progress display test completed successfully!")
        print(f"Final genome: {evolved_genome.gene_count} genes, {evolved_genome.size:,} bp")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_progress_display()