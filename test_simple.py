#!/usr/bin/env python3
"""
简化测试版本 - 测试基本功能
"""

import sys
import numpy as np

def test_basic_imports():
    """Test basic imports"""
    try:
        from core.genome import create_initial_genome, Gene, Genome
        from core.evolution_engine import EvolutionEngine
        from analysis.ani_calculator import ANICalculator
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_genome_creation():
    """Test genome creation"""
    try:
        from core.genome import create_initial_genome
        
        print("Creating test genome...")
        genome = create_initial_genome(gene_count=100, avg_gene_length=500)
        
        print(f"✓ Genome created successfully:")
        print(f"  - Gene count: {genome.gene_count}")
        print(f"  - Genome size: {genome.size:,} bp")
        print(f"  - Average gene length: {genome.size/genome.gene_count:.1f} bp")
        
        return genome
    except Exception as e:
        print(f"✗ Genome creation failed: {e}")
        return None

def test_evolution_engine():
    """Test evolution engine"""
    try:
        from core.genome import create_initial_genome
        from core.evolution_engine import EvolutionEngine
        
        print("\nTesting evolution engine...")
        
        # Create small genome
        genome = create_initial_genome(gene_count=50, avg_gene_length=300)
        initial_stats = genome.get_statistics()
        
        # Create evolution engine
        engine = EvolutionEngine(
            mutation_rate=1e-7,  # Increased mutation rate for observation
            hgt_rate=0.01,       # Increased HGT rate
            recombination_rate=1e-4  # Increased recombination rate
        )
        
        print(f"Initial state: {initial_stats['gene_count']} genes, {initial_stats['total_size']:,} bp")
        
        # Evolve for 10 generations
        print("Evolving for 10 generations...")
        for i in range(10):
            stats = engine.evolve_one_generation(genome)
            if i % 5 == 4:  # Output every 5 generations
                print(f"  Generation {i+1}: {stats['mutations']} mutations, {stats['hgt_events']} HGT, {stats['recombination_events']} recombination")
        
        final_stats = genome.get_statistics()
        print(f"Final state: {final_stats['gene_count']} genes, {final_stats['total_size']:,} bp")
        print(f"Total changes: {final_stats['total_mutations']} mutations, {final_stats['total_hgt_events']} HGT, {final_stats['total_recombination_events']} recombination")
        
        print("✓ Evolution engine test successful")
        return genome
        
    except Exception as e:
        print(f"✗ Evolution engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ani_calculation():
    """Test ANI calculation"""
    try:
        from core.genome import create_initial_genome
        from core.evolution_engine import EvolutionEngine
        from analysis.ani_calculator import ANICalculator
        
        print("\nTesting ANI calculation...")
        
        # Create initial genome
        initial_genome = create_initial_genome(gene_count=30, avg_gene_length=400)
        
        # Create copy and evolve
        evolved_genome = initial_genome.copy()
        engine = EvolutionEngine(mutation_rate=1e-6, hgt_rate=0.005, recombination_rate=1e-4)
        
        # Evolve for 5 generations
        for i in range(5):
            engine.evolve_one_generation(evolved_genome)
        
        # Calculate ANI
        calculator = ANICalculator(ortholog_identity_threshold=0.3)  # Lower threshold
        analysis = calculator.compare_genomes_comprehensive(initial_genome, evolved_genome)
        
        ani_data = analysis['ani_analysis']
        print(f"✓ ANI calculation successful:")
        print(f"  - ANI: {ani_data['ani']:.4f}")
        print(f"  - Orthologous gene pairs: {ani_data['ortholog_count']}")
        print(f"  - Ortholog ratio: {ani_data['ortholog_ratio']:.4f}")
        
        identity_dist = analysis['identity_distribution']
        if identity_dist['sample_size'] > 0:
            print(f"  - Identity distribution: mean={identity_dist['mean']:.4f}, std={identity_dist['std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ ANI calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Prokaryotic Genome Evolution Simulator - Simplified Test")
    print("=" * 50)
    
    # Set random seed
    np.random.seed(42)
    
    # Test steps
    success = True
    
    # 1. Test imports
    if not test_basic_imports():
        success = False
        return
    
    # 2. Test genome creation
    genome = test_genome_creation()
    if genome is None:
        success = False
        return
    
    # 3. Test evolution engine
    evolved_genome = test_evolution_engine()
    if evolved_genome is None:
        success = False
        return
    
    # 4. Test ANI calculation
    if not test_ani_calculation():
        success = False
        return
    
    if success:
        print("\n" + "=" * 50)
        print("✓ All tests passed! Simulator is working properly.")
        print("You can now run the full version: python main.py")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ Some tests failed, please check the code.")
        print("=" * 50)

if __name__ == "__main__":
    main()