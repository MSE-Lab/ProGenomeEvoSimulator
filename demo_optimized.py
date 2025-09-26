#!/usr/bin/env python3
"""
优化版基因组进化模拟器演示
使用优化的突变引擎提供更快的演示体验
"""

import numpy as np
import matplotlib.pyplot as plt
from core.genome import create_initial_genome
from core.evolution_engine_optimized import OptimizedEvolutionEngine
from analysis.ani_calculator import ANICalculator

def run_optimized_demo():
    """运行优化版演示"""
    
    print("=" * 60)
    print("🚀 Optimized Prokaryotic Genome Evolution Simulator - Demo")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. Create initial genome with realistic gene length distribution
    print("1. Creating initial genome with realistic gene length distribution...")
    initial_genome = create_initial_genome(
        gene_count=100,           # 100 genes for demo
        avg_gene_length=1000,     # realistic average 1000bp
        min_gene_length=100       # minimum 100bp
    )
    
    print(f"   ✓ Genome created with realistic prokaryotic gene length distribution")
    print()
    
    # 2. Set evolution parameters (increased rates for observable effects)
    print("2. Setting optimized evolution parameters...")
    evolution_engine = OptimizedEvolutionEngine(
        mutation_rate=1e-6,      # increased mutation rate
        hgt_rate=0.01,          # increased HGT rate
        recombination_rate=1e-4  # increased recombination rate
    )
    print("   ✓ Point mutation rate: 1e-6 (per bp per generation) - OPTIMIZED")
    print("   ✓ HGT rate: 0.01 (per genome per generation)")
    print("   ✓ Recombination rate: 1e-4 (per bp per generation)")
    print("   ✓ Using batch processing and hotspot caching for faster execution")
    print()
    
    # 3. Run evolution simulation (fewer generations)
    print("3. Running optimized evolution simulation (50 generations)...")
    print("   This will show detailed progress tracking with enhanced performance...")
    evolved_genome, snapshots = evolution_engine.simulate_evolution(
        initial_genome=initial_genome,
        generations=50,
        save_snapshots=True,
        snapshot_interval=10
    )
    print()
    
    # 4. Analyze results
    print("4. Analyzing evolution results...")
    ani_calculator = ANICalculator(ortholog_identity_threshold=0.3)  # lower threshold
    
    comprehensive_analysis = ani_calculator.compare_genomes_comprehensive(
        ancestral_genome=initial_genome,
        evolved_genome=evolved_genome
    )
    
    # 5. Output key results
    print_optimized_demo_results(comprehensive_analysis, initial_genome, evolved_genome, evolution_engine)
    
    # 6. Simple visualization
    create_optimized_demo_visualization(comprehensive_analysis, snapshots)
    
    return initial_genome, evolved_genome, comprehensive_analysis

def print_optimized_demo_results(analysis, initial_genome, evolved_genome, engine):
    """Print optimized demo results"""
    
    print("=" * 60)
    print("📊 Optimized Evolution Analysis Results")
    print("=" * 60)
    
    # Basic statistics
    print("Genome Changes:")
    print(f"  Initial: {initial_genome.gene_count} genes, {initial_genome.size:,} bp")
    print(f"  Final: {evolved_genome.gene_count} genes, {evolved_genome.size:,} bp")
    print(f"  Change: {evolved_genome.gene_count - initial_genome.gene_count:+d} genes, {evolved_genome.size - initial_genome.size:+,} bp")
    print()
    
    # Evolution event statistics
    print("Evolution Event Statistics:")
    print(f"  Total mutations: {evolved_genome.total_mutations}")
    print(f"  HGT events: {evolved_genome.total_hgt_events}")
    print(f"  Recombination events: {evolved_genome.total_recombination_events}")
    print()
    
    # Optimization statistics
    print("🚀 Optimization Performance:")
    cache_size = len(engine.point_mutation._hotspot_cache)
    print(f"  Genes cached for hotspot analysis: {cache_size}")
    print(f"  Batch processing enabled: ✓")
    print(f"  Vectorized operations: ✓")
    print()
    
    # ANI analysis
    ani_data = analysis['ani_analysis']
    print("ANI Analysis:")
    print(f"  Average Nucleotide Identity: {ani_data['ani']:.4f}")
    print(f"  Weighted ANI: {ani_data['weighted_ani']:.4f}")
    print(f"  Orthologous gene pairs: {ani_data['ortholog_count']}")
    print(f"  Ortholog ratio: {ani_data['ortholog_ratio']:.4f}")
    print()
    
    # Identity distribution
    if analysis['identity_distribution']['sample_size'] > 0:
        identity_dist = analysis['identity_distribution']
        print("Orthologous Gene Identity Distribution:")
        print(f"  Mean: {identity_dist['mean']:.4f}")
        print(f"  Standard deviation: {identity_dist['std']:.4f}")
        print(f"  Range: {identity_dist['min']:.4f} - {identity_dist['max']:.4f}")
        print(f"  Sample size: {identity_dist['sample_size']}")
    else:
        print("Orthologous Gene Identity Distribution: Insufficient data")
    print()

def create_optimized_demo_visualization(analysis, snapshots):
    """Create optimized demo visualization"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Optimized Genome Evolution Simulation Demo Results', fontsize=14)
        
        # 1. Orthologous gene identity distribution
        if analysis['identity_distribution']['sample_size'] > 0:
            identities = analysis['ani_analysis']['identity_distribution']
            axes[0, 0].hist(identities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 0].axvline(np.mean(identities), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(identities):.3f}')
            axes[0, 0].set_xlabel('Sequence Identity')
            axes[0, 0].set_ylabel('Number of Gene Pairs')
            axes[0, 0].set_title('Orthologous Gene Identity Distribution (Optimized)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data\nfor distribution', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Orthologous Gene Identity Distribution')
        
        # 2. Genome size changes
        if len(snapshots) > 1:
            generations = [s.get('snapshot_generation', 0) for s in snapshots]
            sizes = [s['genome_stats']['total_size'] for s in snapshots]
            axes[0, 1].plot(generations, sizes, 'b-o', markersize=4, linewidth=2)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Genome Size (bp)')
            axes[0, 1].set_title('Genome Size Changes (Optimized Processing)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gene count changes
        if len(snapshots) > 1:
            gene_counts = [s['genome_stats']['gene_count'] for s in snapshots]
            core_genes = [s['genome_stats']['core_genes'] for s in snapshots]
            hgt_genes = [s['genome_stats']['hgt_genes'] for s in snapshots]
            
            axes[1, 0].plot(generations, gene_counts, 'g-o', label='Total genes', markersize=4, linewidth=2)
            axes[1, 0].plot(generations, core_genes, 'b-s', label='Core genes', markersize=4, linewidth=2)
            axes[1, 0].plot(generations, hgt_genes, 'r-^', label='HGT genes', markersize=4, linewidth=2)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Gene Count')
            axes[1, 0].set_title('Gene Count Changes')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Evolution event accumulation
        if len(snapshots) > 1:
            mutations = [s['genome_stats']['total_mutations'] for s in snapshots]
            hgt_events = [s['genome_stats']['total_hgt_events'] for s in snapshots]
            recombinations = [s['genome_stats']['total_recombination_events'] for s in snapshots]
            
            axes[1, 1].plot(generations, mutations, 'purple', label='Mutations (Optimized)', linewidth=2)
            axes[1, 1].plot(generations, hgt_events, 'orange', label='HGT', linewidth=2)
            axes[1, 1].plot(generations, recombinations, 'green', label='Recombination', linewidth=2)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Cumulative Events')
            axes[1, 1].set_title('Evolution Event Accumulation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_optimized_results.png', dpi=300, bbox_inches='tight')
        print("📈 Optimized visualization results saved as 'demo_optimized_results.png'")
        
        # Try to display the plot (if in supported environment)
        try:
            plt.show()
        except:
            print("   (Cannot display plot, but file saved)")
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def main():
    """Main function"""
    
    print("🚀 Starting optimized genome evolution simulation demo...")
    print("   This version uses enhanced mutation models for faster execution")
    print()
    
    try:
        initial_genome, evolved_genome, analysis = run_optimized_demo()
        
        print("=" * 60)
        print("✅ Optimized Demo completed!")
        print()
        print("📝 Summary:")
        print(f"   - Simulated {evolved_genome.generation} generations of evolution")
        print(f"   - {evolved_genome.total_mutations} mutations occurred (optimized processing)")
        print(f"   - {evolved_genome.total_hgt_events} horizontal gene transfer events")
        print(f"   - {evolved_genome.total_recombination_events} homologous recombination events")
        print(f"   - Final ANI: {analysis['ani_analysis']['ani']:.4f}")
        print()
        print("🚀 Performance improvements:")
        print("   - Batch processing for mutations")
        print("   - Hotspot caching for repeated calculations")
        print("   - Vectorized operations for faster computation")
        print()
        print("🔧 You can modify parameters and re-run, or use main_optimized.py for larger-scale simulations")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during optimized demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()