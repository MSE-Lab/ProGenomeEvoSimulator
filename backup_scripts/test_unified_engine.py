#!/usr/bin/env python3
"""
统一进化引擎测试脚本
验证UnifiedEvolutionEngine的所有功能
"""

import time
import numpy as np
from typing import Dict, List

from core.genome import create_initial_genome
from core.unified_evolution_engine import UnifiedEvolutionEngine


def test_basic_functionality():
    """测试基本功能"""
    
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(42)
    genome = create_initial_genome(
        gene_count=100,
        avg_gene_length=500,
        min_gene_length=200
    )
    
    print(f"📊 Test genome: {genome.gene_count} genes, {genome.size:,} bp")
    
    # 创建基础引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-3,
        hgt_rate=0.01,
        recombination_rate=1e-2,
        enable_gene_loss=False,
        enable_parallel=False,
        enable_optimization=True
    )
    
    # 运行短期进化
    print(f"🧬 Running 10 generations...")
    start_time = time.time()
    
    history = engine.evolve_multiple_generations(genome, 10, show_progress=False)
    
    end_time = time.time()
    
    print(f"✅ Evolution completed in {end_time - start_time:.3f} seconds")
    print(f"📈 Final genome: {genome.gene_count} genes, {genome.size:,} bp")
    print(f"🔬 Total mutations: {genome.total_mutations}")
    print(f"🔄 Total HGT events: {genome.total_hgt_events}")
    print(f"🧬 Total recombinations: {genome.total_recombination_events}")
    
    return len(history) == 10 and genome.total_mutations > 0


def test_gene_loss_functionality():
    """测试基因丢失功能"""
    
    print("\n🗑️  Testing Gene Loss Functionality")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(123)
    genome = create_initial_genome(
        gene_count=200,
        avg_gene_length=400,
        min_gene_length=150
    )
    
    initial_gene_count = genome.gene_count
    print(f"📊 Initial genome: {initial_gene_count} genes")
    
    # 创建带基因丢失的引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-3,
        hgt_rate=0.02,
        recombination_rate=1e-2,
        enable_gene_loss=True,
        loss_rate=0.01,  # 高丢失率以便观察
        core_gene_protection=0.8,
        min_genome_size=100,
        enable_parallel=False
    )
    
    # 运行进化
    print(f"🧬 Running 20 generations with gene loss...")
    history = engine.evolve_multiple_generations(genome, 20, show_progress=False)
    
    final_gene_count = genome.gene_count
    genes_lost = initial_gene_count - final_gene_count + genome.total_hgt_events
    
    print(f"📈 Final genome: {final_gene_count} genes")
    print(f"📉 Net gene change: {final_gene_count - initial_gene_count:+d}")
    print(f"🗑️  Estimated genes lost: {genes_lost}")
    
    # 获取基因丢失统计
    if engine.gene_loss:
        loss_stats = engine.gene_loss.get_loss_statistics(genome)
        print(f"📊 Loss statistics: {loss_stats['total_genes_lost']} total lost")
    
    return genes_lost > 0  # 应该有基因丢失


def test_parallel_functionality():
    """测试并行处理功能"""
    
    print("\n⚡ Testing Parallel Processing")
    print("=" * 50)
    
    # 创建较大的基因组以触发并行处理
    np.random.seed(456)
    genome = create_initial_genome(
        gene_count=1000,  # 超过并行阈值
        avg_gene_length=300,
        min_gene_length=100
    )
    
    print(f"📊 Large genome: {genome.gene_count} genes")
    
    # 测试串行处理
    serial_genome = genome.copy()
    serial_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-3,
        hgt_rate=0.01,
        enable_parallel=False,
        enable_optimization=True
    )
    
    print(f"🔄 Testing serial processing...")
    serial_start = time.time()
    serial_engine.evolve_multiple_generations(serial_genome, 5, show_progress=False)
    serial_time = time.time() - serial_start
    
    # 测试并行处理
    parallel_genome = genome.copy()
    parallel_engine = UnifiedEvolutionEngine(
        mutation_rate=1e-3,
        hgt_rate=0.01,
        enable_parallel=True,
        enable_optimization=True
    )
    
    print(f"⚡ Testing parallel processing...")
    parallel_start = time.time()
    parallel_engine.evolve_multiple_generations(parallel_genome, 5, show_progress=False)
    parallel_time = time.time() - parallel_start
    
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    
    print(f"📊 Performance comparison:")
    print(f"   Serial time: {serial_time:.3f} seconds")
    print(f"   Parallel time: {parallel_time:.3f} seconds")
    print(f"   Speedup: {speedup:.2f}x")
    
    # 验证结果一致性（应该相似但不完全相同，因为随机性）
    serial_mutations = serial_genome.total_mutations
    parallel_mutations = parallel_genome.total_mutations
    
    print(f"📈 Results comparison:")
    print(f"   Serial mutations: {serial_mutations}")
    print(f"   Parallel mutations: {parallel_mutations}")
    
    return speedup > 0.5  # 并行应该有一定的性能提升或至少不太慢


def test_configuration_options():
    """测试不同配置选项"""
    
    print("\n⚙️  Testing Configuration Options")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(789)
    genome = create_initial_genome(
        gene_count=300,
        avg_gene_length=400,
        min_gene_length=200
    )
    
    # 测试不同配置
    configs = {
        'minimal': {
            'name': 'Minimal Configuration',
            'params': {
                'mutation_rate': 1e-4,
                'enable_gene_loss': False,
                'enable_parallel': False,
                'enable_optimization': False
            }
        },
        'optimized': {
            'name': 'Optimized Configuration', 
            'params': {
                'mutation_rate': 1e-4,
                'enable_gene_loss': False,
                'enable_parallel': False,
                'enable_optimization': True
            }
        },
        'full_featured': {
            'name': 'Full Featured Configuration',
            'params': {
                'mutation_rate': 1e-4,
                'hgt_rate': 0.02,
                'recombination_rate': 1e-2,
                'enable_gene_loss': True,
                'loss_rate': 1e-4,
                'enable_parallel': True,
                'enable_optimization': True
            }
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        test_genome = genome.copy()
        
        print(f"🧪 Testing {config['name']}...")
        
        engine = UnifiedEvolutionEngine(**config['params'])
        
        start_time = time.time()
        engine.evolve_multiple_generations(test_genome, 10, show_progress=False)
        end_time = time.time()
        
        results[config_name] = {
            'time': end_time - start_time,
            'mutations': test_genome.total_mutations,
            'hgt_events': test_genome.total_hgt_events,
            'final_genes': test_genome.gene_count
        }
        
        print(f"   Time: {results[config_name]['time']:.3f}s, "
              f"Mutations: {results[config_name]['mutations']}, "
              f"Genes: {results[config_name]['final_genes']}")
    
    return len(results) == 3


def test_simulation_interface():
    """测试完整模拟接口"""
    
    print("\n🎯 Testing Simulation Interface")
    print("=" * 50)
    
    # 创建测试基因组
    np.random.seed(999)
    genome = create_initial_genome(
        gene_count=500,
        avg_gene_length=350,
        min_gene_length=150
    )
    
    print(f"📊 Test genome: {genome.gene_count} genes")
    
    # 创建引擎
    engine = UnifiedEvolutionEngine(
        mutation_rate=1e-3,
        hgt_rate=0.01,
        recombination_rate=1e-2,
        enable_gene_loss=True,
        loss_rate=1e-3,
        enable_parallel=True,
        enable_optimization=True
    )
    
    # 运行完整模拟
    print(f"🚀 Running complete simulation...")
    final_genome, snapshots = engine.simulate_evolution(
        initial_genome=genome,
        generations=30,
        save_snapshots=True,
        snapshot_interval=10
    )
    
    print(f"📊 Simulation results:")
    print(f"   Initial: {genome.gene_count} genes")
    print(f"   Final: {final_genome.gene_count} genes")
    print(f"   Snapshots: {len(snapshots)}")
    
    # 获取性能分析
    perf_analysis = engine.get_performance_analysis()
    print(f"📈 Performance analysis available: {'total_generations' in perf_analysis}")
    
    # 获取进化总结
    summary = engine.get_evolution_summary(final_genome)
    print(f"📋 Evolution summary available: {'genome_stats' in summary}")
    
    return len(snapshots) > 0 and final_genome.generation == 30


def run_comprehensive_test():
    """运行综合测试"""
    
    print("🧪 UNIFIED EVOLUTION ENGINE TEST SUITE")
    print("=" * 60)
    
    test_functions = [
        ("Basic Functionality", test_basic_functionality),
        ("Gene Loss", test_gene_loss_functionality), 
        ("Parallel Processing", test_parallel_functionality),
        ("Configuration Options", test_configuration_options),
        ("Simulation Interface", test_simulation_interface)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*60}")
            result = test_func()
            results.append((test_name, result, None))
            
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    # 总结测试结果
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 TEST SUITE COMPLETED")
    print(f"{'='*60}")
    print(f"⏱️  Total test time: {total_time:.2f} seconds")
    print(f"📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<25} {status}")
        if error:
            print(f"      Error: {error}")
        if result:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"🎉 All tests passed! UnifiedEvolutionEngine is working correctly.")
        print(f"\n💡 Ready to use:")
        print(f"   - Run 'python main_unified.py' for interactive simulations")
        print(f"   - Import UnifiedEvolutionEngine in your scripts")
        print(f"   - Check ENGINE_MIGRATION_GUIDE.md for migration help")
    else:
        print(f"⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


def main():
    """主测试函数"""
    
    try:
        success = run_comprehensive_test()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Test interrupted by user.")
        return 1
        
    except Exception as e:
        print(f"\n❌ Fatal test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())