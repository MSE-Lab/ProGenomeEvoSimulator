#!/usr/bin/env python3
"""
快速测试持久化功能
"""

import sys
from pathlib import Path

def test_imports():
    """测试所有必要的导入"""
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__} imported successfully")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} imported successfully")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_persistent_engine():
    """测试持久化引擎"""
    try:
        from core.persistent_evolution_engine import PersistentEvolutionEngine
        print("✅ PersistentEvolutionEngine imported successfully")
        
        # 创建引擎实例
        engine = PersistentEvolutionEngine(
            base_output_dir="test_output",
            snapshot_interval=10,
            compress_data=False  # 测试时不压缩
        )
        print("✅ PersistentEvolutionEngine initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ PersistentEvolutionEngine error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_analyzer():
    """测试数据分析器"""
    try:
        from analysis.persistent_data_analyzer import PersistentDataAnalyzer
        print("✅ PersistentDataAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f"❌ PersistentDataAnalyzer error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genome_creation():
    """测试基因组创建"""
    try:
        from core.genome import create_initial_genome
        
        # 创建一个小的测试基因组
        genome = create_initial_genome(
            num_genes=100,
            avg_gene_length=900,
            gc_content=0.5
        )
        
        print(f"✅ Test genome created: {len(genome.genes)} genes, {genome.total_length} bp")
        return True, genome
    except Exception as e:
        print(f"❌ Genome creation error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """主测试函数"""
    print("🧪 快速测试持久化功能")
    print("=" * 50)
    
    # 测试导入
    print("\n1. 测试库导入...")
    if not test_imports():
        return False
    
    # 测试持久化引擎
    print("\n2. 测试持久化引擎...")
    if not test_persistent_engine():
        return False
    
    # 测试数据分析器
    print("\n3. 测试数据分析器...")
    if not test_data_analyzer():
        return False
    
    # 测试基因组创建
    print("\n4. 测试基因组创建...")
    success, genome = test_genome_creation()
    if not success:
        return False
    
    print("\n🎉 所有测试通过！持久化功能准备就绪！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)