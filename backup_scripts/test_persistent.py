#!/usr/bin/env python3
"""
简单的持久化功能测试脚本
"""

import sys
from pathlib import Path

def test_imports():
    """测试导入"""
    print("🧪 Testing imports...")
    
    try:
        from core.persistent_evolution_engine import PersistentEvolutionEngine
        print("✅ PersistentEvolutionEngine imported")
    except Exception as e:
        print(f"❌ Failed to import PersistentEvolutionEngine: {e}")
        return False
    
    try:
        from analysis.persistent_data_analyzer import PersistentDataAnalyzer
        print("✅ PersistentDataAnalyzer imported")
    except Exception as e:
        print(f"❌ Failed to import PersistentDataAnalyzer: {e}")
        return False
    
    try:
        from main_persistent import create_test_configurations
        configs = create_test_configurations()
        print(f"✅ Test configurations loaded: {list(configs.keys())}")
    except Exception as e:
        print(f"❌ Failed to load configurations: {e}")
        return False
    
    return True

def test_engine_initialization():
    """测试引擎初始化"""
    print("\n🧪 Testing engine initialization...")
    
    try:
        from core.persistent_evolution_engine import PersistentEvolutionEngine
        
        engine = PersistentEvolutionEngine(
            base_output_dir='test_output',
            snapshot_interval=10,
            compress_data=True,
            save_detailed_events=True
        )
        print("✅ PersistentEvolutionEngine initialized successfully")
        print(f"   Output directory: {engine.run_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_creation():
    """测试分析器创建"""
    print("\n🧪 Testing analyzer creation...")
    
    try:
        from analysis.persistent_data_analyzer import PersistentDataAnalyzer
        
        # 创建一个测试目录结构
        test_dir = Path("test_run_dir")
        test_dir.mkdir(exist_ok=True)
        
        # 创建必要的子目录
        (test_dir / "metadata").mkdir(exist_ok=True)
        (test_dir / "snapshots").mkdir(exist_ok=True)
        (test_dir / "statistics").mkdir(exist_ok=True)
        (test_dir / "analysis").mkdir(exist_ok=True)
        
        # 创建基本的配置文件
        import json
        config = {"test": True}
        with open(test_dir / "metadata" / "config.json", 'w') as f:
            json.dump(config, f)
        
        run_info = {"run_id": "test_run", "start_time": "2025-09-27"}
        with open(test_dir / "metadata" / "run_info.json", 'w') as f:
            json.dump(run_info, f)
        
        # 测试分析器
        analyzer = PersistentDataAnalyzer(str(test_dir))
        print("✅ PersistentDataAnalyzer created successfully")
        
        # 测试基本功能
        config_loaded = analyzer.load_config()
        run_info_loaded = analyzer.load_run_info()
        
        print(f"   Config loaded: {config_loaded}")
        print(f"   Run info loaded: {run_info_loaded}")
        
        # 清理测试目录
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🗄️  PERSISTENT FUNCTIONALITY TEST")
    print("=" * 50)
    
    all_passed = True
    
    # 测试导入
    if not test_imports():
        all_passed = False
    
    # 测试引擎初始化
    if not test_engine_initialization():
        all_passed = False
    
    # 测试分析器创建
    if not test_analyzer_creation():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Persistent functionality is ready to use")
        print("\nNext steps:")
        print("1. Run: python main_persistent.py --config fast_test")
        print("2. Or run: python demo_persistent.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)