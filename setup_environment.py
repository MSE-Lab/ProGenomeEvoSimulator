#!/usr/bin/env python3
"""
Environment Setup Script - 环境配置脚本
自动检查和配置ProGenomeEvoSimulator的运行环境

Version: 1.0.0
Author: ProGenomeEvoSimulator Team
Date: 2025-09-27
"""

import sys
import subprocess
import importlib
import warnings
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Minimum required: Python 3.8")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
    return True


def check_package_version(package_name, min_version=None, max_version=None):
    """检查包版本"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if version == 'unknown':
            print(f"⚠️  {package_name}: version unknown")
            return True
        
        print(f"📦 {package_name}: {version}", end="")
        
        # 检查版本范围
        if min_version or max_version:
            from packaging import version as pkg_version
            current_ver = pkg_version.parse(version)
            
            if min_version and current_ver < pkg_version.parse(min_version):
                print(f" ❌ (requires >= {min_version})")
                return False
            
            if max_version and current_ver >= pkg_version.parse(max_version):
                print(f" ⚠️  (recommends < {max_version})")
                return "warning"
        
        print(" ✅")
        return True
        
    except ImportError:
        print(f"❌ {package_name}: not installed")
        return False


def check_dependencies():
    """检查所有依赖"""
    print("\n📋 Checking dependencies...")
    
    dependencies = {
        'numpy': {'min': '1.21.0', 'max': '1.23.0'},
        'scipy': {'min': '1.7.0', 'max': '1.10.0'},
        'matplotlib': {'min': '3.5.0'},
        'pandas': {'min': '1.3.0'},
        'seaborn': {'min': '0.11.0'},
    }
    
    results = {}
    for package, constraints in dependencies.items():
        result = check_package_version(
            package, 
            constraints.get('min'), 
            constraints.get('max')
        )
        results[package] = result
    
    return results


def check_numpy_scipy_compatibility():
    """专门检查NumPy和SciPy的兼容性"""
    print("\n🔍 Checking NumPy-SciPy compatibility...")
    
    try:
        import numpy as np
        import scipy
        
        numpy_version = np.__version__
        scipy_version = scipy.__version__
        
        print(f"NumPy version: {numpy_version}")
        print(f"SciPy version: {scipy_version}")
        
        # 捕获SciPy的警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import scipy.stats  # 触发可能的警告
            
            if w:
                for warning in w:
                    if "NumPy version" in str(warning.message):
                        print(f"⚠️  Compatibility warning: {warning.message}")
                        return False
        
        print("✅ NumPy-SciPy compatibility OK")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def install_compatible_versions():
    """安装兼容版本"""
    print("\n🔧 Installing compatible versions...")
    
    compatible_packages = [
        "numpy>=1.21.0,<1.23.0",
        "scipy>=1.7.0,<1.10.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0"
    ]
    
    try:
        for package in compatible_packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        print("✅ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


def test_imports():
    """测试关键模块导入"""
    print("\n🧪 Testing module imports...")
    
    test_modules = [
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn'
    ]
    
    success = True
    for module in test_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            success = False
    
    return success


def test_simulator_imports():
    """测试模拟器模块导入"""
    print("\n🧪 Testing simulator imports...")
    
    try:
        # 添加当前目录到路径
        sys.path.insert(0, str(Path.cwd()))
        
        from core.persistent_evolution_engine import PersistentEvolutionEngine
        print("✅ PersistentEvolutionEngine")
        
        from analysis.persistent_data_analyzer import PersistentDataAnalyzer
        print("✅ PersistentDataAnalyzer")
        
        from main_persistent import create_test_configurations
        print("✅ main_persistent")
        
        return True
        
    except ImportError as e:
        print(f"❌ Simulator import failed: {e}")
        return False


def main():
    """主函数"""
    print("🔧 PROGENOME EVOLUTION SIMULATOR - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Environment setup failed: Incompatible Python version")
        return False
    
    # 检查依赖
    dep_results = check_dependencies()
    
    # 检查NumPy-SciPy兼容性
    numpy_scipy_ok = check_numpy_scipy_compatibility()
    
    # 判断是否需要重新安装
    need_reinstall = False
    
    if not numpy_scipy_ok:
        need_reinstall = True
        print("\n⚠️  NumPy-SciPy compatibility issues detected")
    
    for package, result in dep_results.items():
        if result is False:
            need_reinstall = True
            print(f"⚠️  {package} needs to be installed/updated")
        elif result == "warning":
            print(f"⚠️  {package} version may cause compatibility issues")
    
    # 询问是否安装兼容版本
    if need_reinstall:
        print("\n🤔 Would you like to install compatible versions? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                if install_compatible_versions():
                    print("\n🔄 Re-checking after installation...")
                    check_numpy_scipy_compatibility()
                else:
                    print("\n❌ Installation failed")
                    return False
            else:
                print("\n⚠️  Proceeding with current versions (may have warnings)")
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup cancelled by user")
            return False
    
    # 测试导入
    if not test_imports():
        print("\n❌ Basic imports failed")
        return False
    
    if not test_simulator_imports():
        print("\n❌ Simulator imports failed")
        return False
    
    print("\n🎉 ENVIRONMENT SETUP COMPLETED!")
    print("=" * 60)
    print("✅ All dependencies are properly configured")
    print("✅ Simulator modules can be imported successfully")
    print("\n🚀 You can now run:")
    print("   python main_persistent.py --config fast_test")
    print("   python demo_persistent.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)