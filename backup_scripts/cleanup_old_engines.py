#!/usr/bin/env python3
"""
清理旧版本进化引擎的脚本
整理core目录，保留统一的进化引擎
"""

import os
import shutil
from pathlib import Path


def backup_old_engines():
    """备份旧的进化引擎到backup目录"""
    
    # 创建备份目录
    backup_dir = Path("backup_engines")
    backup_dir.mkdir(exist_ok=True)
    
    # 需要备份的旧引擎文件
    old_engines = [
        "core/evolution_engine.py",
        "core/evolution_engine_optimized.py", 
        "core/evolution_engine_with_conservation.py",
        "core/evolution_engine_with_gene_loss.py",
        "core/molecular_evolution_engine.py",
        "core/parallel_evolution_engine.py"
    ]
    
    backed_up = []
    
    for engine_file in old_engines:
        if os.path.exists(engine_file):
            try:
                # 复制到备份目录
                backup_path = backup_dir / Path(engine_file).name
                shutil.copy2(engine_file, backup_path)
                backed_up.append(engine_file)
                print(f"✅ Backed up: {engine_file} → {backup_path}")
            except Exception as e:
                print(f"❌ Failed to backup {engine_file}: {e}")
    
    return backed_up


def remove_old_engines(backed_up_files):
    """删除已备份的旧引擎文件"""
    
    removed = []
    
    for engine_file in backed_up_files:
        try:
            os.remove(engine_file)
            removed.append(engine_file)
            print(f"🗑️  Removed: {engine_file}")
        except Exception as e:
            print(f"❌ Failed to remove {engine_file}: {e}")
    
    return removed


def update_imports():
    """更新可能的导入引用"""
    
    # 需要检查的文件
    files_to_check = [
        "main.py",
        "main_parallel.py", 
        "main_with_gene_loss.py",
        "demo_parallel.py",
        "demo_gene_loss.py",
        "test_parallel.py",
        "test_gene_loss.py"
    ]
    
    print(f"\n📝 Checking import statements in existing files...")
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否包含旧的导入
                old_imports = [
                    "from core.evolution_engine import",
                    "from core.evolution_engine_optimized import", 
                    "from core.evolution_engine_with_conservation import",
                    "from core.evolution_engine_with_gene_loss import",
                    "from core.molecular_evolution_engine import",
                    "from core.parallel_evolution_engine import"
                ]
                
                has_old_imports = any(old_import in content for old_import in old_imports)
                
                if has_old_imports:
                    print(f"⚠️  {file_path} contains old import statements")
                    print(f"   Recommend updating to: from core.unified_evolution_engine import UnifiedEvolutionEngine")
                else:
                    print(f"✅ {file_path} - no old imports found")
                    
            except Exception as e:
                print(f"❌ Error checking {file_path}: {e}")


def create_migration_guide():
    """创建迁移指南"""
    
    guide_content = """# 进化引擎迁移指南

## 概述
ProGenomeEvoSimulator已经整合所有进化引擎功能到统一的`UnifiedEvolutionEngine`中。

## 旧版本 → 新版本映射

### 1. 基础进化引擎
```python
# 旧版本
from core.evolution_engine import EvolutionEngine
engine = EvolutionEngine(mutation_rate=1e-6)

# 新版本
from core.unified_evolution_engine import UnifiedEvolutionEngine
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-6,
    enable_optimization=False,  # 关闭优化以匹配旧行为
    enable_parallel=False,      # 关闭并行以匹配旧行为
    enable_gene_loss=False      # 关闭基因丢失以匹配旧行为
)
```

### 2. 优化进化引擎
```python
# 旧版本
from core.evolution_engine_optimized import OptimizedEvolutionEngine
engine = OptimizedEvolutionEngine(mutation_rate=1e-5)

# 新版本
from core.unified_evolution_engine import UnifiedEvolutionEngine
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,
    enable_optimization=True,   # 启用优化
    enable_parallel=False,      # 可选择启用
    enable_gene_loss=False      # 可选择启用
)
```

### 3. 并行进化引擎
```python
# 旧版本
from core.parallel_evolution_engine import ParallelEvolutionEngine
engine = ParallelEvolutionEngine(mutation_rate=1e-5, num_processes=4)

# 新版本
from core.unified_evolution_engine import UnifiedEvolutionEngine
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,
    enable_optimization=True,   # 推荐启用
    enable_parallel=True,       # 启用并行
    num_processes=4,           # 指定进程数
    enable_gene_loss=False      # 可选择启用
)
```

### 4. 基因丢失进化引擎
```python
# 旧版本
from core.evolution_engine_with_gene_loss import EvolutionEngineWithGeneLoss
engine = EvolutionEngineWithGeneLoss(
    mutation_rate=1e-5,
    enable_gene_loss=True,
    loss_rate=1e-6
)

# 新版本
from core.unified_evolution_engine import UnifiedEvolutionEngine
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,
    enable_optimization=True,   # 推荐启用
    enable_parallel=True,       # 推荐启用
    enable_gene_loss=True,      # 启用基因丢失
    loss_rate=1e-6
)
```

## 推荐的新配置

### 快速测试
```python
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-4,
    hgt_rate=0.05,
    recombination_rate=1e-2,
    enable_gene_loss=True,
    loss_rate=1e-4,
    enable_parallel=True,
    enable_optimization=True
)
```

### 真实模拟
```python
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-6,
    hgt_rate=1e-5,
    recombination_rate=1e-6,
    enable_gene_loss=True,
    loss_rate=1e-6,
    enable_parallel=True,
    enable_optimization=True
)
```

## 主要优势

1. **统一接口**: 一个引擎包含所有功能
2. **自动优化**: 智能选择最佳处理模式
3. **完整功能**: 集成所有进化机制
4. **易于使用**: 简化的参数配置
5. **向后兼容**: 支持所有旧功能

## 迁移步骤

1. 更新导入语句
2. 调整参数配置
3. 测试功能正确性
4. 享受性能提升！

## 获取帮助

- 运行 `python main_unified.py` 查看交互式界面
- 查看 `test_unified_engine.py` 了解使用示例
- 参考 `PARALLEL_OPTIMIZATION_GUIDE.md` 了解性能优化
"""

    with open("ENGINE_MIGRATION_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 Created migration guide: ENGINE_MIGRATION_GUIDE.md")


def main():
    """主清理函数"""
    
    print("🧹 EVOLUTION ENGINE CLEANUP UTILITY")
    print("=" * 50)
    print("This script will:")
    print("1. Backup old evolution engines")
    print("2. Remove old engine files") 
    print("3. Check for import references")
    print("4. Create migration guide")
    print("=" * 50)
    
    # 确认操作
    confirm = input("Proceed with cleanup? (y/n): ").strip().lower()
    
    if not confirm.startswith('y'):
        print("❌ Cleanup cancelled.")
        return
    
    try:
        # 1. 备份旧引擎
        print(f"\n📦 Step 1: Backing up old engines...")
        backed_up = backup_old_engines()
        
        if backed_up:
            print(f"✅ Successfully backed up {len(backed_up)} files")
            
            # 2. 删除旧文件
            print(f"\n🗑️  Step 2: Removing old engine files...")
            removed = remove_old_engines(backed_up)
            print(f"✅ Successfully removed {len(removed)} files")
        else:
            print("ℹ️  No old engine files found to backup")
        
        # 3. 检查导入
        print(f"\n🔍 Step 3: Checking import statements...")
        update_imports()
        
        # 4. 创建迁移指南
        print(f"\n📖 Step 4: Creating migration guide...")
        create_migration_guide()
        
        print(f"\n🎉 CLEANUP COMPLETED!")
        print(f"=" * 50)
        print(f"✅ Old engines backed up to: backup_engines/")
        print(f"✅ Unified engine available: core/unified_evolution_engine.py")
        print(f"✅ Migration guide created: ENGINE_MIGRATION_GUIDE.md")
        print(f"✅ Main program available: main_unified.py")
        print(f"\n💡 Next steps:")
        print(f"   1. Run 'python test_unified_engine.py' to test the new engine")
        print(f"   2. Run 'python main_unified.py' for interactive simulations")
        print(f"   3. Update any custom scripts using the migration guide")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()