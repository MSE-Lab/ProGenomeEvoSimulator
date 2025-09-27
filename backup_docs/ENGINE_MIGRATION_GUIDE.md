# 进化引擎迁移指南

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
