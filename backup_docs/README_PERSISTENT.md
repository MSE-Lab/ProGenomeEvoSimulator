# 持久化进化模拟器 - 使用指南

## 🗄️ 概述

持久化进化模拟器是原核生物基因组进化模拟器的增强版本，提供完整的数据持久化功能。所有模拟数据都会自动保存到硬盘，支持中断恢复、历史数据分析和深度可视化。

## 🚀 快速开始

### 运行基本模拟

```bash
# 快速测试（200代，1500基因）
python main_persistent.py --config fast_test

# 真实参数模拟（1000代，3000基因）
python main_persistent.py --config realistic

# 大规模模拟（5000代，5000基因）
python main_persistent.py --config large_scale

# 详细分析配置（保存所有数据）
python main_persistent.py --config detailed_analysis
```

### 运行完整演示

```bash
# 运行所有功能的综合演示
python demo_persistent.py
```

### 分析现有结果

```bash
# 分析指定运行的结果
python main_persistent.py --analyze-only simulation_results/run_20250927_140000
```

## 📁 数据存储结构

每次运行都会创建一个独特的目录结构：

```
simulation_results/
└── run_20250927_140000/          # 运行ID（时间戳）
    ├── metadata/                  # 元数据
    │   ├── config.json           # 模拟配置参数
    │   ├── run_info.json         # 运行信息
    │   ├── initial_genome.json   # 初始基因组
    │   └── evolved_genome.json   # 最终基因组
    ├── snapshots/                 # 基因组快照
    │   ├── generation_000000.json
    │   ├── generation_000100.json
    │   └── ...
    ├── events/                    # 进化事件日志
    │   ├── mutations.jsonl       # 突变事件
    │   ├── hgt_events.jsonl      # HGT事件
    │   ├── recombination.jsonl   # 重组事件
    │   └── gene_loss.jsonl       # 基因丢失事件
    ├── statistics/                # 统计数据
    │   ├── genome_stats.csv      # 基因组统计时间序列
    │   ├── evolution_stats.csv   # 进化事件统计
    │   └── performance_stats.csv # 性能统计
    ├── analysis/                  # 分析结果
    │   ├── conservation_analysis.json  # 保守性分析
    │   ├── ani_identities.json        # ANI身份数据
    │   ├── final_summary.json         # 最终摘要
    │   └── comprehensive_analysis_report.txt  # 综合报告
    └── visualizations/            # 可视化图表
        ├── genome_evolution_timeline.png
        ├── evolution_events_analysis.png
        └── ...
```

## 🔧 配置选项

### 预设配置

| 配置名称 | 描述 | 代数 | 基因数 | 适用场景 |
|---------|------|------|--------|----------|
| `fast_test` | 快速测试 | 200 | 1500 | 功能验证 |
| `realistic` | 真实参数 | 1000 | 3000 | 科研使用 |
| `large_scale` | 大规模模拟 | 5000 | 5000 | 长期进化研究 |
| `detailed_analysis` | 详细分析 | 500 | 2000 | 深度数据分析 |

### 自定义配置

```python
custom_config = {
    'description': '自定义配置',
    'generations': 800,
    'initial_genes': 2500,
    'snapshot_interval': 50,
    'engine_config': {
        'mutation_rate': 1e-6,
        'hgt_rate': 1e-5,
        'recombination_rate': 1e-7,
        # ... 其他进化参数
    },
    'storage_config': {
        'compress_data': True,
        'save_detailed_events': True,
        'save_sequences': True,
        'stats_flush_interval': 10
    }
}

run_directory = run_persistent_simulation(custom_config=custom_config)
```

## 📊 数据分析

### 使用分析器类

```python
from analysis.persistent_data_analyzer import PersistentDataAnalyzer

# 创建分析器
analyzer = PersistentDataAnalyzer('simulation_results/run_20250927_140000')

# 加载数据
genome_stats = analyzer.load_genome_stats()
evolution_stats = analyzer.load_evolution_stats()
snapshots = analyzer.load_snapshots()

# 进化分析
genome_analysis = analyzer.analyze_genome_evolution()
comparison = analyzer.compare_initial_vs_final_genome()

# 生成可视化
analyzer.plot_genome_evolution_timeline()
analyzer.plot_evolution_events_analysis()

# 生成报告
analyzer.generate_comprehensive_report()
```

### 便捷分析函数

```python
from analysis.persistent_data_analyzer import analyze_run

# 一键分析（包含图表和报告）
analyzer = analyze_run('simulation_results/run_20250927_140000')
```

## 🎯 主要功能特性

### 1. 完整数据持久化
- ✅ 自动保存所有模拟数据到硬盘
- ✅ 支持数据压缩节省存储空间
- ✅ 定期快照和统计数据刷新
- ✅ 详细的进化事件日志

### 2. 高级数据分析
- ✅ 时间序列分析和可视化
- ✅ 基因组进化趋势分析
- ✅ 进化事件统计分析
- ✅ ANI身份数据集分析
- ✅ 保守性分析集成

### 3. 丰富的可视化
- ✅ 基因组进化时间线图
- ✅ 进化事件分析图表
- ✅ 统计数据热图
- ✅ 自动保存高质量图片

### 4. 综合报告生成
- ✅ 自动生成分析报告
- ✅ 数据完整性验证
- ✅ 性能统计分析
- ✅ 导出功能支持

## 🔬 科研应用

### 适用研究领域
- **比较基因组学**: 分析基因组结构变化
- **分子进化**: 研究突变和选择压力
- **水平基因转移**: 分析HGT模式和影响
- **基因组稳定性**: 研究基因丢失和保留机制

### 数据输出格式
- **JSON**: 结构化数据，易于程序处理
- **CSV**: 统计数据，适合Excel和R分析
- **JSONL**: 事件流数据，支持流式处理
- **PNG**: 高质量可视化图表

## ⚡ 性能优化

### 存储优化
```python
# 大规模模拟的存储优化配置
storage_config = {
    'compress_data': True,          # 启用压缩
    'save_detailed_events': False,  # 关闭详细事件日志
    'save_sequences': False,        # 不保存完整序列
    'stats_flush_interval': 50      # 减少I/O频率
}
```

### 内存管理
- 定期刷新统计数据缓存
- 压缩存储大型数据结构
- 选择性保存快照数据
- 自动清理临时文件

## 🛠️ 高级用法

### 自定义分析流程

```python
from core.persistent_evolution_engine import PersistentEvolutionEngine
from analysis.persistent_data_analyzer import PersistentDataAnalyzer

# 1. 运行模拟
engine = PersistentEvolutionEngine(
    base_output_dir="my_research",
    snapshot_interval=50,
    compress_data=True
)

final_genome, snapshots = engine.simulate_evolution(
    initial_genome, generations=1000
)

# 2. 自定义分析
analyzer = PersistentDataAnalyzer(engine.get_run_directory())

# 加载特定代数范围的快照
snapshots = analyzer.load_snapshots(generation_range=(100, 500))

# 分析特定类型的进化事件
hgt_events = analyzer.load_evolution_events('hgt_events')

# 自定义可视化
import matplotlib.pyplot as plt
genome_stats = analyzer.load_genome_stats()
plt.plot(genome_stats['generation'], genome_stats['total_size'])
plt.title('Custom Genome Size Analysis')
plt.show()
```

### 批量分析多个运行

```python
import glob
from pathlib import Path

# 分析所有运行
run_dirs = glob.glob('simulation_results/run_*')

for run_dir in run_dirs:
    print(f"Analyzing {run_dir}...")
    analyzer = PersistentDataAnalyzer(run_dir)
    
    # 生成标准化报告
    analyzer.generate_comprehensive_report()
    
    # 导出摘要数据
    summary = analyzer.export_data_summary()
    
    # 保存到汇总文件
    summary_file = Path(run_dir) / "batch_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 使用压缩存储: `compress_data=True`
   - 关闭详细事件日志: `save_detailed_events=False`
   - 增加统计刷新间隔: `stats_flush_interval=50`

2. **磁盘空间不足**
   - 不保存完整序列: `save_sequences=False`
   - 增加快照间隔: `snapshot_interval=200`
   - 定期清理旧的运行数据

3. **分析失败**
   - 检查数据完整性: `analyzer.validate_data_integrity()`
   - 确认文件权限和路径
   - 查看错误日志和异常信息

### 数据恢复

```python
# 检查数据完整性
analyzer = PersistentDataAnalyzer(run_directory)
analyzer.validate_data_integrity()

# 手动加载损坏的数据
try:
    genome_stats = analyzer.load_genome_stats()
except Exception as e:
    print(f"Failed to load genome stats: {e}")
    # 尝试从备份或快照恢复
```

## 📚 API参考

### PersistentEvolutionEngine

主要方法：
- `initialize_storage(config, initial_genome)`: 初始化存储系统
- `save_snapshot(genome, generation)`: 保存基因组快照
- `log_evolution_event(event_type, event_data)`: 记录进化事件
- `save_generation_stats(genome, stats)`: 保存代数统计
- `save_ani_identities(identities_data)`: 保存ANI数据
- `get_run_directory()`: 获取运行目录路径

### PersistentDataAnalyzer

主要方法：
- `load_*()`: 各种数据加载方法
- `analyze_genome_evolution()`: 基因组进化分析
- `compare_initial_vs_final_genome()`: 基因组比较
- `plot_*()`: 各种可视化方法
- `generate_comprehensive_report()`: 生成综合报告
- `export_data_summary()`: 导出数据摘要

## 🤝 贡献指南

欢迎贡献代码和改进建议！

1. Fork项目仓库
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📞 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查GitHub Issues
3. 创建新的Issue描述问题

---

**ProGenomeEvoSimulator Team**  
*Version 1.0.0 - 2025-09-27*