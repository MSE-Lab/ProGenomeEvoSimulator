# Molecular Evolution Simulator - 分子进化模拟器

基于现代分子进化理论的原核生物基因组进化模拟器。该模拟器整合了选择压力、密码子偏好性、功能约束等分子进化的关键机制，提供更真实和准确的进化模拟。

## 🧬 核心特性

### 分子进化机制
- **选择压力模型**: 基于基因功能重要性的选择压力
- **密码子使用偏好性**: 模拟真实的密码子使用模式
- **同义/非同义突变**: 区分不同类型突变的选择效应
- **功能域保护**: 保护重要功能域免受有害突变
- **dN/dS比率计算**: 分子进化速率分析

### 增强的进化机制
- **智能点突变**: 转换/颠换偏好性、突变热点、序列依赖性
- **增强HGT**: 转移屏障、代谢整合、功能选择性
- **改进同源重组**: 重组热点、基因转换、功能保护

### 高级分析功能
- **ANI计算**: 平均核苷酸一致性分析
- **保守性分析**: 基因保守程度评估
- **分子进化统计**: 详细的进化事件统计
- **功能分类**: 基因功能重要性分类

## 📁 项目结构

```
simulator/
├── core/
│   ├── genome.py                           # 基因组和基因类
│   ├── evolution_engine.py                 # 基础进化引擎
│   ├── evolution_engine_optimized.py       # 优化版进化引擎
│   └── molecular_evolution_engine.py       # 分子进化引擎 ⭐
├── mechanisms/
│   ├── point_mutation.py                   # 基础点突变
│   ├── point_mutation_optimized.py         # 优化点突变
│   ├── horizontal_transfer.py              # 基础HGT
│   ├── enhanced_horizontal_transfer.py     # 增强HGT ⭐
│   ├── homologous_recombination.py         # 基础同源重组
│   └── enhanced_homologous_recombination.py # 增强同源重组 ⭐
├── analysis/
│   ├── ani_calculator.py                   # ANI计算器
│   └── conservation_analyzer.py            # 保守性分析器
├── main.py                                 # 基础演示程序
├── demo.py                                 # 简单演示程序
├── molecular_evolution_demo.py             # 分子进化演示 ⭐
└── requirements.txt                        # 依赖包
```

⭐ 表示基于分子进化理论优化的新模块

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行演示程序

#### 分子进化完整演示 (推荐)
```bash
python molecular_evolution_demo.py
```

#### 基础演示
```bash
python demo.py
```

#### 完整模拟
```bash
python main.py
```

## 🔬 分子进化特性详解

### 选择压力模型
模拟器根据基因功能重要性分类基因：
- **必需基因** (Essential): 高选择压力，强功能保护
- **重要基因** (Important): 中等选择压力
- **辅助基因** (Accessory): 低选择压力
- **可有可无基因** (Dispensable): 最低选择压力

### 密码子使用偏好性
基于真实原核生物的密码子使用模式：
- 模拟密码子使用频率差异
- 影响同义突变的选择效应
- 反映翻译效率和准确性

### 功能约束机制
- **功能域识别**: 自动识别重要功能域
- **域保护**: 降低功能域内有害突变概率
- **长度约束**: 维持基因功能所需的最小长度

### 增强HGT机制
- **转移屏障**: GC含量兼容性、序列复杂性检查
- **代谢整合**: 评估新基因的代谢网络整合可能性
- **功能选择**: 偏好有益功能基因的转移
- **来源多样性**: 模拟质粒、噬菌体、转座子等不同来源

### 改进同源重组
- **重组热点**: Chi位点等已知重组热点
- **基因转换**: 单向序列转移机制
- **不等交叉**: 可能导致基因重复或缺失
- **序列依赖性**: 基于序列相似性的重组频率

## 📊 使用示例

### 基本分子进化模拟
```python
from core.genome import create_initial_genome
from core.molecular_evolution_engine import MolecularEvolutionEngine
from analysis.ani_calculator import ANICalculator
from analysis.conservation_analyzer import ConservationAnalyzer

# 创建初始基因组
initial_genome = create_initial_genome(gene_count=2000)

# 设置分子进化引擎
engine = MolecularEvolutionEngine(
    mutation_rate=2e-9,
    enable_selection=True,
    enable_codon_bias=True,
    enable_functional_constraints=True
)

# 运行进化模拟
evolved_genome, snapshots = engine.simulate_molecular_evolution(
    initial_genome, generations=1000
)

# 分析结果
ani_calculator = ANICalculator()
ani_analysis = ani_calculator.compare_genomes_comprehensive(
    initial_genome, evolved_genome
)

conservation_analyzer = ConservationAnalyzer()
conservation_analysis = conservation_analyzer.analyze_genome_conservation(
    evolved_genome, initial_genome
)
```

### 对比不同进化模式
```python
# 无约束进化
unconstrained_engine = MolecularEvolutionEngine(
    enable_selection=False,
    enable_codon_bias=False,
    enable_functional_constraints=False
)

# 有约束进化
constrained_engine = MolecularEvolutionEngine(
    enable_selection=True,
    enable_codon_bias=True,
    enable_functional_constraints=True
)

# 比较进化结果...
```

## ⚙️ 参数配置

### 分子进化引擎参数
```python
MolecularEvolutionEngine(
    mutation_rate=2e-9,                    # 突变率 (每bp每代)
    hgt_rate=0.003,                        # HGT率 (每基因组每代)
    recombination_rate=2e-6,               # 重组率 (每bp每代)
    enable_selection=True,                 # 启用选择压力
    enable_codon_bias=True,                # 启用密码子偏好性
    enable_functional_constraints=True     # 启用功能约束
)
```

### 增强HGT参数
```python
EnhancedHorizontalGeneTransfer(
    hgt_rate=0.003,
    enable_transfer_barriers=True,         # 转移屏障
    enable_metabolic_integration=True,     # 代谢整合
    gc_content_tolerance=0.12             # GC含量容忍度
)
```

### 增强重组参数
```python
EnhancedHomologousRecombination(
    recombination_rate=2e-6,
    min_similarity=0.75,                   # 最小相似性
    enable_recombination_hotspots=True,    # 重组热点
    enable_gene_conversion=True,           # 基因转换
    enable_functional_protection=True      # 功能保护
)
```

## 📈 输出结果

### 1. 控制台输出
- 进化进度实时显示
- ANI分析结果
- 保守性分析统计
- 分子进化详细统计
- 基因组组成变化

### 2. 分子进化统计
- 同义/非同义突变比例
- dN/dS比率估算
- 选择压力效应统计
- 功能域保护效果
- 密码子偏好性影响

### 3. 增强机制分析
- HGT功能分布和来源分析
- 重组类型和热点统计
- 转移屏障和整合成功率
- 保守性分类和分布

## 🔬 分子进化理论基础

### 中性理论 vs 选择理论
- 模拟中性突变和选择性突变
- 区分同义和非同义位点的进化速率
- 反映功能约束对进化的影响

### 分子钟假说
- 基于突变率的进化时间估算
- 考虑选择压力对分子钟的影响
- 不同基因类型的进化速率差异

### 水平基因转移理论
- 模拟HGT对基因组进化的贡献
- 考虑转移屏障和选择压力
- 反映代谢网络整合的复杂性

## 🎯 应用场景

### 科研应用
- **比较基因组学**: 研究基因组进化模式
- **分子进化分析**: 计算dN/dS比率、进化速率
- **功能基因组学**: 分析基因功能重要性
- **系统发育研究**: 构建进化树和时间估算

### 教学应用
- **分子进化教学**: 演示进化机制和理论
- **生物信息学实践**: 基因组分析方法学习
- **计算生物学**: 进化模拟算法理解

### 方法开发
- **算法验证**: 测试新的进化分析方法
- **参数优化**: 评估不同参数对结果的影响
- **模型比较**: 对比不同进化模型的效果

## 📚 理论参考

### 关键概念
- **dN/dS比率**: 非同义/同义替换率比值
- **分子钟**: 基于突变率的进化时间估算
- **中性进化**: 不受选择压力影响的进化
- **正选择**: 有利突变的固定
- **负选择**: 有害突变的清除

### 重要参数
- **突变率**: 典型值 1e-9 到 1e-8 每bp每代
- **HGT率**: 典型值 0.001 到 0.01 每基因组每代
- **重组率**: 典型值 1e-6 到 1e-5 每bp每代
- **选择系数**: 范围 -1.0 到 +1.0

## 🔧 扩展功能

### 计划中的功能
- **群体遗传学模型**: 多个体进化模拟
- **环境选择压力**: 动态环境适应
- **基因调控网络**: 调控元件进化
- **结构变异**: 大片段重排和缺失
- **表观遗传修饰**: DNA甲基化等

### 自定义扩展
- 添加新的进化机制
- 自定义选择压力模型
- 扩展功能分类系统
- 集成外部数据库

## 📝 注意事项

1. **参数设置**: 根据研究目标调整进化参数
2. **计算资源**: 大规模模拟需要较多计算时间
3. **内存使用**: 内存使用与基因组大小和代数成正比
4. **结果解释**: 理解分子进化理论有助于正确解释结果
5. **验证**: 建议与真实数据对比验证模拟结果

## 🚀 性能优化

- **批处理**: 向量化操作提高计算效率
- **缓存机制**: 减少重复计算
- **内存管理**: 优化大规模模拟的内存使用
- **并行计算**: 支持多核并行处理 (计划中)

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进模拟器功能！

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**Molecular Evolution Simulator** - 让分子进化理论在计算机中重现！ 🧬✨