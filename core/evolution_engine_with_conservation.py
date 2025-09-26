#!/usr/bin/env python3
"""
Enhanced Evolution Engine with Conservation Analysis
集成保守基因分析功能的进化引擎
"""

import time
import copy
from typing import Dict, List, Optional, Any
from core.evolution_engine_optimized import OptimizedEvolutionEngine
from core.genome import Genome
from analysis.conservation_analyzer import ConservationAnalyzer

class EvolutionEngineWithConservation(OptimizedEvolutionEngine):
    """集成保守基因分析的进化引擎"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-6,
                 hgt_rate: float = 1e-8,
                 recombination_rate: float = 1e-9,
                 conservation_threshold: float = 0.3,
                 enable_conservation_analysis: bool = True):
        """
        初始化集成保守基因分析的进化引擎
        
        Args:
            conservation_threshold: 保守基因的最低一致性阈值
            enable_conservation_analysis: 是否启用保守基因分析
        """
        super().__init__(mutation_rate, hgt_rate, recombination_rate)
        
        self.enable_conservation_analysis = enable_conservation_analysis
        self.conservation_analyzer = ConservationAnalyzer(
            conservation_threshold=conservation_threshold
        ) if enable_conservation_analysis else None
        
        # 存储初始基因组用于保守性分析
        self.initial_genome: Optional[Genome] = None
        self.conservation_history: List[Dict[str, Any]] = []
    
    def evolve(self, genome: Genome, generations: int, 
               analyze_conservation_every: int = 100) -> Genome:
        """
        进化基因组并定期进行保守基因分析
        
        Args:
            genome: 初始基因组
            generations: 进化代数
            analyze_conservation_every: 每隔多少代进行一次保守基因分析
        """
        print(f"🧬 Starting evolution with conservation analysis...")
        print(f"   Generations: {generations}")
        print(f"   Conservation analysis every: {analyze_conservation_every} generations")
        print(f"   Conservation threshold: {self.conservation_analyzer.conservation_threshold if self.conservation_analyzer else 'N/A'}")
        
        # 保存初始基因组
        if self.initial_genome is None:
            self.initial_genome = copy.deepcopy(genome)
        
        evolved_genome = genome
        
        for generation in range(generations):
            # 执行一代进化
            evolved_genome = super().evolve_single_generation(evolved_genome)
            
            # 定期进行保守基因分析
            if (self.enable_conservation_analysis and 
                self.conservation_analyzer and 
                (generation + 1) % analyze_conservation_every == 0):
                
                print(f"\n📊 Performing conservation analysis at generation {generation + 1}...")
                conservation_result = self.conservation_analyzer.analyze_genome_conservation(
                    evolved_genome, self.initial_genome
                )
                
                # 添加时间戳和代数信息
                conservation_result['generation'] = generation + 1
                conservation_result['timestamp'] = time.time()
                
                # 存储分析结果
                self.conservation_history.append(conservation_result)
                
                # 打印简要摘要
                self._print_brief_conservation_summary(conservation_result)
        
        # 最终保守基因分析
        if self.enable_conservation_analysis and self.conservation_analyzer:
            print(f"\n🔬 Final conservation analysis...")
            final_conservation = self.conservation_analyzer.analyze_genome_conservation(
                evolved_genome, self.initial_genome
            )
            final_conservation['generation'] = generations
            final_conservation['timestamp'] = time.time()
            self.conservation_history.append(final_conservation)
            
            # 打印详细摘要
            self.conservation_analyzer.print_conservation_summary(final_conservation)
        
        return evolved_genome
    
    def _print_brief_conservation_summary(self, conservation_result: Dict[str, Any]):
        """打印简要的保守基因分析摘要"""
        total = conservation_result['total_genes']
        conservative = conservation_result['conservative_genes']
        ratio = conservation_result['conservative_ratio']
        
        print(f"   📈 Conservation: {conservative}/{total} genes ({ratio:.3f} = {ratio*100:.1f}%)")
        
        # 显示保守程度分类
        categories = conservation_result['conservation_categories']
        highly_conserved = categories.get('highly_conserved', 0)
        non_conserved = categories.get('non_conserved', 0)
        
        print(f"   🎯 Highly conserved: {highly_conserved}, Non-conserved: {non_conserved}")
    
    def get_conservation_trend(self) -> Dict[str, List]:
        """获取保守基因比例的变化趋势"""
        if not self.conservation_history:
            return {'generations': [], 'conservation_ratios': [], 'total_genes': []}
        
        generations = [result['generation'] for result in self.conservation_history]
        conservation_ratios = [result['conservative_ratio'] for result in self.conservation_history]
        total_genes = [result['total_genes'] for result in self.conservation_history]
        
        return {
            'generations': generations,
            'conservation_ratios': conservation_ratios,
            'total_genes': total_genes
        }
    
    def get_latest_conservation_analysis(self) -> Optional[Dict[str, Any]]:
        """获取最新的保守基因分析结果"""
        return self.conservation_history[-1] if self.conservation_history else None
    
    def analyze_conservation_dynamics(self) -> Dict[str, Any]:
        """分析保守基因动态变化"""
        if len(self.conservation_history) < 2:
            return {'error': 'Insufficient data for dynamics analysis'}
        
        trend = self.get_conservation_trend()
        ratios = trend['conservation_ratios']
        
        # 计算变化趋势
        initial_ratio = ratios[0]
        final_ratio = ratios[-1]
        ratio_change = final_ratio - initial_ratio
        
        # 计算平均变化率
        generations = trend['generations']
        if len(generations) > 1:
            avg_change_rate = ratio_change / (generations[-1] - generations[0])
        else:
            avg_change_rate = 0
        
        # 分析变化模式
        increasing_periods = 0
        decreasing_periods = 0
        stable_periods = 0
        
        for i in range(1, len(ratios)):
            diff = ratios[i] - ratios[i-1]
            if abs(diff) < 0.001:  # 稳定阈值
                stable_periods += 1
            elif diff > 0:
                increasing_periods += 1
            else:
                decreasing_periods += 1
        
        return {
            'initial_conservation_ratio': initial_ratio,
            'final_conservation_ratio': final_ratio,
            'total_change': ratio_change,
            'average_change_rate': avg_change_rate,
            'change_pattern': {
                'increasing_periods': increasing_periods,
                'decreasing_periods': decreasing_periods,
                'stable_periods': stable_periods
            },
            'trend_direction': 'increasing' if ratio_change > 0.01 else 'decreasing' if ratio_change < -0.01 else 'stable'
        }
    
    def export_conservation_history(self, filename: str):
        """导出保守基因分析历史到文件"""
        import json
        
        # 准备导出数据（移除不能序列化的对象）
        export_data = []
        for result in self.conservation_history:
            export_result = {
                'generation': result['generation'],
                'timestamp': result['timestamp'],
                'total_genes': result['total_genes'],
                'conservative_genes': result['conservative_genes'],
                'conservative_ratio': result['conservative_ratio'],
                'conservation_categories': result['conservation_categories'],
                'structural_analysis': result['structural_analysis'],
                'mechanism_impact': result['mechanism_impact']
            }
            export_data.append(export_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Conservation history exported to: {filename}")
    
    def print_evolution_summary(self, evolved_genome: Genome):
        """打印进化总结，包括保守基因分析"""
        super().print_evolution_summary(evolved_genome)
        
        if self.conservation_history:
            print("\n" + "=" * 60)
            print("🧬 CONSERVATION DYNAMICS SUMMARY")
            print("=" * 60)
            
            dynamics = self.analyze_conservation_dynamics()
            if 'error' not in dynamics:
                print(f"📈 Conservation Trend:")
                print(f"   Initial ratio: {dynamics['initial_conservation_ratio']:.3f}")
                print(f"   Final ratio: {dynamics['final_conservation_ratio']:.3f}")
                print(f"   Total change: {dynamics['total_change']:+.3f}")
                print(f"   Average change rate: {dynamics['average_change_rate']:+.6f} per generation")
                print(f"   Trend direction: {dynamics['trend_direction']}")
                
                pattern = dynamics['change_pattern']
                print(f"\n📊 Change Pattern:")
                print(f"   Increasing periods: {pattern['increasing_periods']}")
                print(f"   Decreasing periods: {pattern['decreasing_periods']}")
                print(f"   Stable periods: {pattern['stable_periods']}")
            
            print("=" * 60)