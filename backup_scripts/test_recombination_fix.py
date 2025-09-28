#!/usr/bin/env python3
"""测试重组修复效果的脚本"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.genome import create_initial_genome
from mechanisms.homologous_recombination import HomologousRecombination

def test_recombination_fix():
    print("🧪 Testing Recombination Fix")
    print("=" * 50)
    
    # 创建测试基因组
    genome = create_initial_genome(gene_count=100, avg_gene_length=500)
    print(f"Created genome: {genome.gene_count} genes")
    
    # 创建重组引擎
    recombination = HomologousRecombination(
        recombination_rate=0.1,
        min_similarity=0.3,
        enable_debug=True
    )
    
    # 执行重组测试
    initial_count = genome.total_recombination_events
    successful = recombination.apply_recombination(genome, generations=1)
    final_count = genome.total_recombination_events
    
    print(f"\n📊 Results:")
    print(f"Successful recombinations: {successful}")
    print(f"Total events: {initial_count} -> {final_count}")
    
    if successful > 0:
        print("✅ RECOMBINATION FIX SUCCESSFUL!")
        return True
    else:
        print("❌ RECOMBINATION STILL NOT WORKING")
        return False

if __name__ == "__main__":
    success = test_recombination_fix()
    sys.exit(0 if success else 1)