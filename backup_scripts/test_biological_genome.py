#!/usr/bin/env python3
"""
测试生物学正确的基因组模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.genome import (
    create_initial_genome, 
    generate_biologically_correct_gene,
    validate_gene_sequence,
    GENETIC_CODE,
    START_CODONS,
    STOP_CODONS
)

def test_single_gene_generation():
    """测试单个基因生成"""
    print("🧬 测试单个基因生成")
    print("=" * 50)
    
    # 测试不同长度的基因
    test_lengths = [150, 300, 600, 999, 1200]
    
    for target_length in test_lengths:
        print(f"\n📏 目标长度: {target_length} bp")
        
        # 生成基因
        gene_sequence = generate_biologically_correct_gene(target_length)
        
        # 验证基因
        validation = validate_gene_sequence(gene_sequence)
        
        print(f"   实际长度: {len(gene_sequence)} bp")
        print(f"   密码子数: {len(gene_sequence) // 3}")
        print(f"   起始密码子: {gene_sequence[:3]}")
        print(f"   终止密码子: {gene_sequence[-3:]}")
        print(f"   验证结果: {'✅ 有效' if validation['is_valid'] else '❌ 无效'}")
        
        if not validation['is_valid']:
            print(f"   错误: {validation['errors']}")
        
        # 显示前几个和后几个密码子
        codons = [gene_sequence[i:i+3] for i in range(0, len(gene_sequence), 3)]
        print(f"   密码子序列: {' '.join(codons[:3])} ... {' '.join(codons[-3:])}")

def test_codon_validation():
    """测试密码子验证"""
    print("\n🔬 测试密码子验证")
    print("=" * 50)
    
    # 测试有效基因
    valid_gene = "ATGAAATTTGGATAA"  # ATG AAA TTT GGA TAA
    validation = validate_gene_sequence(valid_gene)
    print(f"有效基因 '{valid_gene}': {'✅ 通过' if validation['is_valid'] else '❌ 失败'}")
    
    # 测试无效基因（长度不是3的倍数）
    invalid_gene1 = "ATGAAATTTGG"  # 11 bp
    validation = validate_gene_sequence(invalid_gene1)
    print(f"无效基因1 '{invalid_gene1}': {'✅ 通过' if validation['is_valid'] else '❌ 失败'} - {validation['errors']}")
    
    # 测试无效基因（没有起始密码子）
    invalid_gene2 = "AAAATTTGGATAA"  # AAA ATT TGG ATA A
    validation = validate_gene_sequence(invalid_gene2)
    print(f"无效基因2 '{invalid_gene2}': {'✅ 通过' if validation['is_valid'] else '❌ 失败'} - {validation['errors']}")
    
    # 测试无效基因（没有终止密码子）
    invalid_gene3 = "ATGAAATTTGGA"  # ATG AAA TTT GGA
    validation = validate_gene_sequence(invalid_gene3)
    print(f"无效基因3 '{invalid_gene3}': {'✅ 通过' if validation['is_valid'] else '❌ 失败'} - {validation['errors']}")

def test_genome_generation():
    """测试基因组生成"""
    print("\n🧬 测试小型基因组生成")
    print("=" * 50)
    
    # 生成小型测试基因组
    genome = create_initial_genome(
        gene_count=10,
        avg_gene_length=600,
        min_gene_length=150,
        use_biological_sequences=True
    )
    
    print(f"\n📊 基因组验证结果:")
    print(f"   基因数量: {genome.gene_count}")
    print(f"   基因组大小: {genome.size:,} bp")
    print(f"   平均基因长度: {genome.size / genome.gene_count:.1f} bp")
    
    # 验证每个基因
    valid_count = 0
    invalid_count = 0
    
    print(f"\n🔍 逐个基因验证:")
    for i, gene in enumerate(genome.genes):
        validation = validate_gene_sequence(gene.sequence)
        status = "✅" if validation['is_valid'] else "❌"
        print(f"   基因 {i+1:2d}: {status} 长度={len(gene.sequence):3d}bp, 密码子={len(gene.sequence)//3:2d}, "
              f"起始={gene.sequence[:3]}, 终止={gene.sequence[-3:]}")
        
        if validation['is_valid']:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"           错误: {validation['errors']}")
    
    print(f"\n📈 总体验证结果:")
    print(f"   有效基因: {valid_count}/{genome.gene_count} ({valid_count/genome.gene_count*100:.1f}%)")
    print(f"   无效基因: {invalid_count}/{genome.gene_count} ({invalid_count/genome.gene_count*100:.1f}%)")

def test_codon_distribution():
    """测试密码子分布"""
    print("\n📊 测试密码子分布")
    print("=" * 50)
    
    # 生成多个基因并统计密码子使用
    start_codon_count = {}
    stop_codon_count = {}
    
    for _ in range(100):
        gene_seq = generate_biologically_correct_gene(300)
        start_codon = gene_seq[:3]
        stop_codon = gene_seq[-3:]
        
        start_codon_count[start_codon] = start_codon_count.get(start_codon, 0) + 1
        stop_codon_count[stop_codon] = stop_codon_count.get(stop_codon, 0) + 1
    
    print("起始密码子分布:")
    for codon, count in start_codon_count.items():
        print(f"   {codon}: {count:2d} ({count/100*100:.1f}%)")
    
    print("终止密码子分布:")
    for codon, count in stop_codon_count.items():
        print(f"   {codon}: {count:2d} ({count/100*100:.1f}%)")

def main():
    """主测试函数"""
    print("🧬 生物学正确基因组模型测试")
    print("=" * 60)
    
    try:
        # 测试单个基因生成
        test_single_gene_generation()
        
        # 测试密码子验证
        test_codon_validation()
        
        # 测试基因组生成
        test_genome_generation()
        
        # 测试密码子分布
        test_codon_distribution()
        
        print("\n🎉 所有测试完成!")
        print("✅ 基因组模型现在生成生物学上正确的基因序列")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()