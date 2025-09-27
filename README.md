# ProGenomeEvoSimulator

A high-performance prokaryotic genome evolution simulator with comprehensive evolutionary mechanisms and parallel processing capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Performance-5--20x_speedup-red.svg)](#performance)
[![Version](https://img.shields.io/badge/Version-1.0.0-green.svg)](#version)

**Version**: 1.0.0  
**Release Date**: September 27, 2025  
**Status**: Production Ready

## Overview

ProGenomeEvoSimulator is a comprehensive computational platform designed to simulate the evolution of prokaryotic genomes. It implements all major evolutionary mechanisms including point mutations, horizontal gene transfer (HGT), homologous recombination, and gene loss, with advanced parallel processing capabilities for high-performance computing environments.

## Key Features

### Complete Evolutionary Mechanisms
- **Point Mutations** - Optimized algorithms with transition bias and hotspot modeling
- **Horizontal Gene Transfer (HGT)** - Gene acquisition from external sources
- **Homologous Recombination** - Sequence similarity-based genetic exchange
- **Gene Loss** - Intelligent gene deletion with core gene protection

### High-Performance Computing
- **Parallel Processing** - Automatic multi-core CPU utilization with 5-20x speedup
- **Intelligent Optimization** - Adaptive chunking and load balancing
- **Memory Efficiency** - Optimized data structures and caching mechanisms
- **Scalability** - Support for small to large-scale genome simulations

### User-Friendly Interface
- **Unified Engine** - Single interface for all evolutionary mechanisms
- **Interactive Interface** - Menu-driven system for advanced configurations
- **Flexible Configuration** - Multiple presets and custom parameter options
- **Comprehensive Documentation** - Detailed guides and examples

## Installation

### Requirements
- Python 3.8 or higher
- NumPy
- Matplotlib (optional, for visualizations)

### Setup
```bash
git clone <repository-url>
cd ProGenomeEvoSimulator
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```bash
# Run basic simulation (recommended for new users)
python main.py

# Interactive advanced interface
python main_unified.py

# Feature demonstration and performance testing
python demo_unified_engine.py
```

### Programmatic Usage
```python
from core.unified_evolution_engine import UnifiedEvolutionEngine
from core.genome import create_initial_genome

# Create initial genome
genome = create_initial_genome(
    gene_count=2000,
    avg_gene_length=500,
    min_gene_length=200
)

# Initialize evolution engine
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,
    hgt_rate=0.01,
    recombination_rate=1e-3,
    enable_gene_loss=True,
    enable_parallel=True
)

# Run simulation
final_genome, snapshots = engine.simulate_evolution(
    initial_genome=genome,
    generations=1000
)
```

## Architecture Design

### Core Components

```
ProGenomeEvoSimulator/
├── core/
│   ├── unified_evolution_engine.py    # Main evolution engine
│   └── genome.py                      # Genome data structures
├── mechanisms/
│   ├── gene_loss.py                   # Gene loss mechanism
│   ├── point_mutation_optimized.py   # Optimized point mutations
│   ├── horizontal_transfer.py         # Horizontal gene transfer
│   └── homologous_recombination.py   # Homologous recombination
├── analysis/
│   └── evolution_analyzer.py          # Evolution analysis tools
├── main.py                            # Basic simulation entry point
├── main_unified.py                    # Interactive interface
└── demo_unified_engine.py             # Feature demonstrations
```

### Design Principles

1. **Unified Architecture** - Single engine integrating all evolutionary mechanisms
2. **Modular Design** - Clear separation of concerns for maintainability
3. **Performance Optimization** - Parallel processing and algorithmic improvements
4. **Biological Accuracy** - Realistic modeling of prokaryotic evolution
5. **Extensibility** - Easy addition of new mechanisms and features

### Evolution Engine Design

The `UnifiedEvolutionEngine` serves as the central component that orchestrates all evolutionary processes:

- **Mechanism Integration** - Coordinates point mutations, HGT, recombination, and gene loss
- **Parallel Processing** - Automatically distributes work across available CPU cores
- **Parameter Management** - Handles complex parameter interactions and dependencies
- **Performance Monitoring** - Tracks and optimizes computational efficiency

## Configuration Options

### Quick Test Configuration
```python
# High parameter values for rapid evolution observation
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-4,
    hgt_rate=0.05,
    recombination_rate=1e-2,
    loss_rate=1e-4,
    enable_gene_loss=True,
    enable_parallel=True
)
```

### Realistic Simulation Configuration
```python
# Parameters approximating real prokaryotic evolution
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-6,
    hgt_rate=1e-5,
    recombination_rate=1e-6,
    loss_rate=1e-6,
    enable_gene_loss=True,
    enable_parallel=True
)
```

### Performance-Optimized Configuration
```python
# Maximum computational performance
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,
    hgt_rate=0.01,
    enable_parallel=True,
    enable_optimization=True,
    num_processes=None  # Use all available CPU cores
)
```

## Performance

### Parallel Processing Benchmarks

| Genome Size | Generations | Serial Time | Parallel Time | Speedup |
|-------------|-------------|-------------|---------------|---------|
| 1,000 genes | 100         | 45s         | 12s           | **3.75x** |
| 3,000 genes | 500         | 8.5 min     | 1.2 min       | **7.1x** |
| 5,000 genes | 1,000       | 35 min      | 2.8 min       | **12.5x** |

### System Requirements

**Minimum:**
- Python 3.8+, NumPy
- 2GB RAM, Single-core CPU

**Recommended:**
- Python 3.9+, NumPy, Matplotlib
- 8GB+ RAM, 4+ core CPU

**Optimal:**
- Python 3.10+, Full scientific stack
- 16GB+ RAM, 8+ core CPU

## Biological Features

### Evolutionary Mechanisms

- **Point Mutation Rate**: 1e-9 to 1e-4 (configurable)
- **HGT Frequency**: 1e-8 to 0.1 (configurable)
- **Recombination Rate**: 1e-9 to 1e-2 (configurable)
- **Gene Loss Rate**: 1e-8 to 1e-3 (configurable)

### Genome Properties

- **Gene Count**: 100 to 10,000+ genes
- **Gene Length**: 50 to 5,000+ bp
- **Genome Size**: Automatic regulation and balancing
- **Core Gene Protection**: Configurable protection levels

### Gene Loss Mechanism

The gene loss system implements intelligent deletion with:
- **Core Gene Protection** - Essential genes receive 95%+ protection
- **HGT Gene Preference** - Recently acquired genes have higher loss rates
- **Genome Size Regulation** - Maintains optimal genome size through dynamic pressure
- **Adaptive Loss Rates** - Context-dependent deletion probabilities

## Applications

### Research Applications
- **Evolutionary Biology** - Genome evolution pattern analysis
- **Comparative Genomics** - Inter-species genome difference studies
- **Phylogenetic Analysis** - Evolutionary relationship reconstruction
- **Functional Genomics** - Gene function evolution research

### Educational Applications
- **Bioinformatics Teaching** - Evolution algorithm demonstrations
- **Molecular Evolution Courses** - Theoretical concept visualization
- **Computational Biology** - Algorithm implementation learning

### Method Development
- **Algorithm Testing** - New method validation platform
- **Parameter Optimization** - Model parameter tuning
- **Performance Benchmarking** - Algorithm performance comparison

## API Reference

### Core Classes

#### UnifiedEvolutionEngine
Main evolution engine class integrating all mechanisms.

```python
engine = UnifiedEvolutionEngine(
    mutation_rate=1e-5,           # Point mutation rate per base per generation
    hgt_rate=0.01,               # HGT event rate per generation
    recombination_rate=1e-3,     # Recombination rate per gene pair
    enable_gene_loss=True,       # Enable gene loss mechanism
    loss_rate=1e-5,              # Gene loss rate per gene per generation
    core_gene_protection=0.95,   # Core gene protection coefficient
    enable_parallel=True,        # Enable parallel processing
    num_processes=None           # Number of processes (None = auto)
)
```

#### Key Methods

- `simulate_evolution(genome, generations)` - Run complete evolution simulation
- `evolve_multiple_generations(genome, generations)` - Multi-generation evolution
- `get_evolution_summary(genome)` - Comprehensive evolution statistics
- `get_performance_analysis()` - Performance metrics and analysis

### Genome Class
Represents a prokaryotic genome with genes and metadata.

```python
genome = create_initial_genome(
    gene_count=2000,        # Number of genes
    avg_gene_length=500,    # Average gene length in bp
    min_gene_length=200     # Minimum gene length in bp
)
```

## Contributing

### Development Setup
```bash
git clone <repository-url>
cd ProGenomeEvoSimulator
pip install -r requirements.txt
python demo_unified_engine.py  # Verify installation
```

### Code Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Reporting Issues
- Use GitHub Issues for bug reports
- Include detailed error information
- Provide system environment details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ProGenomeEvoSimulator in your research, please cite:

```
ProGenomeEvoSimulator: A High-Performance Prokaryotic Genome Evolution Simulator
[Author information and publication details to be added]
```

## Acknowledgments

We thank the computational biology and evolutionary genomics communities for their contributions to understanding prokaryotic genome evolution.

---

**ProGenomeEvoSimulator** - Making genome evolution simulation faster, more accurate, and more accessible! 🧬⚡🎯