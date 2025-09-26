# Prokaryotic Genome Evolution Simulator

This is a Python program for simulating the evolutionary process of prokaryotic genomes. It can simulate genome changes over a specified time period and analyze the Average Nucleotide Identity (ANI) and orthologous gene identity distribution between genomes before and after evolution.

## Features

### Evolution Mechanisms
- **Point Mutations**: Random point mutations following Poisson distribution with configurable mutation rates
- **Horizontal Gene Transfer (HGT)**: Acquisition of new genes from external gene pools, simulating horizontal gene transfer
- **Homologous Recombination**: Recombination between similar genes, generating multiple site differences in a single event

### Analysis Functions
- **Average Nucleotide Identity (ANI) Calculation**: Compare overall similarity between genomes before and after evolution
- **Orthologous Gene Identification and Analysis**: Identify orthologous gene pairs and analyze their identity distribution
- **Evolution Statistics**: Detailed statistics of evolutionary events and genome composition changes

## Project Structure

```
simulator/
├── core/
│   ├── genome.py              # Genome and gene classes
│   └── evolution_engine.py    # Evolution engine
├── mechanisms/
│   ├── point_mutation.py      # Point mutation mechanism
│   ├── horizontal_transfer.py # Horizontal gene transfer
│   └── homologous_recombination.py # Homologous recombination
├── analysis/
│   └── ani_calculator.py      # ANI calculation and ortholog analysis
├── main.py                    # Main program
├── demo.py                    # Demo program
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Installation and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Simulation
```bash
# Quick demo (recommended for first try)
python demo.py

# Full-scale simulation
python main.py
```

## Parameter Configuration

You can adjust the following parameters in `main.py` or `demo.py`:

```python
run_evolution_simulation(
    generations=1000,           # Evolution generations
    initial_gene_count=3000,    # Initial gene count
    mutation_rate=1e-8,         # Point mutation rate (per bp per generation)
    hgt_rate=0.002,            # HGT rate (per genome per generation)
    recombination_rate=1e-5     # Recombination rate (per bp per generation)
)
```

## Output Results

### 1. Console Output
- Evolution progress
- ANI analysis results
- Orthologous gene identity distribution statistics
- Genome composition and size changes

### 2. Visualization Charts
The program generates visualization files containing:
- Orthologous gene identity distribution histogram
- Genome size change trends
- Gene count changes (total genes, core genes, HGT genes)
- Evolution event accumulation plots (mutations, HGT, recombination)

## Core Concepts

### ANI (Average Nucleotide Identity)
Average Nucleotide Identity measures overall similarity between two genomes. Calculation method:
1. Identify orthologous gene pairs
2. Calculate sequence identity for each orthologous gene pair
3. Compute weighted average by gene length

### Orthologous Gene Identification
Orthologous genes are identified based on:
- Gene ID matching (from common ancestors)
- Sequence similarity threshold (default 50%)
- Minimum alignment length requirement

### Evolution Mechanism Parameters
- **Point mutation rate**: Typical values 1e-9 to 1e-8 per bp per generation
- **HGT rate**: Typical values 0.001 to 0.01 per genome per generation
- **Recombination rate**: Typical values 1e-6 to 1e-5 per bp per generation

## Usage Example

```python
from core.genome import create_initial_genome
from core.evolution_engine import EvolutionEngine
from analysis.ani_calculator import ANICalculator

# Create initial genome
initial_genome = create_initial_genome(gene_count=3000)

# Set up evolution engine
engine = EvolutionEngine(
    mutation_rate=1e-8,
    hgt_rate=0.002,
    recombination_rate=1e-5
)

# Run evolution
evolved_genome, snapshots = engine.simulate_evolution(
    initial_genome, generations=1000
)

# Analyze results
calculator = ANICalculator()
analysis = calculator.compare_genomes_comprehensive(
    initial_genome, evolved_genome
)
```

## Extended Features

Potential extensions for future development:
- Selection pressure simulation
- Gene functional classification
- More complex recombination mechanisms
- Genome structural variations
- Multi-genome comparative analysis

## Notes

1. Simulation parameters should be adjusted according to specific research objectives
2. Large-scale simulations may require considerable runtime
3. Memory usage scales with genome size and number of generations
4. Recommended to test with small-scale parameters before running large-scale simulations

## Quick Start

1. **First-time users**: Run `python demo.py` for a quick demonstration
2. **Testing**: Run `python quick_test.py` to verify all components work correctly
3. **Full simulation**: Run `python main.py` for comprehensive genome evolution simulation

The simulator is now fully functional with English interface and ready for use!