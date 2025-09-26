import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from core.genome import Genome, Gene

class OptimizedPointMutationEngine:
    """Optimized point mutation engine with improved performance for large-scale simulations"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-9,
                 enable_transition_bias: bool = True,
                 transition_transversion_ratio: float = 2.0,
                 enable_hotspots: bool = True,
                 hotspot_multiplier: float = 5.0,
                 hotspot_motifs: Optional[List[str]] = None):
        """
        Initialize optimized point mutation engine
        
        Args:
            mutation_rate: Base mutation rate per bp per generation
            enable_transition_bias: Whether to apply transition/transversion bias
            transition_transversion_ratio: Ratio of transitions to transversions (typically 2-4)
            enable_hotspots: Whether to enable mutation hotspots
            hotspot_multiplier: Mutation rate multiplier for hotspot regions
            hotspot_motifs: List of DNA motifs that are mutation hotspots (default: CpG sites)
        """
        self.mutation_rate = mutation_rate
        self.enable_transition_bias = enable_transition_bias
        self.transition_transversion_ratio = transition_transversion_ratio
        self.enable_hotspots = enable_hotspots
        self.hotspot_multiplier = hotspot_multiplier
        
        # Default hotspot motifs (CpG dinucleotides and other known hotspots)
        self.hotspot_motifs = hotspot_motifs or ['CG', 'GC', 'AT', 'TA']
        
        self.bases = ['A', 'T', 'G', 'C']
        
        # Pre-compute mutation probabilities and matrices
        self._setup_mutation_probabilities()
        self._setup_transition_sets()
        
        # Cache for hotspot positions to avoid recomputation
        self._hotspot_cache = {}
        
        # Statistics tracking
        self.mutation_stats = {
            'transitions': 0,
            'transversions': 0,
            'hotspot_mutations': 0,
            'regular_mutations': 0
        }
        
        # Pre-allocate arrays for batch processing
        self._batch_size = 10000  # Process mutations in batches
    
    def _setup_mutation_probabilities(self):
        """Setup mutation probability matrices based on transition/transversion ratio"""
        if self.enable_transition_bias:
            # Calculate probabilities based on Ti/Tv ratio
            total_weight = self.transition_transversion_ratio * 2 + 4  # 2 transitions + 4 transversions
            transition_prob = self.transition_transversion_ratio / total_weight
            transversion_prob = 1.0 / total_weight
            
            self.mutation_matrix = {
                'A': {
                    'G': transition_prob,      # Transition
                    'C': transversion_prob,    # Transversion
                    'T': transversion_prob     # Transversion
                },
                'T': {
                    'C': transition_prob,      # Transition
                    'A': transversion_prob,    # Transversion
                    'G': transversion_prob     # Transversion
                },
                'G': {
                    'A': transition_prob,      # Transition
                    'C': transversion_prob,    # Transversion
                    'T': transversion_prob     # Transversion
                },
                'C': {
                    'T': transition_prob,      # Transition
                    'G': transversion_prob,    # Transversion
                    'A': transversion_prob     # Transversion
                }
            }
        else:
            # Equal probability for all mutations
            equal_prob = 1.0 / 3.0
            self.mutation_matrix = {
                'A': {'G': equal_prob, 'C': equal_prob, 'T': equal_prob},
                'T': {'C': equal_prob, 'A': equal_prob, 'G': equal_prob},
                'G': {'A': equal_prob, 'T': equal_prob, 'C': equal_prob},
                'C': {'T': equal_prob, 'G': equal_prob, 'A': equal_prob}
            }
        
        # Pre-compute cumulative probabilities for faster sampling
        self.cumulative_probs = {}
        for base, probs in self.mutation_matrix.items():
            bases = list(probs.keys())
            cum_probs = np.cumsum(list(probs.values()))
            self.cumulative_probs[base] = (bases, cum_probs)
    
    def _setup_transition_sets(self):
        """Pre-compute transition pairs for faster lookup"""
        self.transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    
    def _find_hotspot_positions_cached(self, gene: Gene) -> Set[int]:
        """Find positions of mutation hotspot motifs with caching"""
        if not self.enable_hotspots:
            return set()
        
        # Use gene ID as cache key
        cache_key = gene.id
        if cache_key in self._hotspot_cache:
            return self._hotspot_cache[cache_key]
        
        hotspot_positions = set()
        sequence = gene.sequence
        
        for motif in self.hotspot_motifs:
            motif_len = len(motif)
            for i in range(len(sequence) - motif_len + 1):
                if sequence[i:i + motif_len] == motif:
                    # Add all positions within the motif
                    hotspot_positions.update(range(i, i + motif_len))
        
        # Cache the result
        self._hotspot_cache[cache_key] = hotspot_positions
        return hotspot_positions
    
    def _calculate_batch_mutation_rates(self, gene: Gene) -> np.ndarray:
        """Calculate mutation rates for all positions in a gene at once"""
        base_rate = self.mutation_rate
        rates = np.full(gene.length, base_rate, dtype=np.float64)
        
        if self.enable_hotspots:
            hotspot_positions = self._find_hotspot_positions_cached(gene)
            if hotspot_positions:
                # Convert to numpy array for vectorized operations
                hotspot_array = np.array(list(hotspot_positions))
                rates[hotspot_array] *= self.hotspot_multiplier
        
        return rates
    
    def _get_mutated_base_fast(self, original_base: str) -> str:
        """Fast mutated base selection using pre-computed cumulative probabilities"""
        if original_base not in self.cumulative_probs:
            return original_base
        
        bases, cum_probs = self.cumulative_probs[original_base]
        rand_val = np.random.random()
        
        # Find the first cumulative probability greater than random value
        idx = np.searchsorted(cum_probs, rand_val)
        return bases[idx] if idx < len(bases) else bases[-1]
    
    def apply_mutations_optimized(self, genome: Genome, generations: int = 1) -> int:
        """Apply point mutations with optimized batch processing"""
        total_mutations = 0
        
        for gene in genome.genes:
            # Calculate mutation rates for all positions at once
            position_rates = self._calculate_batch_mutation_rates(gene)
            expected_mutations = position_rates * generations
            
            # Use vectorized Poisson sampling for all positions
            mutation_events = np.random.poisson(expected_mutations)
            mutation_positions = np.where(mutation_events > 0)[0]
            
            if len(mutation_positions) == 0:
                continue
            
            # Get hotspot positions for statistics
            hotspot_positions = self._find_hotspot_positions_cached(gene) if self.enable_hotspots else set()
            
            # Process mutations in batches to avoid memory issues
            sequence_list = list(gene.sequence)
            
            for pos in mutation_positions:
                original_base = sequence_list[pos]
                mutated_base = self._get_mutated_base_fast(original_base)
                
                if mutated_base != original_base:
                    sequence_list[pos] = mutated_base
                    total_mutations += 1
                    gene.mutation_count += 1
                    
                    # Track mutation statistics
                    if (original_base, mutated_base) in self.transitions:
                        self.mutation_stats['transitions'] += 1
                    else:
                        self.mutation_stats['transversions'] += 1
                    
                    # Check if it's a hotspot mutation
                    if pos in hotspot_positions:
                        self.mutation_stats['hotspot_mutations'] += 1
                    else:
                        self.mutation_stats['regular_mutations'] += 1
            
            # Update gene sequence once at the end
            gene.sequence = ''.join(sequence_list)
        
        genome.total_mutations += total_mutations
        return total_mutations
    
    def apply_mutations(self, genome: Genome, generations: int = 1) -> int:
        """Apply point mutations - wrapper that uses optimized version"""
        return self.apply_mutations_optimized(genome, generations)
    
    def get_mutation_statistics(self, genome: Genome) -> dict:
        """Get comprehensive mutation statistics"""
        mutation_counts = [gene.mutation_count for gene in genome.genes]
        
        # Calculate Ti/Tv ratio
        ti_tv_ratio = 0
        if self.mutation_stats['transversions'] > 0:
            ti_tv_ratio = self.mutation_stats['transitions'] / self.mutation_stats['transversions']
        
        # Calculate hotspot statistics
        total_tracked_mutations = (self.mutation_stats['hotspot_mutations'] + 
                                 self.mutation_stats['regular_mutations'])
        hotspot_percentage = 0
        if total_tracked_mutations > 0:
            hotspot_percentage = (self.mutation_stats['hotspot_mutations'] / 
                                total_tracked_mutations) * 100
        
        return {
            'total_mutations': sum(mutation_counts),
            'avg_mutations_per_gene': np.mean(mutation_counts) if mutation_counts else 0,
            'max_mutations_per_gene': max(mutation_counts) if mutation_counts else 0,
            'genes_with_mutations': sum(1 for count in mutation_counts if count > 0),
            'mutation_rate_per_bp': sum(mutation_counts) / genome.size if genome.size > 0 else 0,
            
            # Enhanced statistics
            'transitions': self.mutation_stats['transitions'],
            'transversions': self.mutation_stats['transversions'],
            'ti_tv_ratio': ti_tv_ratio,
            'hotspot_mutations': self.mutation_stats['hotspot_mutations'],
            'regular_mutations': self.mutation_stats['regular_mutations'],
            'hotspot_percentage': hotspot_percentage,
            
            # Configuration info
            'transition_bias_enabled': self.enable_transition_bias,
            'hotspots_enabled': self.enable_hotspots,
            'target_ti_tv_ratio': self.transition_transversion_ratio,
            'hotspot_multiplier': self.hotspot_multiplier,
            'hotspot_motifs': self.hotspot_motifs,
            
            # Performance info
            'cache_size': len(self._hotspot_cache),
            'optimization_enabled': True
        }
    
    def reset_statistics(self):
        """Reset mutation statistics counters"""
        self.mutation_stats = {
            'transitions': 0,
            'transversions': 0,
            'hotspot_mutations': 0,
            'regular_mutations': 0
        }
    
    def clear_cache(self):
        """Clear hotspot position cache (useful for memory management)"""
        self._hotspot_cache.clear()
    
    def get_hotspot_analysis(self, genome: Genome) -> Dict:
        """Analyze hotspot distribution across the genome"""
        if not self.enable_hotspots:
            return {'hotspots_enabled': False}
        
        total_hotspot_positions = 0
        total_positions = 0
        hotspot_density_per_gene = []
        
        for gene in genome.genes:
            hotspot_positions = self._find_hotspot_positions_cached(gene)
            total_hotspot_positions += len(hotspot_positions)
            total_positions += gene.length
            
            density = len(hotspot_positions) / gene.length if gene.length > 0 else 0
            hotspot_density_per_gene.append(density)
        
        overall_density = total_hotspot_positions / total_positions if total_positions > 0 else 0
        
        return {
            'hotspots_enabled': True,
            'total_hotspot_positions': total_hotspot_positions,
            'total_genome_positions': total_positions,
            'overall_hotspot_density': overall_density,
            'avg_hotspot_density_per_gene': np.mean(hotspot_density_per_gene),
            'hotspot_motifs': self.hotspot_motifs,
            'hotspot_multiplier': self.hotspot_multiplier,
            'cache_efficiency': len(self._hotspot_cache)
        }