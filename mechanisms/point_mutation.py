import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.genome import Genome, Gene

class PointMutationEngine:
    """Enhanced point mutation engine with transition/transversion control and hotspots"""
    
    def __init__(self, 
                 mutation_rate: float = 1e-9,
                 enable_transition_bias: bool = True,
                 transition_transversion_ratio: float = 2.0,
                 enable_hotspots: bool = True,
                 hotspot_multiplier: float = 5.0,
                 hotspot_motifs: Optional[List[str]] = None):
        """
        Initialize enhanced point mutation engine
        
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
        
        # Calculate transition and transversion probabilities
        self._setup_mutation_probabilities()
        
        # Statistics tracking
        self.mutation_stats = {
            'transitions': 0,
            'transversions': 0,
            'hotspot_mutations': 0,
            'regular_mutations': 0
        }
    
    def _setup_mutation_probabilities(self):
        """Setup mutation probability matrices based on transition/transversion ratio"""
        if self.enable_transition_bias:
            # Calculate probabilities based on Ti/Tv ratio
            # Transitions: A<->G, C<->T (2 types)
            # Transversions: A<->C, A<->T, G<->C, G<->T (4 types)
            
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
    
    def _find_hotspot_positions(self, sequence: str) -> List[int]:
        """Find positions of mutation hotspot motifs in sequence"""
        hotspot_positions = []
        
        if not self.enable_hotspots:
            return hotspot_positions
        
        for motif in self.hotspot_motifs:
            motif_len = len(motif)
            for i in range(len(sequence) - motif_len + 1):
                if sequence[i:i + motif_len] == motif:
                    # Add all positions within the motif
                    hotspot_positions.extend(range(i, i + motif_len))
        
        return list(set(hotspot_positions))  # Remove duplicates
    
    def _calculate_position_mutation_rate(self, gene: Gene, position: int) -> float:
        """Calculate mutation rate for a specific position considering hotspots"""
        base_rate = self.mutation_rate
        
        if self.enable_hotspots:
            hotspot_positions = self._find_hotspot_positions(gene.sequence)
            if position in hotspot_positions:
                return base_rate * self.hotspot_multiplier
        
        return base_rate
    
    def _is_transition(self, original: str, mutated: str) -> bool:
        """Check if a mutation is a transition (A<->G, C<->T)"""
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        return (original, mutated) in transitions
    
    def calculate_mutations_per_gene(self, gene: Gene, generations: int = 1) -> List[Tuple[int, float]]:
        """
        Calculate mutations per gene considering position-specific rates
        Returns list of (position, mutation_rate) tuples
        """
        mutations = []
        
        for position in range(gene.length):
            position_rate = self._calculate_position_mutation_rate(gene, position)
            expected_mutations = position_rate * generations
            
            # Use Poisson distribution for each position
            if np.random.poisson(expected_mutations) > 0:
                mutations.append((position, position_rate))
        
        return mutations
    
    def get_mutated_base(self, original_base: str) -> str:
        """Select mutated base according to mutation matrix"""
        if original_base not in self.mutation_matrix:
            return original_base
        
        mutation_probs = self.mutation_matrix[original_base]
        bases = list(mutation_probs.keys())
        probs = list(mutation_probs.values())
        
        return np.random.choice(bases, p=probs)
    
    def apply_mutations(self, genome: Genome, generations: int = 1) -> int:
        """Apply point mutations to genome with enhanced features"""
        total_mutations = 0
        
        for gene in genome.genes:
            # Get mutations for this gene (considering hotspots)
            mutations = self.calculate_mutations_per_gene(gene, generations)
            
            for position, mutation_rate in mutations:
                original_base = gene.sequence[position]
                mutated_base = self.get_mutated_base(original_base)
                
                if mutated_base != original_base:
                    gene.mutate_position(position, mutated_base)
                    total_mutations += 1
                    
                    # Track mutation statistics
                    if self._is_transition(original_base, mutated_base):
                        self.mutation_stats['transitions'] += 1
                    else:
                        self.mutation_stats['transversions'] += 1
                    
                    # Check if it's a hotspot mutation
                    if self.enable_hotspots:
                        hotspot_positions = self._find_hotspot_positions(gene.sequence)
                        if position in hotspot_positions:
                            self.mutation_stats['hotspot_mutations'] += 1
                        else:
                            self.mutation_stats['regular_mutations'] += 1
                    else:
                        self.mutation_stats['regular_mutations'] += 1
        
        genome.total_mutations += total_mutations
        return total_mutations
    
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
            'hotspot_motifs': self.hotspot_motifs
        }
    
    def reset_statistics(self):
        """Reset mutation statistics counters"""
        self.mutation_stats = {
            'transitions': 0,
            'transversions': 0,
            'hotspot_mutations': 0,
            'regular_mutations': 0
        }
    
    def get_hotspot_analysis(self, genome: Genome) -> Dict:
        """Analyze hotspot distribution across the genome"""
        if not self.enable_hotspots:
            return {'hotspots_enabled': False}
        
        total_hotspot_positions = 0
        total_positions = 0
        hotspot_density_per_gene = []
        
        for gene in genome.genes:
            hotspot_positions = self._find_hotspot_positions(gene.sequence)
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
            'hotspot_multiplier': self.hotspot_multiplier
        }