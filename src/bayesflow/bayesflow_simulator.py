"""
BayesFlow-compatible simulator functions for protein secondary structure inference.
Following the patterns from BayesFlow examples (Two Moons, SIR, etc.).
"""

import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.hmm.protein_hmm import ProteinHMM


class ProteinHMMSimulator:
    """Protein HMM simulator for BayesFlow, following example patterns."""
    
    def __init__(self, min_length: int = 50, max_length: int = 150):
        self.min_length = min_length
        self.max_length = max_length
        self.hmm = ProteinHMM()
        
    def sequence_prior(self):
        """Generate sequence length from prior distribution."""
        length = np.random.randint(self.min_length, self.max_length + 1)
        return dict(seq_length=length)
    
    def sequence_simulator(self, seq_length):
        """Generate protein sequence and state probabilities given length."""
        length = int(seq_length)
        
        # Generate sequence using HMM
        sequence, true_states = self.hmm.generate_sequence(length)
        
        # Get state probabilities using Forward-Backward
        _, state_prob_matrix = self.hmm.predict_states(sequence)
        
        # Convert sequence to indices
        seq_indices = self.hmm.sequence_to_indices(sequence).flatten()
        
        # Return BayesFlow-compatible format
        return dict(
            sequence=seq_indices.astype(np.float32),
            state_probs=state_prob_matrix.astype(np.float32),
            length=np.array([length], dtype=np.float32)
        )


# Create simulator instance
_protein_simulator = ProteinHMMSimulator()

def sequence_prior():
    """BayesFlow-compatible sequence prior function."""
    return _protein_simulator.sequence_prior()

def protein_likelihood(seq_length):
    """BayesFlow-compatible protein likelihood function."""
    return _protein_simulator.sequence_simulator(seq_length)


# Fixed-length version for simpler training
class FixedLengthProteinSimulator:
    """Fixed-length protein simulator for easier BayesFlow integration."""
    
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.hmm = ProteinHMM()
        
    def generate_sequence(self):
        """Generate a single sequence with state probabilities."""
        # Generate sequence using HMM
        sequence, true_states = self.hmm.generate_sequence(self.sequence_length)
        
        # Get state probabilities using Forward-Backward  
        _, state_prob_matrix = self.hmm.predict_states(sequence)
        
        # Convert sequence to indices
        seq_indices = self.hmm.sequence_to_indices(sequence).flatten()
        
        return dict(
            sequence=seq_indices.astype(np.float32),
            state_probs=state_prob_matrix.flatten().astype(np.float32)  # Flatten for BayesFlow
        )


# Fixed-length simulator for simple training
_fixed_simulator = FixedLengthProteinSimulator(sequence_length=100)

def protein_simulator():
    """Simple fixed-length protein simulator for BayesFlow."""
    return _fixed_simulator.generate_sequence()


def create_protein_simulator_functions(fixed_length: bool = True, seq_length: int = 100):
    """
    Create BayesFlow-compatible simulator functions.
    
    Args:
        fixed_length: Whether to use fixed-length sequences
        seq_length: Sequence length (if fixed_length=True)
        
    Returns:
        List of simulator functions for bf.make_simulator()
    """
    if fixed_length:
        # Create a fresh simulator instance for this configuration
        global _fixed_simulator
        _fixed_simulator = FixedLengthProteinSimulator(seq_length)
        # Return the function itself, not the bound method
        return [protein_simulator]
    else:
        # Variable-length simulator with prior
        return [sequence_prior, protein_likelihood]
