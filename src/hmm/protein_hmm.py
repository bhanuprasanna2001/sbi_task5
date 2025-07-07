"""
Core HMM implementation for protein secondary structure prediction.
Implements two-state HMM with fixed probabilities using hmmlearn.
"""

import numpy as np
from hmmlearn import hmm
from typing import List, Tuple, Optional
import logging
from configs.hmm_config import (
    AMINO_ACIDS, STATES, EMISSION_PROBS, TRANSITION_PROBS, 
    INITIAL_PROBS, N_STATES, N_AMINO_ACIDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinHMM:
    """
    Two-state Hidden Markov Model for protein secondary structure prediction.
    States: 0='other' (beta-sheet, coil), 1='alpha_helix'
    """
    
    def __init__(self):
        """Initialize HMM with fixed probabilities from task description."""
        self.amino_acids = AMINO_ACIDS
        self.states = STATES
        self.n_states = N_STATES
        self.n_amino_acids = N_AMINO_ACIDS
        
        # Create amino acid to index mapping
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {idx: aa for idx, aa in enumerate(self.amino_acids)}
        
        # Initialize hmmlearn CategoricalHMM
        self.model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_features=self.n_amino_acids,
            random_state=42
        )
        
        # Set fixed probabilities
        self._set_model_parameters()
        
        logger.info(f"Initialized ProteinHMM with {self.n_states} states and {self.n_amino_acids} amino acids")
    
    def _set_model_parameters(self):
        """Set the fixed HMM parameters from task description."""
        # Set initial state probabilities
        self.model.startprob_ = INITIAL_PROBS.copy()
        
        # Set transition probabilities
        self.model.transmat_ = TRANSITION_PROBS.copy()
        
        # Set emission probabilities
        self.model.emissionprob_ = EMISSION_PROBS.copy()
        
        logger.info("Set fixed HMM parameters from task specification")
    
    def sequence_to_indices(self, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to numerical indices."""
        try:
            indices = [self.aa_to_idx[aa] for aa in sequence.upper()]
            return np.array(indices).reshape(-1, 1)
        except KeyError as e:
            raise ValueError(f"Unknown amino acid: {e}")
    
    def indices_to_sequence(self, indices: np.ndarray) -> str:
        """Convert numerical indices to amino acid sequence."""
        if indices.ndim > 1:
            indices = indices.flatten()
        return ''.join([self.idx_to_aa[idx] for idx in indices])
    
    def generate_sequence(self, length: int) -> Tuple[str, np.ndarray]:
        """
        Generate amino acid sequence and corresponding hidden states.
        
        Args:
            length: Length of sequence to generate
            
        Returns:
            tuple: (amino_acid_sequence, hidden_states)
        """
        # Generate sequence using hmmlearn
        observations, states = self.model.sample(length)
        
        # Convert to amino acid sequence
        aa_sequence = self.indices_to_sequence(observations)
        
        logger.debug(f"Generated sequence of length {length}")
        return aa_sequence, states
    
    def predict_states(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict hidden states for given amino acid sequence using Viterbi algorithm.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            tuple: (predicted_states, state_probabilities)
        """
        # Convert sequence to indices
        obs_indices = self.sequence_to_indices(sequence)
        
        # Predict states using Viterbi algorithm
        log_prob = self.model.score(obs_indices)
        predicted_states = self.model.predict(obs_indices)
        
        # Get state probabilities
        state_probs = self.model.predict_proba(obs_indices)
        
        logger.debug(f"Predicted states for sequence of length {len(sequence)}")
        return predicted_states, state_probs
    
    def get_likelihood(self, sequence: str) -> float:
        """Calculate log likelihood of sequence under the model."""
        obs_indices = self.sequence_to_indices(sequence)
        return self.model.score(obs_indices)
    
    def validate_model(self) -> bool:
        """Validate model parameters are properly set."""
        try:
            # Check probability sums
            assert np.allclose(self.model.startprob_.sum(), 1.0)
            assert np.allclose(self.model.transmat_.sum(axis=1), 1.0)
            assert np.allclose(self.model.emissionprob_.sum(axis=1), 1.0)
            
            # Check parameter shapes
            assert self.model.startprob_.shape == (self.n_states,)
            assert self.model.transmat_.shape == (self.n_states, self.n_states)
            assert self.model.emissionprob_.shape == (self.n_states, self.n_amino_acids)
            
            logger.info("Model validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get detailed information about the model parameters."""
        return {
            'n_states': self.n_states,
            'n_amino_acids': self.n_amino_acids,
            'states': self.states,
            'amino_acids': self.amino_acids,
            'initial_probs': self.model.startprob_.tolist(),
            'transition_probs': self.model.transmat_.tolist(),
            'emission_probs': self.model.emissionprob_.tolist()
        }
