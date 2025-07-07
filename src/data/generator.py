"""
Data generation utilities for protein secondary structure HMM.
Handles sequence generation, preprocessing, and dataset creation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

from src.hmm import ProteinHMM
from configs.hmm_config import (
    DEFAULT_SEQUENCE_LENGTH, DEFAULT_N_SEQUENCES, 
    RANDOM_SEED, TRAIN_TEST_SPLIT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SequenceData:
    """Data structure for protein sequence with annotations."""
    sequence: str
    states: np.ndarray
    state_probs: Optional[np.ndarray] = None
    length: int = 0
    alpha_helix_fraction: float = 0.0
    
    def __post_init__(self):
        self.length = len(self.sequence)
        self.alpha_helix_fraction = np.mean(self.states == 1)

class DataGenerator:
    """Generate synthetic protein sequences using HMM."""
    
    def __init__(self, hmm_model: Optional[ProteinHMM] = None, random_seed: int = RANDOM_SEED):
        """
        Initialize data generator.
        
        Args:
            hmm_model: ProteinHMM instance, creates new if None
            random_seed: Random seed for reproducibility
        """
        self.hmm_model = hmm_model if hmm_model else ProteinHMM()
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized DataGenerator with seed {random_seed}")
    
    def generate_sequences(self, 
                          n_sequences: int = DEFAULT_N_SEQUENCES,
                          sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                          length_variation: bool = True,
                          min_length: int = 50,
                          max_length: int = 200) -> List[SequenceData]:
        """
        Generate multiple protein sequences with varying lengths.
        
        Args:
            n_sequences: Number of sequences to generate
            sequence_length: Base sequence length
            length_variation: Whether to vary sequence lengths
            min_length: Minimum sequence length if varying
            max_length: Maximum sequence length if varying
            
        Returns:
            List of SequenceData objects
        """
        sequences = []
        
        for i in range(n_sequences):
            if length_variation:
                length = np.random.randint(min_length, max_length + 1)
            else:
                length = sequence_length
            
            # Generate sequence and states
            aa_sequence, states = self.hmm_model.generate_sequence(length)
            
            # Create SequenceData object
            seq_data = SequenceData(
                sequence=aa_sequence,
                states=states
            )
            
            sequences.append(seq_data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{n_sequences} sequences")
        
        logger.info(f"Generated {n_sequences} sequences with lengths {min_length}-{max_length}")
        return sequences
    
    def add_viterbi_predictions(self, sequences: List[SequenceData]) -> List[SequenceData]:
        """Add Viterbi state predictions and probabilities to sequences."""
        for i, seq_data in enumerate(sequences):
            predicted_states, state_probs = self.hmm_model.predict_states(seq_data.sequence)
            seq_data.state_probs = state_probs
            
            if (i + 1) % 100 == 0:
                logger.info(f"Added predictions for {i + 1}/{len(sequences)} sequences")
        
        return sequences
    
    def create_dataset(self,
                      n_train: int = 800,
                      n_test: int = 200,
                      sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                      **kwargs) -> Tuple[List[SequenceData], List[SequenceData]]:
        """
        Create train/test datasets.
        
        Args:
            n_train: Number of training sequences
            n_test: Number of test sequences
            sequence_length: Base sequence length
            **kwargs: Additional arguments for generate_sequences
            
        Returns:
            Tuple of (train_sequences, test_sequences)
        """
        total_sequences = n_train + n_test
        
        # Generate all sequences
        all_sequences = self.generate_sequences(
            n_sequences=total_sequences,
            sequence_length=sequence_length,
            **kwargs
        )
        
        # Add Viterbi predictions
        all_sequences = self.add_viterbi_predictions(all_sequences)
        
        # Split train/test
        train_sequences = all_sequences[:n_train]
        test_sequences = all_sequences[n_train:]
        
        logger.info(f"Created dataset: {n_train} train, {n_test} test sequences")
        return train_sequences, test_sequences

class DataProcessor:
    """Process and analyze sequence data."""
    
    @staticmethod
    def sequences_to_dataframe(sequences: List[SequenceData]) -> pd.DataFrame:
        """Convert sequence list to pandas DataFrame."""
        data = []
        for i, seq in enumerate(sequences):
            data.append({
                'sequence_id': i,
                'sequence': seq.sequence,
                'length': seq.length,
                'alpha_helix_fraction': seq.alpha_helix_fraction,
                'states': seq.states.tolist(),
                'state_probs': seq.state_probs.tolist() if seq.state_probs is not None else None
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def get_dataset_statistics(sequences: List[SequenceData]) -> Dict:
        """Calculate dataset statistics."""
        lengths = [seq.length for seq in sequences]
        alpha_fractions = [seq.alpha_helix_fraction for seq in sequences]
        
        stats = {
            'n_sequences': len(sequences),
            'length_stats': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            },
            'alpha_helix_stats': {
                'mean_fraction': np.mean(alpha_fractions),
                'std_fraction': np.std(alpha_fractions),
                'min_fraction': np.min(alpha_fractions),
                'max_fraction': np.max(alpha_fractions)
            }
        }
        
        return stats
    
    @staticmethod
    def save_sequences(sequences: List[SequenceData], 
                      filepath: Path,
                      format: str = 'pickle') -> None:
        """Save sequences to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(sequences, f)
        elif format == 'json':
            # Convert to serializable format
            data = []
            for seq in sequences:
                data.append({
                    'sequence': seq.sequence,
                    'states': seq.states.tolist(),
                    'state_probs': seq.state_probs.tolist() if seq.state_probs is not None else None,
                    'length': seq.length,
                    'alpha_helix_fraction': seq.alpha_helix_fraction
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(sequences)} sequences to {filepath}")
    
    @staticmethod
    def load_sequences(filepath: Path, format: str = 'pickle') -> List[SequenceData]:
        """Load sequences from file."""
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                sequences = pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sequences = []
            for item in data:
                seq = SequenceData(
                    sequence=item['sequence'],
                    states=np.array(item['states']),
                    state_probs=np.array(item['state_probs']) if item['state_probs'] else None,
                    length=item['length'],
                    alpha_helix_fraction=item['alpha_helix_fraction']
                )
                sequences.append(seq)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded {len(sequences)} sequences from {filepath}")
        return sequences
