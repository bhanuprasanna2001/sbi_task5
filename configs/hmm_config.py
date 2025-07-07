"""
Configuration settings for protein secondary structure HMM.
Contains fixed emission and transition probabilities from task description.
"""

import numpy as np

# Amino acid alphabet (20 standard amino acids)
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# State definitions
STATES = ['other', 'alpha_helix']  # 0: other, 1: alpha_helix
N_STATES = len(STATES)
N_AMINO_ACIDS = len(AMINO_ACIDS)

# Emission probabilities from task description
# Alpha-helix state probabilities (in order of AMINO_ACIDS)
ALPHA_HELIX_EMISSIONS = np.array([
    0.12, 0.06, 0.03, 0.05, 0.01, 0.09, 0.05, 0.04, 0.02, 0.07,  # A-I
    0.12, 0.06, 0.03, 0.04, 0.02, 0.05, 0.04, 0.01, 0.03, 0.06   # L-V
])

# Other state probabilities (in order of AMINO_ACIDS)
OTHER_EMISSIONS = np.array([
    0.06, 0.05, 0.05, 0.06, 0.02, 0.05, 0.03, 0.09, 0.03, 0.05,  # A-I
    0.08, 0.06, 0.02, 0.04, 0.06, 0.07, 0.06, 0.01, 0.04, 0.07   # L-V
])

# Emission probability matrix (states x amino_acids)
EMISSION_PROBS = np.array([OTHER_EMISSIONS, ALPHA_HELIX_EMISSIONS])

# Transition probabilities from task description
# From other state: other=0.95, alpha_helix=0.05
# From alpha_helix state: other=0.10, alpha_helix=0.90
TRANSITION_PROBS = np.array([
    [0.95, 0.05],  # from other state
    [0.10, 0.90]   # from alpha_helix state
])

# Initial state probabilities (always start in "other" state)
INITIAL_PROBS = np.array([1.0, 0.0])  # [other, alpha_helix]

# Data generation settings
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_N_SEQUENCES = 1000

# Training settings
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

# Validation settings
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 500

# Output settings
RESULTS_PRECISION = 4
