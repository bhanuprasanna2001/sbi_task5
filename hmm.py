import numpy as np
import random
from hmmlearn import hmm

# Define states and amino acids
states = ['H', 'O']  # H = helix, O = other
amino_acids = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
index_to_state = {0: 'H', 1: 'O'}

# Transition matrix: rows = from state, cols = to state
# Order: H=0, O=1
transition_matrix = np.array([
    [0.90, 0.10],  # From H
    [0.05, 0.95]   # From O
])

# Emission probabilities
emission_probs = {
    'H': {
        'A': 0.12, 'R': 0.06, 'N': 0.03, 'D': 0.05, 'C': 0.01,
        'E': 0.09, 'Q': 0.05, 'G': 0.04, 'H': 0.02, 'I': 0.07,
        'L': 0.12, 'K': 0.06, 'M': 0.03, 'F': 0.04, 'P': 0.02,
        'S': 0.05, 'T': 0.04, 'W': 0.01, 'Y': 0.03, 'V': 0.06
    },
    'O': {
        'A': 0.06, 'R': 0.05, 'N': 0.05, 'D': 0.06, 'C': 0.02,
        'E': 0.05, 'Q': 0.03, 'G': 0.09, 'H': 0.03, 'I': 0.05,
        'L': 0.08, 'K': 0.06, 'M': 0.02, 'F': 0.04, 'P': 0.06,
        'S': 0.07, 'T': 0.06, 'W': 0.01, 'Y': 0.04, 'V': 0.07
    }
}

# Convert emission probabilities to matrix for hmmlearn
emission_matrix = np.array([
    [emission_probs['H'][aa] for aa in amino_acids],  # Row 0: for H
    [emission_probs['O'][aa] for aa in amino_acids]   # Row 1: for O
])

# Initial state probabilities: always start in 'O' (index 1)
start_prob = np.array([0.0, 1.0])

# Sample from a dictionary distribution
def sample_from_dict(dist_dict):
    labels = list(dist_dict.keys())
    weights = [dist_dict[label] for label in labels]
    return random.choices(labels, weights=weights, k=1)[0]

# Sample from weights list directly
def sample_from_weights(weights, labels):
    return random.choices(labels, weights=weights, k=1)[0]

# Simulate sequence from HMM
def simulate_sequence(length=50):
    current_state = 'O'
    states_seq = []
    observed_seq = []

    for _ in range(length):
        # Emit amino acid
        aa = sample_from_dict(emission_probs[current_state])
        observed_seq.append(aa)
        states_seq.append(current_state)

        # Transition to next state
        if current_state == 'H':
            current_state = sample_from_weights([0.90, 0.10], ['H', 'O'])
        else:
            current_state = sample_from_weights([0.05, 0.95], ['H', 'O'])

    return observed_seq, states_seq

# Simulate data
obs_seq, true_states = simulate_sequence(length=50)

# One-hot encode amino acids for MultinomialHMM
obs_seq_onehot = np.zeros((len(obs_seq), len(amino_acids)), dtype=int)
for i, aa in enumerate(obs_seq):
    aa_index = aa_to_index[aa]
    obs_seq_onehot[i, aa_index] = 1

# Define HMM model
model = hmm.MultinomialHMM(n_components=2, init_params="", n_trials=1)
model.startprob_ = start_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

# Decode sequence using Viterbi algorithm
logprob, pred_state_indices = model.decode(obs_seq_onehot, algorithm="viterbi")
pred_states = [index_to_state[i] for i in pred_state_indices]

# --- Output ---
print("\n🔬 Simulated Amino Acid Sequence:\n", ''.join(obs_seq))
print("\n✅ True Hidden States:\n", ''.join(true_states))
print("\n🔍 Viterbi Predicted States:\n", ''.join(pred_states))
