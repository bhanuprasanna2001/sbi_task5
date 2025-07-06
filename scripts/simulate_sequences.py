from models.hmm_model import HMM
import numpy as np
import os

def simulate_and_save():
    hmm = HMM("data/transition_probabilities.json", "data/emission_probabilities.json")
    all_seqs, all_states = [], []
    for _ in range(500):
        seq, states = hmm.sample_sequence(50)
        all_seqs.append(seq)
        all_states.append(states)
    np.save("data/sequences/simulated_sequences.npy", all_seqs)
    np.save("data/sequences/ground_truth_labels.npy", all_states)