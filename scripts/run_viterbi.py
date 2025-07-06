import numpy as np
from models.hmm_model import HMM
from models.viterbi_decoder import ViterbiDecoder

def run_viterbi_on_simulated():
    seqs = np.load("data/sequences/simulated_sequences.npy", allow_pickle=True)
    hmm = HMM("data/transition_probabilities.json", "data/emission_probabilities.json")
    decoder = ViterbiDecoder(hmm)
    preds = [decoder.decode(seq) for seq in seqs]
    np.save("data/sequences/viterbi_predictions.npy", preds)