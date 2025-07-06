import numpy as np
import json

class HMM:
    def __init__(self, trans_path, emit_path):
        with open(trans_path) as f:
            self.transitions = json.load(f)
        with open(emit_path) as f:
            self.emissions = json.load(f)
        self.states = list(self.transitions.keys())
        self.aa_list = list("ARNDCEQGHILKMFPSTWYV")

    def sample_sequence(self, length):
        state_seq = []
        aa_seq = []
        state = "other"
        for _ in range(length):
            state_seq.append(state)
            probs = self.emissions[state]
            aa = np.random.choice(self.aa_list, p=list(probs.values()))
            aa_seq.append(aa)
            state = np.random.choice(self.states, p=[
                self.transitions[state][s] for s in self.states
            ])
        return aa_seq, state_seq