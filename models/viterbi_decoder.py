import numpy as np
from utils.amino_acid_utils import aa_index

class ViterbiDecoder:
    def __init__(self, hmm):
        self.hmm = hmm

    def decode(self, sequence):
        n = len(sequence)
        m = len(self.hmm.states)
        T = np.zeros((m, n))
        path = np.zeros((m, n), dtype=int)
        state_idx = {s: i for i, s in enumerate(self.hmm.states)}
        aa_idx = aa_index()

        for i, state in enumerate(self.hmm.states):
            T[i, 0] = self.hmm.transitions['other'][state] * self.hmm.emissions[state][sequence[0]]

        for t in range(1, n):
            for j, state in enumerate(self.hmm.states):
                max_prob = -1
                for i_prev, prev_state in enumerate(self.hmm.states):
                    prob = T[i_prev, t - 1] * self.hmm.transitions[prev_state][state] * self.hmm.emissions[state][sequence[t]]
                    if prob > max_prob:
                        max_prob = prob
                        path[j, t] = i_prev
                        T[j, t] = prob

        best_path = np.zeros(n, dtype=int)
        best_path[-1] = np.argmax(T[:, -1])
        for t in range(n - 2, -1, -1):
            best_path[t] = path[best_path[t + 1], t + 1]

        idx_state = {i: s for s, i in state_idx.items()}
        return [idx_state[i] for i in best_path]