import matplotlib.pyplot as plt
import numpy as np

def plot_sequence(seq, states):
    color_map = {'alpha-helix': 'red', 'other': 'blue'}
    colors = [color_map[s] for s in states]
    plt.figure(figsize=(12, 1))
    plt.bar(range(len(seq)), [1]*len(seq), color=colors)
    plt.xticks(range(len(seq)), seq)
    plt.title("Amino Acid Sequence and States")
    plt.show()
