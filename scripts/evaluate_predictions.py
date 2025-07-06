import numpy as np
from utils.metrics import compute_accuracy

def evaluate_model():
    gt = np.load("data/sequences/ground_truth_labels.npy", allow_pickle=True)
    pred = np.load("data/sequences/viterbi_predictions.npy", allow_pickle=True)
    flat_gt = [s for seq in gt for s in seq]
    flat_pred = [s for seq in pred for s in seq]
    print("Accuracy:", compute_accuracy(flat_gt, flat_pred))