from scripts.simulate_sequences import simulate_and_save
from scripts.run_viterbi import run_viterbi_on_simulated
from scripts.train_bayesflow import train_bayesflow_model
from scripts.evaluate_predictions import evaluate_model

if __name__ == '__main__':
    simulate_and_save()
    run_viterbi_on_simulated()
    train_bayesflow_model()
    evaluate_model()