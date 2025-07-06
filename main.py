from scripts.simulate_sequences import simulate_and_save
from scripts.run_viterbi import run_viterbi_on_simulated
from scripts.train_bayesflow import train_bayesflow_model
from scripts.evaluate_predictions import evaluate_model

if __name__ == '__main__':
    print("=== Starting Protein Structure Inference Pipeline ===\n")

    print("➡️  Step 1: Simulating amino acid sequences...")
    simulate_and_save()
    print("✅ Simulation complete.\n")

    print("➡️  Step 2: Running Viterbi decoding on sequences...")
    run_viterbi_on_simulated()
    print("✅ Viterbi decoding complete.\n")

    print("➡️  Step 3: Training BayesFlow model...")
    train_bayesflow_model()
    print("✅ BayesFlow model training complete.\n")

    print("➡️  Step 4: Evaluating model predictions...")
    evaluate_model()
    print("✅ Evaluation complete.\n")