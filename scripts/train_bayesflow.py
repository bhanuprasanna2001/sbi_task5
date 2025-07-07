#!/usr/bin/env python3
"""
BayesFlow training script for protein secondary structure inference.
Follows the patterns from BayesFlow examples (Two Moons, SIR, Linear Regression).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from pathlib import Path

# Set Keras backend before importing
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import bayesflow as bf
import tensorflow as tf
from src.bayesflow.bayesflow_simulator import create_protein_simulator_functions
from src.utils.visualization import OutputManager, calculate_accuracy_metrics
from src.utils.advanced_diagnostics import AdvancedDiagnostics, create_training_recommendations
from src.validation.insulin_validation import InsulinValidator


def main():
    """Main training function following BayesFlow examples."""
    
    print("="*60)
    print("BAYESFLOW PROTEIN SECONDARY STRUCTURE TRAINING")
    print("="*60)
    
    # Initialize output manager and create run directory
    output_manager = OutputManager()
    run_dir = output_manager.create_run_directory()
    
    print(f"📁 Output directory: {run_dir}")
    
    # Configuration - Full-scale production training
    config = {
        "sequence_length": 100,
        "num_total_samples": 20000,
        "train_ratio": 0.7,
        "val_ratio": 0.15, 
        "test_ratio": 0.15,
        "batch_size": 64,  # Larger batch size for better training
        "epochs": 50,  # Full training epochs
        "num_posterior_samples": 2000,  # More posterior samples for better diagnostics
        "learning_rate": 1e-4,  # Explicit learning rate
        "early_stopping_patience": 15,  # Early stopping
        "timestamp": datetime.datetime.now().isoformat(),
        "keras_backend": os.environ.get("KERAS_BACKEND", "tensorflow")
    }
    
    # Save configuration
    output_manager.save_run_config(config)
    
    print(f"Configuration:")
    for key, value in config.items():
        if key != "timestamp":
            print(f"  {key}: {value}")
    
    sequence_length = config["sequence_length"]
    num_total_samples = config["num_total_samples"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
    # Calculate split sizes
    num_train = int(num_total_samples * train_ratio)
    num_val = int(num_total_samples * val_ratio)
    num_test = int(num_total_samples * test_ratio)
    
    print(f"Data split: Train={num_train}, Val={num_val}, Test={num_test}")
    
    # Create BayesFlow simulator
    print("\nCreating BayesFlow simulator...")
    simulator_functions = create_protein_simulator_functions(
        fixed_length=True, 
        seq_length=sequence_length
    )
    simulator = bf.make_simulator(simulator_functions)
    
    # Test simulator
    print("Testing simulator...")
    test_sample = simulator.sample(3)
    print(f"Simulator output keys: {list(test_sample.keys())}")
    print(f"Sequence shape: {test_sample['sequence'].shape}")
    print(f"State probs shape: {test_sample['state_probs'].shape}")
    
    # Create adapter
    print("\nCreating data adapter...")
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("state_probs", "inference_variables")
        .rename("sequence", "summary_variables")
        .expand_dims("summary_variables", axis=-1)  # Add feature dimension for embedding
    )
    
    # Test adapter
    adapted_sample = adapter(test_sample)
    print(f"Adapted keys: {list(adapted_sample.keys())}")
    print(f"Summary variables shape: {adapted_sample['summary_variables'].shape}")
    print(f"Inference variables shape: {adapted_sample['inference_variables'].shape}")
    
    # Generate data with proper train/test/val split
    print("\nGenerating complete dataset...")
    full_data = simulator.sample(num_total_samples)
    
    # Split data
    training_data = {
        'sequence': full_data['sequence'][:num_train],
        'state_probs': full_data['state_probs'][:num_train]
    }
    validation_data = {
        'sequence': full_data['sequence'][num_train:num_train+num_val],
        'state_probs': full_data['state_probs'][num_train:num_train+num_val]
    }
    test_data = {
        'sequence': full_data['sequence'][num_train+num_val:],
        'state_probs': full_data['state_probs'][num_train+num_val:]
    }
    
    print(f"Training data: {training_data['sequence'].shape[0]} samples")
    print(f"Validation data: {validation_data['sequence'].shape[0]} samples")
    print(f"Test data: {test_data['sequence'].shape[0]} samples")
    
    # Create inference network with improved architecture
    print("\nCreating inference network...")
    # Use Flow Matching with larger, more expressive networks
    inference_network = bf.networks.FlowMatching(
        subnet="mlp",
        subnet_kwargs={
            "dropout": 0.15,  # Slightly more regularization 
            "widths": (256, 256, 128, 64)  # Deeper, wider network
        }
    )
    
    # Create summary network with more capacity for sequences
    summary_network = bf.networks.DeepSet(
        dense_kwargs={
            "dropout": 0.1,
            "widths": (128, 256, 128)  # Increased capacity
        }
    )
    
    # Create workflow with explicit learning rate
    print("Creating BayesFlow workflow...")
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network
    )
    
    # Compile with explicit optimizer settings
    workflow.approximator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    )
    
    # Train with early stopping and monitoring
    print(f"\nStarting training for up to {epochs} epochs...")
        
    history = workflow.fit_offline(
        data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=1
    )
    
    print("✓ Training completed!")
    
    # Plot training history
    print("\nPlotting training results...")
    f = bf.diagnostics.plots.loss(history, figsize=(12, 4))
    plt.suptitle("Protein HMM BayesFlow Training")
    plt.tight_layout()
    
    # Save plot
    plot_path = output_manager.get_figure_path("training_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close instead of show
    print(f"📊 Training plot saved: {plot_path}")
    
    # Validation using proper test split
    print("\nRunning posterior sampling on test data...")
    
    # Sample from posterior on test data
    posterior_samples = workflow.sample(conditions=test_data, num_samples=config["num_posterior_samples"])
    
    print(f"Posterior samples keys: {list(posterior_samples.keys())}")
    
    # Handle different possible key structures
    if 'inference_variables' in posterior_samples:
        posterior_key = 'inference_variables'
    elif 'state_probs' in posterior_samples:
        posterior_key = 'state_probs'
    else:
        # Use the first available key
        posterior_key = list(posterior_samples.keys())[0]
        print(f"Using key: {posterior_key}")
    
    print(f"Posterior samples shape: {posterior_samples[posterior_key].shape}")
    
    # Save all data splits
    print("\nSaving complete dataset...")
    data_path = output_manager.get_data_path("complete_dataset.npz")
    np.savez(data_path, 
             train_sequences=training_data['sequence'],
             train_state_probs=training_data['state_probs'],
             val_sequences=validation_data['sequence'],
             val_state_probs=validation_data['state_probs'],
             test_sequences=test_data['sequence'],
             test_state_probs=test_data['state_probs'])
    print(f"💾 Data saved: {data_path}")
    
    # ADVANCED BAYESFLOW DIAGNOSTICS
    print("\n" + "="*60)
    print("ADVANCED BAYESFLOW DIAGNOSTICS")
    print("="*60)
    
    # Initialize advanced diagnostics
    advanced_diag = AdvancedDiagnostics()
    
    # Compute comprehensive diagnostics
    diagnostic_results = advanced_diag.compute_comprehensive_diagnostics(
        workflow=workflow,
        test_data=test_data,
        num_posterior_samples=config["num_posterior_samples"],
        num_prior_samples=2000
    )
    
    # Save diagnostic results
    diag_path = output_manager.get_report_path("advanced_diagnostics.csv")
    diagnostic_results.to_csv(diag_path, index=False)
    print(f"📊 Advanced diagnostics saved: {diag_path}")
    
    # Create diagnostic plots
    diag_plot_path = output_manager.get_figure_path("advanced_diagnostics.png")
    advanced_diag.plot_diagnostic_summary(diagnostic_results, diag_plot_path)
    plt.close()
    
    # Generate training recommendations
    recommendations = create_training_recommendations(diagnostic_results)
    
    # Save recommendations
    rec_path = output_manager.get_report_path("training_recommendations.json")
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"📋 Training recommendations saved: {rec_path}")
    
    # Print recommendations summary
    print(f"\n🎯 TRAINING ASSESSMENT:")
    print(f"   Priority: {recommendations['priority']}")
    print(f"   Status: {recommendations['status']}")
    if recommendations['immediate_actions']:
        print(f"   Immediate Actions Required:")
        for action in recommendations['immediate_actions']:
            print(f"     • {action}")
    
    print("\n" + "="*60)
    
    # HUMAN INSULIN VALIDATION
    print("\n" + "="*60)
    print("TASK 5 REQUIREMENT: HUMAN INSULIN VALIDATION")
    print("="*60)
    
    # Initialize insulin validator
    insulin_validator = InsulinValidator()
    
    # Run validation
    insulin_results = insulin_validator.validate_model(workflow, output_manager)
    
    # Compare with literature benchmarks
    benchmarks = insulin_validator.compare_with_literature()
    
    print(f"\n📚 LITERATURE COMPARISON:")
    accuracy = insulin_results['accuracy_metrics']['overall_accuracy']
    
    for method, bench in benchmarks.items():
        if bench['min'] <= accuracy <= bench['max']:
            print(f"  ✅ {bench['description']}: {accuracy:.3f} (within range {bench['min']:.2f}-{bench['max']:.2f})")
        else:
            print(f"  ❌ {bench['description']}: {accuracy:.3f} (outside range {bench['min']:.2f}-{bench['max']:.2f})")
    
    print("\n" + "="*60)
    
    # Custom model evaluation (skip BayesFlow diagnostics)
    print("\nEvaluating model accuracy...")
    
    # Collect accuracy metrics for all test samples
    all_accuracy_metrics = []
    individual_mse_scores = []
    
    for i in range(min(50, len(test_data['state_probs']))):  # Evaluate first 50 samples
        true_state_probs = test_data['state_probs'][i].reshape(-1, 2)  # Reshape back to (seq_len, 2)
        pred_samples = posterior_samples[posterior_key][i]  # (num_samples, seq_len*2)
        
        # Reshape predictions
        pred_reshaped = pred_samples.reshape(pred_samples.shape[0], -1, 2)  # (num_samples, seq_len, 2)
        pred_mean = np.mean(pred_reshaped, axis=0)  # (seq_len, 2)
        
        # Calculate MSE
        mse = np.mean((pred_mean - true_state_probs)**2)
        individual_mse_scores.append(mse)
        
        # Convert probabilities to discrete states for accuracy calculation
        true_states = np.argmax(true_state_probs, axis=1)  # Most likely state
        pred_states = np.argmax(pred_mean, axis=1)  # Most likely predicted state
        
        # Calculate accuracy metrics for this sequence
        accuracy_metrics = calculate_accuracy_metrics(true_states, pred_states)
        all_accuracy_metrics.append(accuracy_metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    metric_keys = all_accuracy_metrics[0].keys()
    for key in metric_keys:
        if key in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            avg_metrics[key] = int(np.mean([m[key] for m in all_accuracy_metrics]))
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_accuracy_metrics])
    
    avg_mse = np.mean(individual_mse_scores)
    
    # Print results
    print(f"\n🎯 ACCURACY RESULTS (evaluated on {len(all_accuracy_metrics)} sequences):")
    print(f"  Overall Accuracy: {avg_metrics['overall_accuracy']:.4f}")
    print(f"  Alpha-helix Accuracy: {avg_metrics['alpha_helix_accuracy']:.4f}")
    print(f"  Other Structure Accuracy: {avg_metrics['other_accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"  F1-Score: {avg_metrics['f1_score']:.4f}")
    print(f"  Average MSE: {avg_mse:.6f}")
    
    # Save detailed results
    results = {
        "config": config,
        "training_history": {
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "best_val_loss": float(min(history.history['val_loss'])),
            "epochs_trained": len(history.history['loss'])
        },
        "accuracy_metrics": avg_metrics,
        "mse_scores": {
            "average": float(avg_mse),
            "std": float(np.std(individual_mse_scores)),
            "individual": [float(x) for x in individual_mse_scores]
        },
        "insulin_validation": insulin_results
    }
    
    # Save results to JSON
    results_path = output_manager.get_report_path("results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📊 Results saved: {results_path}")
    
    # Model-specific plots inspired by BayesFlow examples
    print("\nCreating model-specific visualizations...")
    
    # 1. Posterior vs Prior comparison (similar to Two Moons example)
    fig_model, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sample predictions vs truth for a few sequences
    sample_idx = 0
    true_probs = test_data['state_probs'][sample_idx].reshape(-1, 2)
    pred_samples_reshaped = posterior_samples[posterior_key][sample_idx].reshape(-1, sequence_length, 2)
    pred_mean = np.mean(pred_samples_reshaped, axis=0)
    pred_std = np.std(pred_samples_reshaped, axis=0)
    
    positions = range(sequence_length)
    ax1.plot(positions, true_probs[:, 1], 'r-', label='True Alpha-helix Prob', linewidth=2)
    ax1.plot(positions, pred_mean[:, 1], 'b-', label='Predicted Mean', linewidth=2)
    ax1.fill_between(positions, pred_mean[:, 1] - pred_std[:, 1], pred_mean[:, 1] + pred_std[:, 1], 
                     alpha=0.3, color='blue', label='±1 STD')
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Alpha-helix Probability')
    ax1.set_title('Posterior Predictions vs Truth (Sample Sequence)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Posterior uncertainty distribution
    all_uncertainties = []
    for i in range(min(20, len(test_data['state_probs']))):
        pred_samples_i = posterior_samples[posterior_key][i].reshape(-1, sequence_length, 2)
        uncertainties = np.std(pred_samples_i, axis=0)
        all_uncertainties.extend(uncertainties[:, 1])  # Alpha-helix uncertainties
    
    ax2.hist(all_uncertainties, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Posterior Uncertainty (STD)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Posterior Uncertainties')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prior vs Posterior comparison for alpha-helix probabilities
    # Sample from prior (using simulator)
    prior_samples = simulator.sample(100)
    prior_probs = prior_samples['state_probs'].reshape(-1, 2)
    
    # Get posterior samples
    posterior_probs = []
    for i in range(min(10, len(test_data['state_probs']))):
        pred_samples_i = posterior_samples[posterior_key][i].reshape(-1, sequence_length, 2)
        posterior_probs.extend(pred_samples_i[:, :, 1].flatten())  # Alpha-helix probs
    
    ax3.hist(prior_probs[:, 1], bins=50, alpha=0.5, label='Prior', color='red', density=True)
    ax3.hist(posterior_probs[:1000], bins=50, alpha=0.5, label='Posterior', color='blue', density=True)
    ax3.set_xlabel('Alpha-helix Probability')
    ax3.set_ylabel('Density')
    ax3.set_title('Prior vs Posterior Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prediction quality scatter plot
    true_vals = []
    pred_vals = []
    for i in range(min(20, len(test_data['state_probs']))):
        true_probs_i = test_data['state_probs'][i].reshape(-1, 2)
        pred_samples_i = posterior_samples[posterior_key][i].reshape(-1, sequence_length, 2)
        pred_mean_i = np.mean(pred_samples_i, axis=0)
        
        true_vals.extend(true_probs_i[:, 1])  # Alpha-helix true probs
        pred_vals.extend(pred_mean_i[:, 1])   # Alpha-helix predicted probs
    
    ax4.scatter(true_vals, pred_vals, alpha=0.6, color='purple')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Prediction')
    ax4.set_xlabel('True Alpha-helix Probability')
    ax4.set_ylabel('Predicted Alpha-helix Probability')
    ax4.set_title('Prediction Accuracy Scatter Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    model_plot_path = output_manager.get_figure_path("model_analysis.png")
    plt.savefig(model_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Model analysis plot saved: {model_plot_path}")
    
    # Save model
    print("\nSaving trained model...")
    model_path = output_manager.get_model_path("bayesflow_model.keras")
    workflow.approximator.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📁 All outputs saved to: {run_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
