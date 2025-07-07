"""
Advanced BayesFlow diagnostics for protein secondary structure inference.
Implements proper calibration, posterior contraction, and NRMSE metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import bayesflow as bf


class AdvancedDiagnostics:
    """Advanced diagnostic tools for BayesFlow protein HMM inference."""
    
    def __init__(self):
        """Initialize advanced diagnostics."""
        pass
    
    def compute_posterior_contraction(self, 
                                    posterior_samples: np.ndarray,
                                    prior_samples: np.ndarray) -> float:
        """
        Compute posterior contraction metric.
        
        Posterior contraction = 1 - (Var[posterior] / Var[prior])
        
        A value close to 1 indicates strong learning from data.
        A value close to 0 indicates little learning.
        """
        posterior_var = np.var(posterior_samples, axis=0)
        prior_var = np.var(prior_samples, axis=0)
        
        # Handle division by zero and very small variances with warnings suppressed
        with np.errstate(divide='ignore', invalid='ignore'):
            contraction = np.where(prior_var > 1e-10, 
                                 1 - (posterior_var / prior_var), 
                                 0.0)
        
        # Ensure contraction is finite
        contraction = np.where(np.isfinite(contraction), contraction, 0.0)
        
        # Return mean contraction across all dimensions
        return float(np.mean(contraction))
    
    def compute_calibration_error(self, 
                                posterior_samples: np.ndarray,
                                true_values: np.ndarray,
                                coverage_levels: List[float] = [0.5, 0.8, 0.9, 0.95]) -> Dict[str, float]:
        """
        Compute calibration error using coverage diagnostics.
        
        Well-calibrated posteriors should have empirical coverage
        matching nominal coverage levels.
        """
        calibration_errors = {}
        
        for coverage in coverage_levels:
            alpha = 1 - coverage
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            # Compute empirical coverage
            lower_bounds = np.quantile(posterior_samples, lower_quantile, axis=0)
            upper_bounds = np.quantile(posterior_samples, upper_quantile, axis=0)
            
            # Check if true values fall within bounds
            within_bounds = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            empirical_coverage = np.mean(within_bounds)
            
            # Calibration error is absolute difference from nominal coverage
            calibration_error = abs(empirical_coverage - coverage)
            calibration_errors[f'{int(coverage*100)}%'] = calibration_error
        
        # Return mean calibration error across all coverage levels
        mean_error = np.mean(list(calibration_errors.values()))
        calibration_errors['mean'] = mean_error
        
        return calibration_errors
    
    def compute_nrmse(self, 
                     posterior_samples: np.ndarray,
                     true_values: np.ndarray) -> float:
        """
        Compute Normalized Root Mean Square Error (NRMSE).
        
        NRMSE = RMSE / (max(true_values) - min(true_values))
        """
        # Use posterior mean as point estimate
        posterior_mean = np.mean(posterior_samples, axis=0)
        
        # Compute RMSE
        mse = np.mean((posterior_mean - true_values) ** 2)
        rmse = np.sqrt(mse)
        
        # Normalize by range of true values
        true_range = np.max(true_values) - np.min(true_values)
        
        # Handle zero variance case (all true values are the same)
        if true_range < 1e-10:
            # If there's no variation in true values, NRMSE is undefined
            # Return 0 if RMSE is also very small, otherwise inf
            if rmse < 1e-10:
                nrmse = 0.0
            else:
                nrmse = np.inf
        else:
            nrmse = rmse / true_range
        
        return nrmse
    
    def compute_comprehensive_diagnostics(self,
                                        workflow: bf.BasicWorkflow,
                                        test_data: Dict,
                                        num_posterior_samples: int = 1000,
                                        num_prior_samples: int = 1000) -> pd.DataFrame:
        """
        Compute comprehensive diagnostics including all key metrics.
        """
        print("Computing comprehensive BayesFlow diagnostics...")
        
        # Generate posterior samples
        print("Sampling from posterior...")
        posterior_samples = workflow.sample(
            conditions=test_data, 
            num_samples=num_posterior_samples
        )
        
        # Generate prior samples for comparison
        print("Sampling from prior...")
        prior_data = workflow.simulate(num_prior_samples)
        
        # Handle different possible keys in posterior samples
        if 'inference_variables' in posterior_samples:
            posterior_key = 'inference_variables'
        elif 'state_probs' in posterior_samples:
            posterior_key = 'state_probs'
        else:
            posterior_key = list(posterior_samples.keys())[0]
        
        # Get shapes
        n_test = len(test_data['state_probs'])
        seq_len = test_data['state_probs'].shape[1] // 2  # Assuming flattened (seq_len * 2)
        
        print(f"Analyzing {n_test} test sequences of length {seq_len}")
        
        # Compute diagnostics for each sequence position
        results = []
        
        for seq_idx in range(min(n_test, 50)):  # Limit to first 50 sequences for efficiency
            for pos_idx in range(seq_len):
                # Extract true state probabilities for this position
                true_probs = test_data['state_probs'][seq_idx].reshape(seq_len, 2)[pos_idx]
                
                # Extract posterior samples for this position
                post_samples = posterior_samples[posterior_key][seq_idx]  # (num_samples, seq_len*2)
                post_samples_reshaped = post_samples.reshape(num_posterior_samples, seq_len, 2)
                post_samples_pos = post_samples_reshaped[:, pos_idx, :]  # (num_samples, 2)
                
                # Extract prior samples for comparison
                prior_samples_all = prior_data['state_probs']  # (num_prior_samples, seq_len*2)
                prior_samples_reshaped = prior_samples_all.reshape(num_prior_samples, seq_len, 2)
                prior_samples_pos = prior_samples_reshaped[:, pos_idx, :]  # (num_samples, 2)
                
                # Compute metrics for alpha-helix probability (state 1)
                posterior_alpha = post_samples_pos[:, 1]
                prior_alpha = prior_samples_pos[:, 1]
                true_alpha = true_probs[1]
                
                # Posterior contraction
                contraction = self.compute_posterior_contraction(
                    posterior_alpha.reshape(-1, 1), 
                    prior_alpha.reshape(-1, 1)
                )
                
                # Calibration error
                calib_error = self.compute_calibration_error(
                    posterior_alpha.reshape(-1, 1),
                    np.array([true_alpha])
                )
                
                # NRMSE
                nrmse = self.compute_nrmse(
                    posterior_alpha.reshape(-1, 1),
                    np.array([true_alpha])
                )
                
                results.append({
                    'sequence_idx': seq_idx,
                    'position_idx': pos_idx,
                    'true_alpha_prob': true_alpha,
                    'posterior_mean_alpha': np.mean(posterior_alpha),
                    'posterior_std_alpha': np.std(posterior_alpha),
                    'posterior_contraction': contraction,
                    'calibration_error_mean': calib_error['mean'],
                    'nrmse': nrmse
                })
        
        results_df = pd.DataFrame(results)
        
        # Filter out infinite values for meaningful statistics
        finite_nrmse = results_df['nrmse'][np.isfinite(results_df['nrmse'])]
        infinite_nrmse_count = (~np.isfinite(results_df['nrmse'])).sum()
        
        # Compute aggregate statistics
        print("\n=== COMPREHENSIVE DIAGNOSTICS SUMMARY ===")
        print(f"Posterior Contraction (mean): {results_df['posterior_contraction'].mean():.6f}")
        print(f"Calibration Error (mean): {results_df['calibration_error_mean'].mean():.6f}")
        
        if len(finite_nrmse) > 0:
            print(f"NRMSE (mean, finite values): {finite_nrmse.mean():.6f}")
            print(f"NRMSE (median, finite values): {finite_nrmse.median():.6f}")
        else:
            print("NRMSE: All values are infinite (no variance in true values)")
        
        if infinite_nrmse_count > 0:
            print(f"NRMSE infinite values: {infinite_nrmse_count}/{len(results_df)} ({100*infinite_nrmse_count/len(results_df):.1f}%)")
        
        # Count problematic cases
        low_contraction = (results_df['posterior_contraction'] < 0.1).sum()
        high_calib_error = (results_df['calibration_error_mean'] > 0.1).sum()
        high_nrmse = (finite_nrmse > 1.0).sum() if len(finite_nrmse) > 0 else 0
        
        print(f"\nProblematic cases:")
        print(f"  Low posterior contraction (<0.1): {low_contraction}/{len(results_df)} ({100*low_contraction/len(results_df):.1f}%)")
        print(f"  High calibration error (>0.1): {high_calib_error}/{len(results_df)} ({100*high_calib_error/len(results_df):.1f}%)")
        if len(finite_nrmse) > 0:
            print(f"  High NRMSE (>1.0): {high_nrmse}/{len(finite_nrmse)} ({100*high_nrmse/len(finite_nrmse):.1f}% of finite values)")
        
        return results_df
    
    def plot_diagnostic_summary(self, 
                               results_df: pd.DataFrame,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """Plot comprehensive diagnostic summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Posterior Contraction Distribution
        ax1.hist(results_df['posterior_contraction'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results_df['posterior_contraction'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {results_df["posterior_contraction"].mean():.3f}')
        ax1.set_xlabel('Posterior Contraction')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Posterior Contraction Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Calibration Error Distribution
        ax2.hist(results_df['calibration_error_mean'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(results_df['calibration_error_mean'].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df["calibration_error_mean"].mean():.3f}')
        ax2.set_xlabel('Calibration Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Calibration Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. NRMSE Distribution (log scale)
        nrmse_finite = results_df['nrmse'][np.isfinite(results_df['nrmse'])]
        
        if len(nrmse_finite) > 0 and nrmse_finite.min() > 0:
            # Only plot if we have finite, positive values
            with np.errstate(divide='ignore', invalid='ignore'):
                log_nrmse = np.log10(nrmse_finite)
                log_nrmse = log_nrmse[np.isfinite(log_nrmse)]
            
            if len(log_nrmse) > 0:
                ax3.hist(log_nrmse, bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax3.axvline(np.mean(log_nrmse), color='red', linestyle='--',
                           label=f'Mean: {nrmse_finite.mean():.3f}')
                ax3.set_xlabel('log10(NRMSE)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('NRMSE Distribution (log scale)')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No positive finite\nNRMSE values\nto plot', 
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat'))
                ax3.set_title('NRMSE Distribution (log scale)')
        else:
            # No finite values to plot
            ax3.text(0.5, 0.5, 'All NRMSE values\nare infinite\n(no variance in data)', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat'))
            ax3.set_title('NRMSE Distribution (log scale)')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. Posterior Mean vs True Values
        ax4.scatter(results_df['true_alpha_prob'], results_df['posterior_mean_alpha'], 
                   alpha=0.6, color='purple', s=20)
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Prediction')
        ax4.set_xlabel('True Alpha-helix Probability')
        ax4.set_ylabel('Posterior Mean Alpha-helix Probability')
        ax4.set_title('Posterior Mean vs True Values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Diagnostic plot saved to {save_path}")
        
        return fig
    
    def diagnose_training_issues(self, results_df: pd.DataFrame) -> Dict[str, str]:
        """Analyze diagnostic results and suggest fixes for training issues."""
        issues = {}
        suggestions = {}
        
        mean_contraction = results_df['posterior_contraction'].mean()
        mean_calib_error = results_df['calibration_error_mean'].mean()
        mean_nrmse = results_df['nrmse'].mean()
        
        # Analyze posterior contraction
        if mean_contraction < 0.1:
            issues['posterior_contraction'] = "Very low - model not learning from data"
            suggestions['posterior_contraction'] = [
                "Increase training epochs significantly",
                "Reduce learning rate for more stable training",
                "Check if loss is decreasing during training",
                "Verify data preprocessing and adapter configuration",
                "Consider larger network capacity"
            ]
        elif mean_contraction < 0.3:
            issues['posterior_contraction'] = "Low - limited learning from data"
            suggestions['posterior_contraction'] = [
                "Train for more epochs",
                "Monitor training loss convergence",
                "Consider network architecture adjustments"
            ]
        else:
            issues['posterior_contraction'] = "Good - model learning from data"
            suggestions['posterior_contraction'] = ["Continue current training approach"]
        
        # Analyze calibration error
        if mean_calib_error > 0.2:
            issues['calibration_error'] = "Very high - poor uncertainty quantification"
            suggestions['calibration_error'] = [
                "Use Flow Matching or Normalizing Flows",
                "Increase posterior sampling",
                "Check adapter transformations",
                "Consider simulation-based calibration training"
            ]
        elif mean_calib_error > 0.1:
            issues['calibration_error'] = "High - uncertainty underestimated"
            suggestions['calibration_error'] = [
                "Train longer for better calibration",
                "Increase network regularization",
                "Check posterior sampling procedure"
            ]
        else:
            issues['calibration_error'] = "Good - well-calibrated posteriors"
            suggestions['calibration_error'] = ["Maintain current calibration approach"]
        
        # Analyze NRMSE
        if not np.isfinite(mean_nrmse) or mean_nrmse > 2.0:
            issues['nrmse'] = "Very high/infinite - prediction completely inaccurate"
            suggestions['nrmse'] = [
                "Check data preprocessing and scaling",
                "Verify simulator and adapter compatibility",
                "Restart training with simpler architecture",
                "Check for NaN/inf values in training data"
            ]
        elif mean_nrmse > 1.0:
            issues['nrmse'] = "High - poor prediction accuracy"
            suggestions['nrmse'] = [
                "Train for more epochs",
                "Increase model capacity",
                "Check training data quality"
            ]
        else:
            issues['nrmse'] = "Good - accurate predictions"
            suggestions['nrmse'] = ["Maintain current prediction accuracy"]
        
        return {'issues': issues, 'suggestions': suggestions}


def create_training_recommendations(diagnostics_results: pd.DataFrame) -> Dict[str, Union[str, List[str]]]:
    """Create specific training recommendations based on diagnostic results."""
    
    advanced_diag = AdvancedDiagnostics()
    analysis = advanced_diag.diagnose_training_issues(diagnostics_results)
    
    recommendations = {
        'priority': 'HIGH',
        'status': 'NEEDS_IMPROVEMENT',
        'immediate_actions': [],
        'training_adjustments': [],
        'architecture_changes': []
    }
    
    # Determine priority based on severity of issues
    mean_contraction = diagnostics_results['posterior_contraction'].mean()
    mean_calib_error = diagnostics_results['calibration_error_mean'].mean()
    
    if mean_contraction < 0.1 or mean_calib_error > 0.2:
        recommendations['priority'] = 'CRITICAL'
        recommendations['status'] = 'TRAINING_FAILED'
        recommendations['immediate_actions'] = [
            "Stop current training approach",
            "Review data preprocessing pipeline",
            "Verify BayesFlow adapter configuration",
            "Check for data leakage or preprocessing errors"
        ]
    elif mean_contraction < 0.3 or mean_calib_error > 0.1:
        recommendations['priority'] = 'HIGH'
        recommendations['immediate_actions'] = [
            "Increase training epochs to 50-100",
            "Monitor training loss convergence",
            "Implement early stopping based on validation loss"
        ]
    
    # Training adjustments
    recommendations['training_adjustments'] = [
        "Use larger dataset (10k-50k samples)",
        "Train for 50-100 epochs minimum",
        "Implement learning rate scheduling",
        "Use validation-based early stopping",
        "Monitor both training and validation loss"
    ]
    
    # Architecture changes if needed
    if mean_contraction < 0.2:
        recommendations['architecture_changes'] = [
            "Increase summary network capacity",
            "Use bidirectional LSTM with more units",
            "Consider attention mechanisms for sequence modeling",
            "Increase Flow Matching network depth"
        ]
    
    return recommendations
