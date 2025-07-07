"""
Utility functions for protein secondary structure project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import datetime
import json

from src.data import SequenceData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """Visualization utilities for HMM analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """Initialize visualizer with plotting settings."""
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_sequence_states(self, 
                           sequence_data: SequenceData,
                           title: Optional[str] = None,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot amino acid sequence with state annotations."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        sequence = sequence_data.sequence
        states = sequence_data.states
        positions = range(len(sequence))
        
        # Plot states
        colors = ['lightblue', 'orange']
        state_names = ['Other', 'Alpha-helix']
        ax1.bar(positions, [1]*len(states), color=[colors[s] for s in states], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Hidden States')
        ax1.set_ylim(0, 1.2)
        ax1.set_yticks([0.5])
        ax1.set_yticklabels(['States'])
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.7) 
                  for i in range(len(state_names))]
        ax1.legend(handles, state_names, loc='upper right')
        
        # Plot state probabilities if available
        if sequence_data.state_probs is not None:
            probs = sequence_data.state_probs
            ax2.plot(positions, probs[:, 0], label='P(Other)', color='lightblue', linewidth=2)
            ax2.plot(positions, probs[:, 1], label='P(Alpha-helix)', color='orange', linewidth=2)
            ax2.fill_between(positions, probs[:, 0], alpha=0.3, color='lightblue')
            ax2.fill_between(positions, probs[:, 1], alpha=0.3, color='orange')
            ax2.set_ylabel('State Probabilities')
            ax2.legend()
        else:
            # Just show amino acid sequence
            ax2.text(0.02, 0.5, sequence, transform=ax2.transAxes, 
                    fontfamily='monospace', fontsize=8)
            ax2.set_ylabel('Amino Acids')
        
        ax2.set_xlabel('Position')
        ax2.set_xlim(0, len(sequence)-1)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f'Protein Sequence (Length: {len(sequence)}, '
                        f'Alpha-helix: {sequence_data.alpha_helix_fraction:.2%})', 
                        fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_dataset_statistics(self, 
                              sequences: List[SequenceData],
                              title: str = "Dataset Statistics",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """Plot dataset statistics and distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        lengths = [seq.length for seq in sequences]
        alpha_fractions = [seq.alpha_helix_fraction for seq in sequences]
        
        # Length distribution
        axes[0, 0].hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].axvline(np.mean(lengths), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(lengths):.1f}')
        axes[0, 0].legend()
        
        # Alpha-helix fraction distribution
        axes[0, 1].hist(alpha_fractions, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Alpha-helix Fraction')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Alpha-helix Content Distribution')
        axes[0, 1].axvline(np.mean(alpha_fractions), color='red', linestyle='--',
                          label=f'Mean: {np.mean(alpha_fractions):.2f}')
        axes[0, 1].legend()
        
        # Length vs alpha-helix fraction scatter
        axes[0, 2].scatter(lengths, alpha_fractions, alpha=0.6, color='green')
        axes[0, 2].set_xlabel('Sequence Length')
        axes[0, 2].set_ylabel('Alpha-helix Fraction')
        axes[0, 2].set_title('Length vs Alpha-helix Content')
        
        # Amino acid composition
        all_sequences = ''.join([seq.sequence for seq in sequences])
        aa_counts = {}
        for aa in all_sequences:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        aa_names = sorted(aa_counts.keys())
        aa_freqs = [aa_counts[aa] / len(all_sequences) for aa in aa_names]
        
        axes[1, 0].bar(aa_names, aa_freqs, color='lightcoral', alpha=0.7)
        axes[1, 0].set_xlabel('Amino Acid')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Amino Acid Composition')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # State transition statistics
        total_transitions = {'other_to_other': 0, 'other_to_helix': 0, 
                           'helix_to_other': 0, 'helix_to_helix': 0}
        
        for seq in sequences:
            states = seq.states
            for i in range(len(states) - 1):
                current, next_state = states[i], states[i + 1]
                if current == 0 and next_state == 0:
                    total_transitions['other_to_other'] += 1
                elif current == 0 and next_state == 1:
                    total_transitions['other_to_helix'] += 1
                elif current == 1 and next_state == 0:
                    total_transitions['helix_to_other'] += 1
                elif current == 1 and next_state == 1:
                    total_transitions['helix_to_helix'] += 1
        
        trans_names = list(total_transitions.keys())
        trans_counts = list(total_transitions.values())
        
        axes[1, 1].bar(trans_names, trans_counts, color='lightgreen', alpha=0.7)
        axes[1, 1].set_xlabel('Transition Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('State Transition Counts')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Summary statistics
        stats_text = f"""Dataset Summary:
        N sequences: {len(sequences)}
        Avg length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}
        Avg alpha-helix: {np.mean(alpha_fractions):.2%} ± {np.std(alpha_fractions):.2%}
        Total amino acids: {len(all_sequences):,}
        Unique amino acids: {len(aa_counts)}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_axis_off()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics plot to {save_path}")
        
        return fig

class OutputManager:
    """Manage output directories and run tracking."""
    
    def __init__(self, base_output_dir: Path = Path("outputs")):
        """Initialize output manager."""
        self.base_output_dir = Path(base_output_dir)
        self.current_run_dir = None
    
    def create_run_directory(self, run_name: Optional[str] = None) -> Path:
        """Create new run directory with timestamp."""
        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        self.current_run_dir = self.base_output_dir / run_name
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "figures").mkdir(exist_ok=True)
        (self.current_run_dir / "data").mkdir(exist_ok=True)
        (self.current_run_dir / "models").mkdir(exist_ok=True)
        (self.current_run_dir / "reports").mkdir(exist_ok=True)
        
        logger.info(f"Created run directory: {self.current_run_dir}")
        return self.current_run_dir
    
    def save_run_config(self, config: Dict) -> None:
        """Save run configuration to JSON."""
        if self.current_run_dir is None:
            raise ValueError("No run directory created")
        
        config_path = self.current_run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Saved run config to {config_path}")
    
    def get_figure_path(self, filename: str) -> Path:
        """Get path for saving figures."""
        if self.current_run_dir is None:
            raise ValueError("No run directory created")
        return self.current_run_dir / "figures" / filename
    
    def get_data_path(self, filename: str) -> Path:
        """Get path for saving data."""
        if self.current_run_dir is None:
            raise ValueError("No run directory created")
        return self.current_run_dir / "data" / filename
    
    def get_model_path(self, filename: str) -> Path:
        """Get path for saving models."""
        if self.current_run_dir is None:
            raise ValueError("No run directory created")
        return self.current_run_dir / "models" / filename
    
    def get_report_path(self, filename: str) -> Path:
        """Get path for saving reports."""
        if self.current_run_dir is None:
            raise ValueError("No run directory created")
        return self.current_run_dir / "reports" / filename

def calculate_accuracy_metrics(true_states: np.ndarray, 
                             predicted_states: np.ndarray) -> Dict[str, float]:
    """Calculate accuracy metrics for state prediction."""
    accuracy = np.mean(true_states == predicted_states)
    
    # Per-state metrics
    alpha_mask = true_states == 1
    other_mask = true_states == 0
    
    alpha_accuracy = np.mean(true_states[alpha_mask] == predicted_states[alpha_mask]) if alpha_mask.any() else 0.0
    other_accuracy = np.mean(true_states[other_mask] == predicted_states[other_mask]) if other_mask.any() else 0.0
    
    # Confusion matrix elements
    tp = np.sum((true_states == 1) & (predicted_states == 1))  # True positive (alpha-helix)
    tn = np.sum((true_states == 0) & (predicted_states == 0))  # True negative (other)
    fp = np.sum((true_states == 0) & (predicted_states == 1))  # False positive
    fn = np.sum((true_states == 1) & (predicted_states == 0))  # False negative
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'overall_accuracy': accuracy,
        'alpha_helix_accuracy': alpha_accuracy,
        'other_accuracy': other_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
