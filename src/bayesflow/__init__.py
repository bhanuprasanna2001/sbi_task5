"""
BayesFlow implementation for protein secondary structure inference.
"""

from .networks import (
    SummaryNetwork,
    InferenceNetwork, 
    create_summary_network,
    create_inference_network
)

from .bayesflow_simulator import (
    create_protein_simulator_functions,
    protein_simulator,
    ProteinHMMSimulator,
    FixedLengthProteinSimulator
)

__all__ = [
    'SummaryNetwork',
    'InferenceNetwork', 
    'create_summary_network',
    'create_inference_network',
    'create_protein_simulator_functions',
    'protein_simulator',
    'ProteinHMMSimulator',
    'FixedLengthProteinSimulator'
]
