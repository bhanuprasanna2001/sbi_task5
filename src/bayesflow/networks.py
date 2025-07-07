"""
BayesFlow neural networks for protein secondary structure inference.
Implements summary and inference networks optimized for sequence data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SummaryNetwork(keras.Model):
    """
    Summary network that processes protein sequences into fixed-size representations.
    
    Uses embeddings + LSTM/Transformer to handle variable-length sequences
    and produces a summary suitable for the inference network.
    """
    
    def __init__(self,
                 vocab_size: int = 20,  # 20 amino acids
                 embedding_dim: int = 64,
                 lstm_units: int = 128,
                 dense_units: int = 256,
                 output_dim: int = 128,
                 dropout_rate: float = 0.1,
                 use_bidirectional: bool = True,
                 **kwargs):
        """
        Initialize summary network.
        
        Args:
            vocab_size: Size of amino acid vocabulary (20)
            embedding_dim: Dimension of amino acid embeddings
            lstm_units: Number of LSTM units
            dense_units: Number of dense layer units
            output_dim: Output dimension of summary
            dropout_rate: Dropout rate for regularization
            use_bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_bidirectional = use_bidirectional
        
        # Amino acid embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,  # Handle padding
            name='amino_acid_embedding'
        )
        
        # LSTM layer for sequence processing
        lstm_layer = layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='sequence_lstm'
        )
        
        if use_bidirectional:
            self.lstm = layers.Bidirectional(lstm_layer, name='bidirectional_lstm')
            self.lstm_output_dim = 2 * lstm_units
        else:
            self.lstm = lstm_layer
            self.lstm_output_dim = lstm_units
        
        # Attention mechanism for sequence summarization
        self.attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=self.lstm_output_dim // 8,
            name='sequence_attention'
        )
        
        # Global pooling layers
        self.global_avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')
        self.global_max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')
        
        # Dense layers for summary computation
        self.dense1 = layers.Dense(
            dense_units,
            activation='relu',
            name='summary_dense1'
        )
        self.dropout1 = layers.Dropout(dropout_rate, name='summary_dropout1')
        
        self.dense2 = layers.Dense(
            dense_units // 2,
            activation='relu',
            name='summary_dense2'
        )
        self.dropout2 = layers.Dropout(dropout_rate, name='summary_dropout2')
        
        # Output layer
        self.output_layer = layers.Dense(
            output_dim,
            activation='linear',
            name='summary_output'
        )
        
        logger.info(f"Initialized SummaryNetwork with output_dim={output_dim}")
    
    def call(self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of summary network.
        
        Args:
            inputs: Dict containing 'sequences', 'masks', 'lengths'
            training: Whether in training mode
            
        Returns:
            Summary tensor of shape (batch_size, output_dim)
        """
        sequences = inputs['sequences']  # (batch_size, max_seq_len)
        masks = inputs.get('masks', None)  # (batch_size, max_seq_len)
        
        # Embed amino acid sequences
        embedded = self.embedding(sequences)  # (batch_size, max_seq_len, embedding_dim)
        
        # Apply LSTM
        lstm_out = self.lstm(embedded, training=training)  # (batch_size, max_seq_len, lstm_output_dim)
        
        # Apply self-attention
        attended = self.attention(
            lstm_out, lstm_out, 
            attention_mask=masks,
            training=training
        )  # (batch_size, max_seq_len, lstm_output_dim)
        
        # Global pooling to get fixed-size representation
        avg_pooled = self.global_avg_pool(attended)  # (batch_size, lstm_output_dim)
        max_pooled = self.global_max_pool(attended)  # (batch_size, lstm_output_dim)
        
        # Concatenate pooled representations
        pooled = tf.concat([avg_pooled, max_pooled], axis=-1)  # (batch_size, 2*lstm_output_dim)
        
        # Apply dense layers
        x = self.dense1(pooled)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        # Output summary
        summary = self.output_layer(x)  # (batch_size, output_dim)
        
        return summary


class InferenceNetwork(keras.Model):
    """
    Inference network that predicts state probabilities from sequence summaries.
    
    Takes the output of the summary network and predicts the posterior
    distribution over state probabilities for each position in the sequence.
    """
    
    def __init__(self,
                 summary_dim: int = 128,
                 hidden_units: int = 512,
                 n_layers: int = 4,
                 max_seq_len: int = 200,
                 n_states: int = 2,
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize inference network.
        
        Args:
            summary_dim: Dimension of input summary
            hidden_units: Number of hidden units per layer
            n_layers: Number of hidden layers
            max_seq_len: Maximum sequence length
            n_states: Number of HMM states (2)
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.summary_dim = summary_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_states = n_states
        self.dropout_rate = dropout_rate
        
        # Dense layers for processing summary
        self.dense_layers = []
        self.dropout_layers = []
        
        for i in range(n_layers):
            self.dense_layers.append(
                layers.Dense(
                    hidden_units,
                    activation='relu',
                    name=f'inference_dense_{i+1}'
                )
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'inference_dropout_{i+1}')
            )
        
        # Output layer - predicts parameters for state probability distributions
        # For each sequence position, predict mean and log_std for each state probability
        self.output_layer = layers.Dense(
            max_seq_len * n_states * 2,  # mean + log_std for each state at each position
            activation='linear',
            name='inference_output'
        )
        
        logger.info(f"Initialized InferenceNetwork with {n_layers} layers")
    
    def call(self, summary: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of inference network.
        
        Args:
            summary: Summary tensor from summary network (batch_size, summary_dim)
            training: Whether in training mode
            
        Returns:
            Output tensor with distribution parameters
        """
        x = summary
        
        # Apply dense layers with dropout
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x, training=training)
        
        # Output layer
        output = self.output_layer(x)  # (batch_size, max_seq_len * n_states * 2)
        
        # Reshape to (batch_size, max_seq_len, n_states, 2) for mean and log_std
        batch_size = tf.shape(output)[0]
        reshaped = tf.reshape(
            output, 
            (batch_size, self.max_seq_len, self.n_states, 2)
        )
        
        return reshaped
    
    def sample_posterior(self, 
                        summary: tf.Tensor, 
                        n_samples: int = 1000,
                        training: bool = False) -> tf.Tensor:
        """
        Sample from the predicted posterior distribution.
        
        Args:
            summary: Summary tensor
            n_samples: Number of posterior samples
            training: Whether in training mode
            
        Returns:
            Posterior samples of shape (n_samples, batch_size, max_seq_len, n_states)
        """
        # Get distribution parameters
        dist_params = self.call(summary, training=training)
        
        # Extract mean and log_std
        mean = dist_params[..., 0]  # (batch_size, max_seq_len, n_states)
        log_std = dist_params[..., 1]  # (batch_size, max_seq_len, n_states)
        std = tf.exp(log_std)
        
        # Create normal distribution and sample
        dist = tf.random.normal(
            shape=(n_samples, tf.shape(mean)[0], self.max_seq_len, self.n_states),
            mean=mean[None, ...],
            stddev=std[None, ...]
        )
        
        # Apply softmax to ensure state probabilities sum to 1
        samples = tf.nn.softmax(dist, axis=-1)
        
        return samples


def create_summary_network(**kwargs) -> SummaryNetwork:
    """Factory function to create summary network."""
    return SummaryNetwork(**kwargs)


def create_inference_network(**kwargs) -> InferenceNetwork:
    """Factory function to create inference network."""
    return InferenceNetwork(**kwargs)
