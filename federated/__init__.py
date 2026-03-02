"""
Federated Learning Module for Human Activity Recognition

This package implements the FedAvg (Federated Averaging) algorithm
for distributed training of HAR models across multiple clients.

Usage:
    python federated_train.py              # Run standard FL training
    python hyperparameter_optimization.py  # Optimize hyperparameters
    python train_with_optimized_params.py  # Use optimized configuration
    
For configuration and documentation, see README.md
"""

__version__ = "2.0.0"
__author__ = "Federated Learning Implementation"

from .model_utils import create_cnn_model, compile_model, average_weights
from .data_utils import load_all_data, partition_data_non_iid
from .client import FederatedClient, ClientManager
from .server import FederatedServer
from .hyperparameter_optimization import HyperparameterOptimizer, HyperparameterSpace

__all__ = [
    'create_cnn_model',
    'compile_model',
    'average_weights',
    'load_all_data',
    'partition_data_non_iid',
    'FederatedClient',
    'ClientManager',
    'FederatedServer',
    'HyperparameterOptimizer',
    'HyperparameterSpace',
]
