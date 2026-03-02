"""
Model Utilities for Federated Learning
- Creates same CNN architecture as train_combined_model.py
- Handles weight serialization for FL communication
- Provides utility functions for model operations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def create_cnn_model(input_shape=(128, 6), num_classes=4):
    """
    Create CNN model identical to train_combined_model.py
    
    Architecture:
    - 4 Conv1D blocks with BatchNormalization and Dropout
    - GlobalAveragePooling1D
    - 2 Dense layers
    - Output: softmax (num_classes)
    
    Args:
        input_shape: (timesteps, features) - default (128, 6)
        num_classes: Number of output classes - default 4
    
    Returns:
        Compiled keras model
    """
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Normalization(),
        
        # Conv Block 1: 64 filters
        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 2: 128 filters
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 3: 128 filters
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 4: 256 filters
        layers.Conv1D(256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        # Output layer (4 classes)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with same configuration as train_combined_model.py
    
    Args:
        model: Keras model
        learning_rate: Adam optimizer learning rate (default 0.001)
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_weights(model):
    """
    Extract model weights as NumPy arrays.
    Used for federated weight exchange.
    
    Args:
        model: Keras model
    
    Returns:
        List of weight arrays
    """
    return [w.numpy() if hasattr(w, 'numpy') else np.array(w) for w in model.get_weights()]


def set_model_weights(model, weights):
    """
    Set model weights from NumPy arrays.
    Used after weight aggregation in FedAvg.
    
    Args:
        model: Keras model
        weights: List of weight arrays
    """
    model.set_weights(weights)


def average_weights(client_weights_list, client_sizes=None):
    """
    Perform Federated Averaging (FedAvg).
    
    Aggregates weights from multiple clients, optionally weighted by client data size.
    
    Args:
        client_weights_list: List of weight lists from each client
                           Each element is a list of weight arrays
        client_sizes: Optional list of client dataset sizes for weighted averaging
                     If None, uses simple averaging (all clients equally weighted)
    
    Returns:
        Averaged weights (list of arrays)
    """
    
    num_clients = len(client_weights_list)
    
    # Default: equal weights for all clients
    if client_sizes is None:
        client_sizes = [1.0] * num_clients
    
    # Normalize sizes to create weights
    total_size = sum(client_sizes)
    client_weights = [size / total_size for size in client_sizes]
    
    # Average weights
    num_layers = len(client_weights_list[0])
    aggregated_weights = []
    
    for layer_idx in range(num_layers):
        # Initialize with zeros (ensure float dtype for proper aggregation)
        layer_shape = client_weights_list[0][layer_idx].shape
        layer_dtype = client_weights_list[0][layer_idx].dtype
        
        # Neural network weights should be floating point
        # If we encounter int types, convert to float32
        if np.issubdtype(layer_dtype, np.integer):
            layer_dtype = np.float32
        
        aggregated_layer = np.zeros(layer_shape, dtype=np.float64)
        
        # Weighted sum
        for client_idx in range(num_clients):
            client_layer = client_weights_list[client_idx][layer_idx].astype(np.float64)
            aggregated_layer += client_weights[client_idx] * client_layer
        
        # Convert back to original dtype
        aggregated_layer = aggregated_layer.astype(layer_dtype)
        aggregated_weights.append(aggregated_layer)
    
    return aggregated_weights


def clone_model(model):
    """
    Create a deep copy of a model with same architecture and weights.
    
    Args:
        model: Keras model
    
    Returns:
        New model with same architecture and weights
    """
    
    # Clone architecture
    cloned = keras.models.clone_model(model)
    
    # Copy weights
    cloned.set_weights(model.get_weights())
    
    return cloned


def evaluate_model(model, X_test, y_test, batch_size=32, verbose=0):
    """
    Evaluate model on test data.
    
    Args:
        model: Keras model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        batch_size: Batch size for evaluation
        verbose: Verbosity level
    
    Returns:
        (loss, accuracy)
    """
    
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return loss, accuracy


def predict_batch(model, X, batch_size=32):
    """
    Get predictions for a batch of samples.
    
    Args:
        model: Keras model
        X: Input features
        batch_size: Batch size
    
    Returns:
        Predictions (probabilities)
    """
    
    return model.predict(X, batch_size=batch_size, verbose=0)


if __name__ == "__main__":
    # Test model creation
    print("Creating CNN model...")
    model = create_cnn_model()
    model = compile_model(model)
    
    print("\nModel summary:")
    model.summary()
    
    # Test weight operations
    print("\nTesting weight operations...")
    weights = get_model_weights(model)
    print(f"✅ Extracted {len(weights)} weight arrays")
    
    # Test averaging
    print("\nTesting weight averaging...")
    weights_copy = [w.copy() for w in weights]
    averaged = average_weights([weights, weights_copy])
    print(f"✅ Averaged {len(averaged)} weight arrays")
    
    # Test cloning
    print("\nTesting model cloning...")
    cloned = clone_model(model)
    print(f"✅ Cloned model created")
    
    print("\n✅ All model utilities working correctly")
