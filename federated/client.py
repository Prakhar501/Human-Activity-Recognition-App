"""
Federated Learning Client
- Receives global model from server
- Trains locally on its partition of data
- Sends updated weights back to server
"""

import numpy as np
from tensorflow import keras
from model_utils import create_cnn_model, compile_model, get_model_weights, set_model_weights
from data_utils import get_client_batch


class FederatedClient:
    """
    Represents a single federated learning client.
    Performs local training and maintains local model state.
    """
    
    def __init__(self, client_id, X_train, y_train, batch_size=32, learning_rate=0.001):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            X_train: Client's training data (N, 128, 6)
            y_train: Client's training labels (N, 4) - one-hot encoded
            batch_size: Batch size for local training
            learning_rate: Learning rate for optimizer
        """
        
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_samples = len(X_train)
        
        # Create and compile local model
        self.model = create_cnn_model(input_shape=(128, 6), num_classes=4)
        self.model = compile_model(self.model, learning_rate=learning_rate)
        
        # Metrics tracking
        self.local_epochs_trained = 0
        self.training_history = []
    
    def get_model_weights(self):
        """
        Get current model weights.
        
        Returns:
            List of weight arrays
        """
        return get_model_weights(self.model)
    
    def set_model_weights(self, weights):
        """
        Update model weights (from global model).
        
        Args:
            weights: List of weight arrays from server
        """
        set_model_weights(self.model, weights)
    
    def train_local(self, epochs=5, verbose=0):
        """
        Perform local training on client's data.
        
        Args:
            epochs: Number of local epochs to train
            verbose: Verbosity level (0=silent, 1=progress bar)
        
        Returns:
            Dictionary with training metrics
        """
        
        print(f"\n[Client {self.client_id}] Starting local training ({epochs} epochs)...")
        
        # Track metrics
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # Train on all batches
            for X_batch, y_batch in get_client_batch(self.X_train, self.y_train, self.batch_size):
                loss, accuracy = self.model.train_on_batch(X_batch, y_batch)
                epoch_loss += loss
                epoch_accuracy += accuracy
                num_batches += 1
            
            # Average metrics for epoch
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            if verbose > 0:
                print(f"[Client {self.client_id}] Epoch {epoch+1}/{epochs} - "
                      f"loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}")
        
        self.local_epochs_trained += epochs
        self.training_history.append({
            'epochs': epochs,
            'losses': epoch_losses,
            'accuracies': epoch_accuracies
        })
        
        metrics = {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'epochs_trained': epochs,
            'final_loss': epoch_losses[-1],
            'final_accuracy': epoch_accuracies[-1],
            'avg_loss': np.mean(epoch_losses),
            'avg_accuracy': np.mean(epoch_accuracies)
        }
        
        print(f"[Client {self.client_id}] Local training complete - "
              f"Final Accuracy: {metrics['final_accuracy']:.4f}")
        
        return metrics
    
    def evaluate_local(self, X_test=None, y_test=None, verbose=0):
        """
        Evaluate model on client's test data (if provided).
        
        Args:
            X_test: Test data (optional)
            y_test: Test labels (optional)
            verbose: Verbosity level
        
        Returns:
            (loss, accuracy) or None if no test data provided
        """
        
        if X_test is None or y_test is None:
            return None
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=verbose)
        return loss, accuracy
    
    def get_client_info(self):
        """
        Get client information summary.
        
        Returns:
            Dictionary with client metadata
        """
        
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'epochs_trained': self.local_epochs_trained,
            'history_length': len(self.training_history)
        }


class ClientManager:
    """
    Manages multiple federated learning clients.
    Coordinates client initialization and training.
    """
    
    def __init__(self, clients_data, batch_size=32, learning_rate=0.001):
        """
        Initialize client manager with partitioned data.
        
        Args:
            clients_data: List of (X_client, y_client) tuples
            batch_size: Batch size for local training
            learning_rate: Learning rate for all clients
        """
        
        self.clients = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create clients
        for client_id, (X_client, y_client) in enumerate(clients_data):
            client = FederatedClient(
                client_id=client_id,
                X_train=X_client,
                y_train=y_client,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            self.clients.append(client)
        
        print(f"\n[ClientManager] Initialized {len(self.clients)} clients")
    
    def train_all_clients(self, epochs=5, verbose=1):
        """
        Train all clients locally.
        
        Args:
            epochs: Number of local epochs per client
            verbose: Verbosity level
        
        Returns:
            List of training metrics from each client
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING ALL CLIENTS (Local Epochs: {epochs})")
        print(f"{'='*80}")
        
        metrics_list = []
        
        for client in self.clients:
            metrics = client.train_local(epochs=epochs, verbose=verbose)
            metrics_list.append(metrics)
        
        return metrics_list
    
    def get_client_weights(self):
        """
        Get weights from all clients.
        Used by server for aggregation.
        
        Returns:
            List of weight lists (one per client)
        """
        
        return [client.get_model_weights() for client in self.clients]
    
    def set_global_weights(self, global_weights):
        """
        Distribute global weights to all clients.
        Called before each round of training.
        
        Args:
            global_weights: Aggregated weights from server
        """
        
        for client in self.clients:
            client.set_model_weights(global_weights)
    
    def get_client_sizes(self):
        """
        Get number of samples per client.
        Used for weighted averaging.
        
        Returns:
            List of sample counts
        """
        
        return [client.num_samples for client in self.clients]
    
    def get_client_info(self):
        """
        Get summary information for all clients.
        
        Returns:
            List of client info dictionaries
        """
        
        return [client.get_client_info() for client in self.clients]


if __name__ == "__main__":
    # Test client creation and training
    print("Testing Federated Client...")
    
    # Create dummy data
    X_train = np.random.randn(100, 128, 6).astype(np.float32)
    y_train = np.zeros((100, 4))
    y_train[np.arange(100), np.random.randint(0, 4, 100)] = 1
    
    # Create client
    client = FederatedClient(0, X_train, y_train)
    print(f"✅ Client created: {client.get_client_info()}")
    
    # Train locally
    metrics = client.train_local(epochs=2, verbose=1)
    print(f"✅ Training complete: {metrics}")
    
    print("\n✅ Client utilities working correctly")
