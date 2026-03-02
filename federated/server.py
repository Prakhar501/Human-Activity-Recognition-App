"""
Federated Learning Server (FedAvg Aggregator)
- Receives model updates from all clients
- Performs Federated Averaging (FedAvg)
- Broadcasts global model to clients
- Evaluates global model
"""

import numpy as np
from tensorflow import keras
from model_utils import create_cnn_model, compile_model, get_model_weights, set_model_weights, average_weights


class FederatedServer:
    """
    Federated Learning Server (FedAvg implementation).
    Aggregates client models and maintains global model state.
    """
    
    def __init__(self, learning_rate=0.001):
        """
        Initialize federated server.
        
        Args:
            learning_rate: Learning rate for global model optimizer
        """
        
        # Create global model
        self.global_model = create_cnn_model(input_shape=(128, 6), num_classes=4)
        self.global_model = compile_model(self.global_model, learning_rate=learning_rate)
        
        # Global state tracking
        self.global_round = 0
        self.history = {
            'global_rounds': [],
            'global_losses': [],
            'global_accuracies': [],
            'num_clients': []
        }
        
        print("[Server] Initialized with CNN model (4 activities)")
    
    def get_global_weights(self):
        """
        Get current global model weights.
        Sent to clients at beginning of each round.
        
        Returns:
            List of weight arrays
        """
        return get_model_weights(self.global_model)
    
    def aggregate_weights(self, client_weights_list, client_sizes=None):
        """
        Perform Federated Averaging (FedAvg) on client weights.
        
        Algorithm:
        - Receive updated weights from all K clients
        - Compute weighted average of weights
        - Update global model with averaged weights
        
        Args:
            client_weights_list: List of weight lists from each client
            client_sizes: Optional list of client dataset sizes
                         If None, uses equal weighting
        
        Returns:
            Aggregated weights
        """
        
        print(f"\n[Server] Aggregating weights from {len(client_weights_list)} clients...")
        
        # Perform averaging
        aggregated_weights = average_weights(client_weights_list, client_sizes)
        
        # Update global model
        set_model_weights(self.global_model, aggregated_weights)
        
        print("[Server] Weights aggregated and global model updated")
        
        return aggregated_weights
    
    def evaluate_global_model(self, X_test, y_test, verbose=0):
        """
        Evaluate current global model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity level
        
        Returns:
            (loss, accuracy)
        """
        
        loss, accuracy = self.global_model.evaluate(X_test, y_test, verbose=verbose)
        return loss, accuracy
    
    def complete_round(self, X_test, y_test, client_metrics=None):
        """
        Process end of round: log metrics and update history.
        
        Args:
            X_test: Test features for evaluation
            y_test: Test labels
            client_metrics: Optional list of metrics from clients
        
        Returns:
            Dictionary with round summary
        """
        
        self.global_round += 1
        
        # Evaluate global model
        loss, accuracy = self.evaluate_global_model(X_test, y_test, verbose=0)
        
        # Store history
        self.history['global_rounds'].append(self.global_round)
        self.history['global_losses'].append(loss)
        self.history['global_accuracies'].append(accuracy)
        if client_metrics:
            self.history['num_clients'].append(len(client_metrics))
        
        round_summary = {
            'global_round': self.global_round,
            'global_loss': loss,
            'global_accuracy': accuracy,
            'num_samples_aggregated': None
        }
        
        # Include client info if available
        if client_metrics:
            total_samples = sum([m['num_samples'] for m in client_metrics])
            avg_client_accuracy = np.mean([m['final_accuracy'] for m in client_metrics])
            
            round_summary['num_clients'] = len(client_metrics)
            round_summary['num_samples_aggregated'] = total_samples
            round_summary['avg_client_accuracy'] = avg_client_accuracy
        
        return round_summary
    
    def log_round_summary(self, round_summary, client_metrics=None):
        """
        Print summary of completed round.
        
        Args:
            round_summary: Dictionary from complete_round()
            client_metrics: Optional client metrics
        """
        
        print(f"\n{'─'*80}")
        print(f"📊 GLOBAL ROUND {round_summary['global_round']} SUMMARY")
        print(f"{'─'*80}")
        print(f"Global Loss:     {round_summary['global_loss']:.6f}")
        print(f"Global Accuracy: {round_summary['global_accuracy']:.4f} ({round_summary['global_accuracy']*100:.2f}%)")
        
        if 'num_clients' in round_summary:
            print(f"Clients Trained: {round_summary['num_clients']}")
            print(f"Total Aggregated Samples: {round_summary['num_samples_aggregated']}")
            print(f"Avg Client Accuracy: {round_summary['avg_client_accuracy']:.4f}")
        
        if client_metrics:
            print(f"\nPer-Client Accuracy:")
            for m in client_metrics:
                print(f"  Client {m['client_id']}: {m['final_accuracy']:.4f}")
        
        print(f"{'─'*80}")
    
    def get_history(self):
        """
        Get global training history.
        
        Returns:
            Dictionary with history arrays
        """
        return self.history
    
    def print_final_results(self, centralized_accuracy=None):
        """
        Print final results summary.
        
        Args:
            centralized_accuracy: Optional baseline centralized accuracy for comparison
        """
        
        if not self.history['global_accuracies']:
            print("No training history available")
            return
        
        final_accuracy = self.history['global_accuracies'][-1]
        final_loss = self.history['global_losses'][-1]
        
        print(f"\n{'='*80}")
        print(f"🎉 FEDERATED LEARNING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal Global Model Performance:")
        print(f"  Loss:     {final_loss:.6f}")
        print(f"  Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        if len(self.history['global_accuracies']) > 1:
            initial_accuracy = self.history['global_accuracies'][0]
            improvement = final_accuracy - initial_accuracy
            print(f"\nImprovement over {len(self.history['global_rounds'])} rounds:")
            print(f"  Initial Accuracy: {initial_accuracy:.4f} ({initial_accuracy*100:.2f}%)")
            print(f"  Final Accuracy:   {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
            print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if centralized_accuracy is not None:
            difference = final_accuracy - centralized_accuracy
            print(f"\nComparison with Centralized Baseline:")
            print(f"  Centralized Accuracy: {centralized_accuracy:.4f} ({centralized_accuracy*100:.2f}%)")
            print(f"  Federated Accuracy:   {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
            print(f"  Difference: {difference:+.4f} ({difference*100:+.2f}%)")
            
            if abs(difference) < 0.01:
                print(f"  ✅ Federated learning achieved comparable results to centralized!")
            elif difference > 0:
                print(f"  ✅ Federated learning outperformed centralized baseline!")
            else:
                print(f"  ⚠️  Federated learning underperformed (expected due to data distribution)")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Test server creation
    print("Testing Federated Server...")
    
    server = FederatedServer()
    print(f"✅ Server initialized")
    
    # Test weight operations
    weights = server.get_global_weights()
    print(f"✅ Retrieved {len(weights)} weight arrays")
    
    # Test dummy aggregation
    client_weights = [w.copy() for w in weights]
    client_weights_list = [client_weights, client_weights]
    
    aggregated = server.aggregate_weights(client_weights_list)
    print(f"✅ Aggregated weights from 2 clients")
    
    # Test evaluation with dummy data
    X_test = np.random.randn(100, 128, 6).astype(np.float32)
    y_test = np.zeros((100, 4))
    y_test[np.arange(100), np.random.randint(0, 4, 100)] = 1
    
    loss, accuracy = server.evaluate_global_model(X_test, y_test)
    print(f"✅ Evaluated on test data: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    # Test round completion
    round_summary = server.complete_round(X_test, y_test)
    server.log_round_summary(round_summary)
    
    print("\n✅ Server utilities working correctly")
