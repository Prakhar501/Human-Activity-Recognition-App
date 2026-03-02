"""
Hyperparameter Optimization for Federated Learning
- Systematically test parameter combinations
- Track results for each configuration
- Find optimal hyperparameters
- Generate comparison reports and visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
import warnings

# Import federated learning modules
from data_utils import load_all_data, partition_data_non_iid
from model_utils import create_cnn_model, compile_model
from client import ClientManager
from server import FederatedServer


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

class HyperparameterSpace:
    """Define the search space for hyperparameter optimization"""
    
    # Conservative search space (recommended for initial runs)
    CONSERVATIVE = {
        'num_clients': [3, 5, 7],
        'num_global_rounds': [10, 15, 20],
        'local_epochs': [3, 5, 7],
        'batch_size': [32],  # Fixed (matches baseline)
        'learning_rate': [0.001],  # Fixed (matches baseline)
    }
    
    # Standard search space (balanced)
    STANDARD = {
        'num_clients': [2, 5, 10],
        'num_global_rounds': [5, 10, 15, 20],
        'local_epochs': [2, 5, 10],
        'batch_size': [32],  # Fixed (matches baseline)
        'learning_rate': [0.0005, 0.001, 0.002],
    }
    
    # Extensive search space (comprehensive but slow)
    EXTENSIVE = {
        'num_clients': [2, 3, 5, 7, 10],
        'num_global_rounds': [5, 10, 15, 20, 25],
        'local_epochs': [2, 3, 5, 7, 10],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
    }
    
    # Quick test space (for fast validation)
    QUICK = {
        'num_clients': [2, 5],
        'num_global_rounds': [3, 5],
        'local_epochs': [2, 3],
        'batch_size': [32],
        'learning_rate': [0.001],
    }
    
    @staticmethod
    def get_space(mode='standard'):
        """
        Get parameter search space.
        
        Args:
            mode: 'conservative', 'standard', 'extensive', or 'quick'
        
        Returns:
            Dictionary with parameter ranges
        """
        spaces = {
            'conservative': HyperparameterSpace.CONSERVATIVE,
            'standard': HyperparameterSpace.STANDARD,
            'extensive': HyperparameterSpace.EXTENSIVE,
            'quick': HyperparameterSpace.QUICK,
        }
        return spaces.get(mode.lower(), HyperparameterSpace.STANDARD)
    
    @staticmethod
    def count_combinations(space):
        """Count total number of combinations in search space"""
        total = 1
        for param_values in space.values():
            total *= len(param_values)
        return total


# ============================================================================
# FEDERATED LEARNING WITH SINGLE CONFIG
# ============================================================================

def run_federated_with_config(X_train, y_train, X_test, y_test, config, verbose=0):
    """
    Run federated learning with a specific configuration.
    
    Args:
        X_train: Training features
        y_train: Training labels (one-hot)
        X_test: Test features
        y_test: Test labels (one-hot)
        config: Dictionary with hyperparameters
        verbose: Verbosity level
    
    Returns:
        Dictionary with results
    """
    
    # Extract config
    num_clients = config['num_clients']
    num_global_rounds = config['num_global_rounds']
    local_epochs = config['local_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    
    if verbose > 0:
        print(f"\n[Config] Clients={num_clients}, Rounds={num_global_rounds}, "
              f"Epochs={local_epochs}, LR={learning_rate}")
    
    try:
        # 1. Partition data
        clients_data = partition_data_non_iid(X_train, y_train, num_clients, seed=42)
        
        # 2. Initialize server and clients
        server = FederatedServer(learning_rate=learning_rate)
        client_manager = ClientManager(clients_data, batch_size=batch_size, learning_rate=learning_rate)
        
        # 3. Federated training loop
        for round_num in range(1, num_global_rounds + 1):
            if verbose > 1:
                print(f"  Round {round_num}/{num_global_rounds}")
            
            # Broadcast, train, aggregate
            global_weights = server.get_global_weights()
            client_manager.set_global_weights(global_weights)
            client_metrics = client_manager.train_all_clients(epochs=local_epochs, verbose=0)
            client_weights = client_manager.get_client_weights()
            client_sizes = client_manager.get_client_sizes()
            server.aggregate_weights(client_weights, client_sizes=client_sizes)
            
            # Evaluate
            loss, accuracy = server.evaluate_global_model(X_test, y_test, verbose=0)
        
        # Final evaluation
        final_loss, final_accuracy = server.evaluate_global_model(X_test, y_test, verbose=0)
        
        # Convergence metric (improvement from first to last round)
        initial_accuracy = server.history['global_accuracies'][0] if server.history['global_accuracies'] else 0
        improvement = final_accuracy - initial_accuracy
        
        result = {
            'config': config,
            'final_accuracy': float(final_accuracy),
            'final_loss': float(final_loss),
            'initial_accuracy': float(initial_accuracy),
            'improvement': float(improvement),
            'status': 'success'
        }
        
        if verbose > 0:
            print(f"  → Accuracy: {final_accuracy:.4f} (improvement: {improvement:+.4f})")
    
    except Exception as e:
        print(f"\n  ❌ Error with config: {str(e)[:100]}")
        result = {
            'config': config,
            'final_accuracy': 0.0,
            'final_loss': float('inf'),
            'initial_accuracy': 0.0,
            'improvement': 0.0,
            'status': f'failed: {str(e)[:50]}'
        }
    
    return result


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

class HyperparameterOptimizer:
    """
    Systematic hyperparameter optimization for federated learning.
    Uses grid search to find optimal parameters.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, space_mode='standard'):
        """
        Initialize optimizer.
        
        Args:
            X_train, y_train, X_test, y_test: Data
            space_mode: 'quick', 'conservative', 'standard', or 'extensive'
        """
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.search_space = HyperparameterSpace.get_space(space_mode)
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0
        
        # Calculate total combinations
        self.total_combinations = HyperparameterSpace.count_combinations(self.search_space)
        
        print(f"\n{'='*80}")
        print(f"🔍 HYPERPARAMETER OPTIMIZATION ({space_mode.upper()})")
        print(f"{'='*80}")
        print(f"Search Space Mode: {space_mode}")
        print(f"Total Combinations: {self.total_combinations}")
        print(f"Estimated Runtime: {self.total_combinations * 2}-{self.total_combinations * 5} minutes")
        print(f"{'='*80}\n")
    
    def optimize(self, verbose=1):
        """
        Run hyperparameter optimization using grid search.
        
        Args:
            verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
        
        Returns:
            List of results sorted by accuracy
        """
        
        print(f"Starting optimization...\n")
        
        # Generate all combinations
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]
        combinations = list(product(*param_values))
        
        # Test each combination
        for combo_idx, values in enumerate(combinations, 1):
            # Create config from combination
            config = dict(zip(param_names, values))
            
            print(f"[{combo_idx}/{self.total_combinations}] "
                  f"Testing: Clients={config['num_clients']}, "
                  f"Rounds={config['num_global_rounds']}, "
                  f"Epochs={config['local_epochs']}, "
                  f"LR={config['learning_rate']}")
            
            # Run federated learning with this config
            result = run_federated_with_config(
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                config,
                verbose=max(0, verbose - 1)
            )
            
            self.results.append(result)
            
            # Track best
            if result['final_accuracy'] > self.best_accuracy:
                self.best_accuracy = result['final_accuracy']
                self.best_config = config
                print(f"           → 🎯 NEW BEST: {self.best_accuracy:.4f}")
            else:
                improvement = result['improvement']
                print(f"           → Accuracy: {result['final_accuracy']:.4f} "
                      f"(Δ{improvement:+.4f})")
        
        # Sort results
        self.results.sort(key=lambda x: x['final_accuracy'], reverse=True)
        
        return self.results
    
    def print_results(self, top_n=10):
        """
        Print top N results.
        
        Args:
            top_n: Number of top results to show
        """
        
        print(f"\n{'='*80}")
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"{'='*80}\n")
        
        print(f"🏆 BEST CONFIGURATION:")
        print(f"{'─'*80}")
        if self.best_config:
            for param, value in self.best_config.items():
                print(f"  {param:20s}: {value}")
            print(f"  Final Accuracy:      {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print(f"{'─'*80}\n")
        
        print(f"📊 TOP {min(top_n, len(self.results))} CONFIGURATIONS:")
        print(f"{'─'*80}")
        
        for rank, result in enumerate(self.results[:top_n], 1):
            config = result['config']
            accuracy = result['final_accuracy']
            
            print(f"\n#{rank} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"    Clients: {config['num_clients']}, "
                  f"Rounds: {config['num_global_rounds']}, "
                  f"Epochs: {config['local_epochs']}, "
                  f"LR: {config['learning_rate']}")
    
    def get_best_config(self):
        """Return best configuration found"""
        return self.best_config
    
    def get_results_dataframe(self):
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with all results
        """
        
        data = []
        for result in self.results:
            row = result['config'].copy()
            row['final_accuracy'] = result['final_accuracy']
            row['final_loss'] = result['final_loss']
            row['improvement'] = result['improvement']
            row['status'] = result['status']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, output_dir='federated/results'):
        """
        Save optimization results to files.
        
        Args:
            output_dir: Directory to save results
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'best_config': self.best_config,
            'best_accuracy': float(self.best_accuracy),
            'total_combinations_tested': len(self.results),
            'top_10_results': [
                {
                    'rank': i+1,
                    'config': r['config'],
                    'accuracy': float(r['final_accuracy']),
                    'improvement': float(r['improvement'])
                }
                for i, r in enumerate(self.results[:10])
            ]
        }
        
        json_path = os.path.join(output_dir, 'optimization_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_path}")
        
        # Save as CSV
        df = self.get_results_dataframe()
        csv_path = os.path.join(output_dir, 'optimization_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved to: {csv_path}")
        
        # Save best config
        config_path = os.path.join(output_dir, 'best_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        print(f"✅ Best config saved to: {config_path}")
        
        return output_dir
    
    def plot_results(self, output_dir='federated/results'):
        """
        Create visualizations of optimization results.
        
        Args:
            output_dir: Directory to save plots
        """
        
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_results_dataframe()
        
        # Plot 1: Accuracy vs Number of Clients
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        for clients in sorted(df['num_clients'].unique()):
            subset = df[df['num_clients'] == clients]
            ax.scatter(subset['num_global_rounds'], subset['final_accuracy'], 
                      label=f'{clients} clients', s=100, alpha=0.6)
        ax.set_xlabel('Global Rounds', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Accuracy vs Global Rounds (by Client Count)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Local Epochs
        ax = axes[0, 1]
        for epochs in sorted(df['local_epochs'].unique()):
            subset = df[df['local_epochs'] == epochs]
            ax.scatter(subset['num_clients'], subset['final_accuracy'],
                      label=f'{epochs} epochs', s=100, alpha=0.6)
        ax.set_xlabel('Number of Clients', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Accuracy vs Client Count (by Local Epochs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Impact
        ax = axes[1, 0]
        for lr in sorted(df['learning_rate'].unique()):
            subset = df[df['learning_rate'] == lr]
            ax.scatter(subset.index, subset['final_accuracy'],
                      label=f'LR={lr}', s=100, alpha=0.6)
        ax.set_xlabel('Experiment Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Accuracy vs Learning Rate', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Top configurations
        ax = axes[1, 1]
        top_configs = df.nlargest(10, 'final_accuracy')
        config_labels = [f"C{row['num_clients']}_R{row['num_global_rounds']}_E{row['local_epochs']}" 
                        for _, row in top_configs.iterrows()]
        ax.barh(range(len(top_configs)), top_configs['final_accuracy'], color='#4ECDC4', alpha=0.8)
        ax.set_yticks(range(len(top_configs)))
        ax.set_yticklabels(config_labels, fontsize=9)
        ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Configurations', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'optimization_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {plot_path}")
        plt.close()
        
        # Plot 5: Improvement distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(df['final_accuracy'], df['improvement'], s=100, alpha=0.6, c=df['num_global_rounds'],
                  cmap='viridis')
        
        # Add best point
        best_idx = df['final_accuracy'].idxmax()
        best_row = df.loc[best_idx]
        ax.scatter(best_row['final_accuracy'], best_row['improvement'], 
                  s=300, c='red', marker='*', edgecolors='black', linewidths=2,
                  label='Best Config', zorder=5)
        
        ax.set_xlabel('Final Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement (Δ Accuracy)', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Analysis: Accuracy vs Improvement', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Global Rounds', fontsize=11, fontweight='bold')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        convergence_path = os.path.join(output_dir, 'convergence_analysis.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        print(f"✅ Convergence plot saved to: {convergence_path}")
        plt.close()


# ============================================================================
# MAIN OPTIMIZATION RUNNER
# ============================================================================

def main():
    """Main hyperparameter optimization pipeline"""
    
    print("\n" + "="*80)
    print("🔍 FEDERATED LEARNING - HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    print("\n[Setup] Loading datasets...")
    X_train, y_train, X_test, y_test, _ = load_all_data()
    
    print(f"[Setup] Data ready:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Ask for search space mode - for now default to 'conservative'
    # Can be changed to 'quick', 'standard', or 'extensive'
    space_mode = 'standard'  # Default: good balance between speed and comprehensiveness
    
    print(f"\n[Setup] Using '{space_mode}' search space")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(X_train, y_train, X_test, y_test, space_mode=space_mode)
    results = optimizer.optimize(verbose=1)
    
    # Print results
    optimizer.print_results(top_n=10)
    
    # Save and visualize
    print("\n[Saving] Writing results to disk...")
    optimizer.save_results()
    
    print("\n[Visualization] Creating plots...")
    optimizer.plot_results()
    
    # Return best config for use
    best_config = optimizer.get_best_config()
    
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Best Configuration Found:")
    print(f"   num_clients: {best_config['num_clients']}")
    print(f"   num_global_rounds: {best_config['num_global_rounds']}")
    print(f"   local_epochs: {best_config['local_epochs']}")
    print(f"   batch_size: {best_config['batch_size']}")
    print(f"   learning_rate: {best_config['learning_rate']}")
    print(f"\n   Expected Accuracy: {optimizer.best_accuracy:.4f} ({optimizer.best_accuracy*100:.2f}%)")
    
    print(f"\n📝 To use this configuration in federated_train.py:")
    print(f"   Update CONFIG dict with:")
    print(f"   {{")
    for key, value in best_config.items():
        print(f"       '{key}': {value},")
    print(f"   }}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()
