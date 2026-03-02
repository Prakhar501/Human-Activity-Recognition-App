"""
Random Search Hyperparameter Optimization for Federated Learning
- Randomly samples configurations instead of exhaustive grid search
- Much faster: Tests 30-50 configs instead of 108
- Expected time: 2-4 hours instead of 12+ hours
- Often finds near-optimal solutions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Import federated learning modules
from data_utils import load_all_data, partition_data_non_iid
from model_utils import create_cnn_model, compile_model
from client import ClientManager
from server import FederatedServer


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

class RandomSearchSpace:
    """Define search space for random sampling"""
    
    # Define ranges for each hyperparameter
    SEARCH_SPACE = {
        'num_clients': [2, 3, 5, 7, 10],
        'num_global_rounds': [5, 10, 15, 20, 25],
        'local_epochs': [2, 3, 5, 7, 10],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
    }
    
    # Focused search (best parameters found so far)
    FOCUSED_SPACE = {
        'num_clients': [2, 3, 5],
        'num_global_rounds': [5, 10, 15],
        'local_epochs': [3, 5, 7],
        'batch_size': [32],  # Fixed based on baseline
        'learning_rate': [0.0005, 0.001, 0.002],
    }
    
    @staticmethod
    def sample_random_config(space_dict, seed=None):
        """Sample one random configuration"""
        if seed is not None:
            random.seed(seed)
        
        config = {}
        for param, values in space_dict.items():
            config[param] = random.choice(values)
        
        return config
    
    @staticmethod
    def generate_random_configs(n_samples, space_dict, seed=42):
        """Generate n_samples random configurations (unique)"""
        random.seed(seed)
        configs = []
        seen = set()
        
        max_attempts = n_samples * 10  # Prevent infinite loop
        attempts = 0
        
        while len(configs) < n_samples and attempts < max_attempts:
            config = RandomSearchSpace.sample_random_config(space_dict)
            config_tuple = tuple(sorted(config.items()))
            
            if config_tuple not in seen:
                seen.add(config_tuple)
                configs.append(config)
            
            attempts += 1
        
        return configs


# ============================================================================
# RANDOM SEARCH OPTIMIZER
# ============================================================================

class RandomSearchOptimizer:
    """Random search hyperparameter optimizer"""
    
    def __init__(self, X_train, y_train, X_test, y_test, 
                 n_samples=30, space_mode='standard', seed=42):
        """
        Initialize random search optimizer
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            n_samples: Number of random configurations to test
            space_mode: 'standard' or 'focused'
            seed: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_samples = n_samples
        self.seed = seed
        
        # Select search space
        if space_mode == 'focused':
            self.search_space = RandomSearchSpace.FOCUSED_SPACE
        else:
            self.search_space = RandomSearchSpace.SEARCH_SPACE
        
        # Results tracking
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0
        self.configs_tested = []
    
    def train_and_evaluate_config(self, config):
        """
        Train federated model with given config and return accuracy
        
        Args:
            config: Dictionary with hyperparameters
            
        Returns:
            Final test accuracy
        """
        try:
            # Partition data
            clients_data = partition_data_non_iid(
                self.X_train, self.y_train, 
                num_clients=config['num_clients'],
                seed=self.seed
            )
            
            # Get client sizes for weighted averaging
            client_sizes = [len(data[0]) for data in clients_data]
            
            # Initialize server
            server = FederatedServer(learning_rate=config['learning_rate'])
            
            # Initialize clients (ClientManager expects list of tuples)
            client_manager = ClientManager(
                clients_data,
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate']
            )
            
            # Federated training rounds
            for round_num in range(config['num_global_rounds']):
                # Broadcast weights
                global_weights = server.get_global_weights()
                client_manager.set_global_weights(global_weights)
                
                # Train clients locally
                client_manager.train_all_clients(config['local_epochs'], verbose=0)
                
                # Aggregate weights
                client_weights = client_manager.get_client_weights()
                server.aggregate_weights(client_weights, client_sizes)
            
            # Final evaluation
            loss, accuracy = server.evaluate_global_model(self.X_test, self.y_test, verbose=0)
            
            return accuracy
            
        except Exception as e:
            print(f"  ❌ Error with config: {e}")
            return 0.0
    
    def optimize(self, verbose=1):
        """
        Run random search optimization
        
        Args:
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
            
        Returns:
            List of results
        """
        print("\n" + "=" * 80)
        print(f"🎲 RANDOM SEARCH OPTIMIZATION")
        print("=" * 80)
        print(f"Random Samples: {self.n_samples}")
        print(f"Search Space: {len(self.search_space)} parameters")
        print(f"Estimated Runtime: {self.n_samples * 2}-{self.n_samples * 5} minutes")
        print("=" * 80 + "\n")
        
        # Generate random configurations
        configs = RandomSearchSpace.generate_random_configs(
            self.n_samples, 
            self.search_space,
            seed=self.seed
        )
        
        print(f"Generated {len(configs)} unique random configurations\n")
        print("Starting optimization...\n")
        
        # Test each configuration
        for idx, config in enumerate(configs, 1):
            config_str = f"Clients={config['num_clients']}, Rounds={config['num_global_rounds']}, " \
                        f"Epochs={config['local_epochs']}, LR={config['learning_rate']}"
            
            if verbose >= 1:
                print(f"[{idx}/{len(configs)}] Testing: {config_str}")
            
            # Train and evaluate
            accuracy = self.train_and_evaluate_config(config)
            
            # Store results
            result = {
                'config_id': idx,
                **config,
                'accuracy': accuracy
            }
            self.results.append(result)
            self.configs_tested.append(config)
            
            # Update best
            improvement = ""
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = config.copy()
                improvement = f" → 🎯 NEW BEST: {accuracy:.4f}"
            
            if verbose >= 1:
                print(f"           → Accuracy: {accuracy:.4f} (Δ+{accuracy:.4f}){improvement}")
            
        print("\n" + "=" * 80)
        print("✅ Random Search Complete!")
        print("=" * 80 + "\n")
        
        return self.results
    
    def get_best_config(self):
        """Get best configuration found"""
        return self.best_config
    
    def print_results(self, top_n=10):
        """Print top N results"""
        if not self.results:
            print("No results available")
            return
        
        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        print("\n" + "=" * 80)
        print(f"🏆 TOP {min(top_n, len(sorted_results))} CONFIGURATIONS")
        print("=" * 80 + "\n")
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"Rank {i}:")
            print(f"  Clients: {result['num_clients']}, Rounds: {result['num_global_rounds']}, "
                  f"Epochs: {result['local_epochs']}, LR: {result['learning_rate']}, "
                  f"Batch: {result['batch_size']}")
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print()
    
    def save_results(self, output_dir='results'):
        """Save results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best config
        best_config_path = os.path.join(output_dir, 'best_config_random.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_accuracy': float(self.best_accuracy),
                'n_samples_tested': len(self.results),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✅ Best config saved to {best_config_path}")
        
        # Save all results
        results_df = pd.DataFrame(self.results)
        results_csv_path = os.path.join(output_dir, 'random_search_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"✅ All results saved to {results_csv_path}")
        
        # Save as JSON
        results_json_path = os.path.join(output_dir, 'random_search_results.json')
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Results saved to {results_json_path}")
    
    def plot_results(self, output_dir='results'):
        """Create visualization plots"""
        if not self.results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy by number of clients
        ax = axes[0, 0]
        client_acc = df.groupby('num_clients')['accuracy'].agg(['mean', 'max', 'min'])
        x = client_acc.index
        ax.plot(x, client_acc['mean'], 'o-', label='Mean', linewidth=2)
        ax.fill_between(x, client_acc['min'], client_acc['max'], alpha=0.3, label='Min-Max Range')
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Number of Clients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Accuracy by learning rate
        ax = axes[0, 1]
        lr_acc = df.groupby('learning_rate')['accuracy'].agg(['mean', 'max', 'min'])
        x = range(len(lr_acc))
        ax.bar(x, lr_acc['mean'], alpha=0.7)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Learning Rate')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lr:.4f}' for lr in lr_acc.index], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Accuracy distribution
        ax = axes[1, 0]
        ax.hist(df['accuracy'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(self.best_accuracy, color='red', linestyle='--', linewidth=2, label=f'Best: {self.best_accuracy:.4f}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title('Accuracy Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Top 10 configurations
        ax = axes[1, 1]
        top_10 = df.nlargest(10, 'accuracy')
        labels = [f"C{r['num_clients']}_R{r['num_global_rounds']}_E{r['local_epochs']}" 
                  for _, r in top_10.iterrows()]
        ax.barh(range(len(top_10)), top_10['accuracy'], alpha=0.7)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Accuracy')
        ax.set_title('Top 10 Configurations')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'random_search_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Analysis plot saved to {plot_path}")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("🎲 FEDERATED LEARNING - RANDOM SEARCH HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Configuration
    N_SAMPLES = 30  # Number of random configs to test (increase for more thorough search)
    SPACE_MODE = 'standard'  # 'standard' or 'focused'
    
    print(f"\nConfiguration:")
    print(f"  Random Samples: {N_SAMPLES}")
    print(f"  Search Space: {SPACE_MODE}")
    print(f"  Estimated Time: {N_SAMPLES * 2}-{N_SAMPLES * 5} minutes\n")
    
    # Load data
    print("[Setup] Loading datasets...")
    X_train, y_train, X_test, y_test, _ = load_all_data()
    
    print(f"\n[Setup] Data ready:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Run random search
    optimizer = RandomSearchOptimizer(
        X_train, y_train, X_test, y_test,
        n_samples=N_SAMPLES,
        space_mode=SPACE_MODE
    )
    
    results = optimizer.optimize(verbose=1)
    
    # Print results
    optimizer.print_results(top_n=10)
    
    # Save results
    print("\n[Saving] Writing results to disk...")
    optimizer.save_results()
    
    print("\n[Visualization] Creating plots...")
    optimizer.plot_results()
    
    # Print best config
    best_config = optimizer.get_best_config()
    
    print(f"\n{'='*80}")
    print(f"🎯 RANDOM SEARCH OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Best Configuration Found:")
    print(f"   num_clients: {best_config['num_clients']}")
    print(f"   num_global_rounds: {best_config['num_global_rounds']}")
    print(f"   local_epochs: {best_config['local_epochs']}")
    print(f"   batch_size: {best_config['batch_size']}")
    print(f"   learning_rate: {best_config['learning_rate']}")
    print(f"\n   Best Accuracy: {optimizer.best_accuracy:.4f} ({optimizer.best_accuracy*100:.2f}%)")
    print(f"\n📊 Tested {len(results)}/{N_SAMPLES} configurations")
    
    print(f"\n📝 Results saved in results/ folder:")
    print(f"   - best_config_random.json")
    print(f"   - random_search_results.csv")
    print(f"   - random_search_results.json")
    print(f"   - random_search_analysis.png")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
