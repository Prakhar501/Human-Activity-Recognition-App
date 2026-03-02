"""
Run Federated Learning with Optimized Hyperparameters
- Automatically uses best configuration from optimization
- Falls back to defaults if no optimization results found
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_train import CONFIG, load_optimized_config, run_federated_training
from federated_train import load_all_data, compute_centralized_baseline, plot_comparison
from federated_train import main as original_main


def main():
    """Run federated learning with optimized hyperparameters"""
    
    print("\n" + "="*80)
    print("🚀 FEDERATED LEARNING WITH OPTIMIZED HYPERPARAMETERS")
    print("="*80)
    
    # Try to load optimized configuration
    optimized_config_path = 'results/best_config.json'
    if os.path.exists(optimized_config_path):
        print(f"\n✅ Found optimized configuration at: {optimized_config_path}")
        load_optimized_config(optimized_config_path)
    else:
        print(f"\n⚠️ No optimized configuration found at: {optimized_config_path}")
        print("   Using default configuration")
        print("\n   💡 Tip: Run hyperparameter_optimization.py first to find optimal parameters!")
    
    # Continue with standard training using (optimized or default) config
    original_main()


if __name__ == "__main__":
    main()
