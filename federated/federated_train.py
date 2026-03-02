"""
Federated Learning Training Coordinator
- Controls FedAvg training loop
- Compares federated vs centralized accuracy
- Logs and visualizes results
- Supports optimized configurations from hyperparameter tuning
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras

# Import custom modules
from data_utils import load_all_data, partition_data_non_iid
from model_utils import create_cnn_model, compile_model, evaluate_model
from client import ClientManager
from server import FederatedServer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default configuration (can be overridden)
CONFIG = {
    # FL Configuration
    'num_clients': 5,
    'num_global_rounds': 10,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.001,
    
    # Data Configuration
    'dataset_dir': 'dataset',
    'personal_dataset_path': 'dataset/Personal Dataset/har_dataset.csv',
    
    # Output
    'output_dir': 'federated/results',
    'save_plots': True,
    'verbose': 1
}


def load_optimized_config(config_path='federated/results/best_config.json'):
    """
    Load optimized configuration from hyperparameter optimization.
    
    Args:
        config_path: Path to best_config.json from optimization
    
    Returns:
        Updated CONFIG dictionary with optimized parameters
    """
    
    global CONFIG
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                optimized = json.load(f)
            
            # Update only the optimizable parameters
            for key in ['num_clients', 'num_global_rounds', 'local_epochs', 
                       'batch_size', 'learning_rate']:
                if key in optimized:
                    CONFIG[key] = optimized[key]
            
            print(f"✅ Loaded optimized configuration from {config_path}")
            return CONFIG
        
        except Exception as e:
            print(f"⚠️ Could not load optimized config: {e}")
            print("   Using default configuration instead")
            return CONFIG
    
    return CONFIG


def print_config():
    """Print FL configuration."""
    
    print("\n" + "="*80)
    print("⚙️  FEDERATED LEARNING CONFIGURATION")
    print("="*80)
    print(f"Number of Clients:      {CONFIG['num_clients']}")
    print(f"Global Rounds:          {CONFIG['num_global_rounds']}")
    print(f"Local Epochs per Round: {CONFIG['local_epochs']}")
    print(f"Batch Size:             {CONFIG['batch_size']}")
    print(f"Learning Rate:          {CONFIG['learning_rate']}")
    print("="*80 + "\n")


def compute_centralized_baseline(X_train, y_train, X_test, y_test):
    """
    Train a centralized model (baseline for comparison).
    Uses SAME training setup as production combined model (train_combined_model.py).
    This ensures a fair comparison with federated learning.
    
    Args:
        X_train: Training features
        y_train: Training labels (one-hot)
        X_test: Test features
        y_test: Test labels (one-hot)
    
    Returns:
        Centralized model accuracy on test set
    """
    
    print("\n" + "="*80)
    print("🔵 CENTRALIZED BASELINE TRAINING")
    print("="*80)
    print("Training centralized model with PRODUCTION setup (50 epochs + callbacks)...\n")
    
    # Create and compile model
    model = create_cnn_model(input_shape=(128, 6), num_classes=4)
    model = compile_model(model, learning_rate=0.001)  # Match production learning rate
    
    # Create output directory if needed
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    baseline_model_path = os.path.join(CONFIG['output_dir'], 'baseline_centralized_model.h5')
    
    # Callbacks - Match production model exactly
    checkpoint = keras.callbacks.ModelCheckpoint(
        baseline_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train for 50 epochs (match production model)
    print(f"Training for 50 epochs on {len(X_train)} samples...\n")
    print("Using callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,  # Match production batch size
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1 if CONFIG['verbose'] > 0 else 0
    )
    
    # Evaluate (with best weights restored by early stopping)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*80}")
    print(f"✅ Centralized Baseline Complete")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Loss:     {loss:.6f}")
    print(f"   Epochs:   {len(history.history['accuracy'])} (stopped early)" if early_stop.stopped_epoch > 0 else f"   Epochs:   50")
    print(f"{'='*80}\n")
    
    return accuracy, history


def run_federated_training(X_train, y_train, X_test, y_test):
    """
    Run federated learning training with FedAvg.
    
    Args:
        X_train: Training features
        y_train: Training labels (one-hot)
        X_test: Test features
        y_test: Test labels (one-hot)
    
    Returns:
        (server, final_accuracy)
    """
    
    print("\n" + "="*80)
    print("🔴 FEDERATED LEARNING TRAINING (FedAvg)")
    print("="*80)
    
    # 1. Partition data among clients
    print("\n[FL] Step 1: Partitioning dataset...")
    clients_data = partition_data_non_iid(
        X_train, y_train,
        num_clients=CONFIG['num_clients'],
        seed=42
    )
    
    # 2. Initialize server and clients
    print("\n[FL] Step 2: Initializing server and clients...")
    server = FederatedServer(learning_rate=CONFIG['learning_rate'])
    client_manager = ClientManager(
        clients_data,
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate']
    )
    
    print("\n[FL] Step 3: Starting federated training loop...\n")
    
    # 3. Federated training loop
    for global_round in range(1, CONFIG['num_global_rounds'] + 1):
        
        print(f"\n{'#'*80}")
        print(f"# GLOBAL ROUND {global_round}/{CONFIG['num_global_rounds']}")
        print(f"{'#'*80}")
        
        # Step 1: Server sends global weights to all clients
        print(f"\n[Server] Broadcasting global model to {CONFIG['num_clients']} clients...")
        global_weights = server.get_global_weights()
        client_manager.set_global_weights(global_weights)
        
        # Step 2: Clients train locally
        print(f"[Clients] Starting local training...")
        client_metrics = client_manager.train_all_clients(
            epochs=CONFIG['local_epochs'],
            verbose=CONFIG['verbose']
        )
        
        # Step 3: Server aggregates weights (FedAvg)
        print(f"\n[Server] Aggregating client updates...")
        client_weights = client_manager.get_client_weights()
        client_sizes = client_manager.get_client_sizes()
        
        server.aggregate_weights(client_weights, client_sizes=client_sizes)
        
        # Step 4: Evaluate global model
        print(f"\n[Server] Evaluating global model...")
        round_summary = server.complete_round(X_test, y_test, client_metrics=client_metrics)
        server.log_round_summary(round_summary, client_metrics=client_metrics)
    
    print(f"\n{'='*80}")
    print(f"✅ Federated Learning Complete")
    print(f"{'='*80}\n")
    
    final_accuracy = server.history['global_accuracies'][-1]
    
    return server, final_accuracy


def plot_comparison(server_history, centralized_history, centralized_accuracy, output_dir):
    """
    Plot federated vs centralized training curves.
    
    Args:
        server_history: FedAvg server history dict
        centralized_history: Centralized training history object
        centralized_accuracy: Final centralized accuracy
        output_dir: Directory to save plots
    """
    
    print("\n[Visualization] Creating comparison plots...\n")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0]
    
    # Federated accuracy
    federated_rounds = server_history['global_rounds']
    federated_acc = server_history['global_accuracies']
    ax1.plot(federated_rounds, federated_acc, 'o-', linewidth=2, markersize=6,
             label='Federated Learning (FedAvg)', color='#FF6B6B')
    
    # Centralized accuracy per epoch (scaled to FL rounds)
    centralized_acc_per_epoch = centralized_history.history['accuracy']
    centralized_val_acc = centralized_history.history['val_accuracy']
    
    epochs_per_round = CONFIG['local_epochs']
    rounds_epochs = np.arange(0, len(centralized_val_acc), epochs_per_round)
    centralized_acc_at_rounds = [centralized_val_acc[i] if i < len(centralized_val_acc) else centralized_val_acc[-1]
                                 for i in rounds_epochs]
    
    ax1.plot(rounds_epochs[:len(centralized_acc_at_rounds)], centralized_acc_at_rounds, 's-', linewidth=2, markersize=6,
             label='Centralized (Baseline)', color='#4ECDC4')
    
    ax1.set_xlabel('Global Rounds (or Epoch Groups)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Federated vs Centralized: Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Loss Comparison
    ax2 = axes[1]
    
    # Federated loss
    federated_loss = server_history['global_losses']
    ax2.plot(federated_rounds, federated_loss, 'o-', linewidth=2, markersize=6,
             label='Federated Learning (FedAvg)', color='#FF6B6B')
    
    # Centralized loss per epoch
    centralized_loss = centralized_history.history['loss']
    centralized_val_loss = centralized_history.history['val_loss']
    loss_at_rounds = [centralized_val_loss[i] if i < len(centralized_val_loss) else centralized_val_loss[-1]
                      for i in rounds_epochs]
    
    ax2.plot(rounds_epochs[:len(loss_at_rounds)], loss_at_rounds, 's-', linewidth=2, markersize=6,
             label='Centralized (Baseline)', color='#4ECDC4')
    
    ax2.set_xlabel('Global Rounds (or Epoch Groups)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Federated vs Centralized: Loss Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'federated_vs_centralized.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved comparison plot: {plot_path}")
    
    # Summary statistics plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Final Accuracy']
    federated_val = [server_history['global_accuracies'][-1]]
    centralized_val = [centralized_accuracy]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, federated_val, width, label='Federated Learning', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, centralized_val, width, label='Centralized Baseline', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Final Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(output_dir, 'accuracy_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved summary plot: {summary_path}")
    
    plt.close('all')


def main():
    """Main federated learning training pipeline."""
    
    print("\n" + "="*80)
    print("🎯 FEDERATED LEARNING IMPLEMENTATION (FedAvg)")
    print("   Human Activity Recognition (HAR) - 4 Activities")
    print("="*80)
    
    # Print configuration
    print_config()
    
    # 1. Load and prepare data
    print("\n[Main] Loading datasets...")
    X_train, y_train, X_test, y_test, _ = load_all_data(
        dataset_dir=CONFIG['dataset_dir'],
        personal_path=CONFIG['personal_dataset_path']
    )
    
    print(f"[Main] Data prepared:")
    print(f"       Training: {X_train.shape[0]} samples, {X_train.shape[1:]} shape")
    print(f"       Test:     {X_test.shape[0]} samples, {X_test.shape[1:]} shape")
    
    # 2. Compute centralized baseline
    centralized_accuracy, centralized_history = compute_centralized_baseline(
        X_train, y_train, X_test, y_test
    )
    
    # 3. Run federated learning
    server, federated_accuracy = run_federated_training(
        X_train, y_train, X_test, y_test
    )
    
    # 4. Print final comparison
    server.print_final_results(centralized_accuracy=centralized_accuracy)
    
    # 5. Generate plots
    if CONFIG['save_plots']:
        output_dir = CONFIG['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        plot_comparison(
            server.history,
            centralized_history,
            centralized_accuracy,
            output_dir
        )
    
    # 6. Save results summary
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'centralized_accuracy': float(centralized_accuracy),
        'federated_accuracy': float(federated_accuracy),
        'accuracy_difference': float(federated_accuracy - centralized_accuracy),
        'num_rounds': len(server.history['global_rounds']),
        'global_losses': [float(x) for x in server.history['global_losses']],
        'global_accuracies': [float(x) for x in server.history['global_accuracies']]
    }
    
    results_path = os.path.join(CONFIG['output_dir'], 'results_summary.txt')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEDERATED LEARNING RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {results_summary['timestamp']}\n\n")
        
        f.write("Configuration:\n")
        for key, val in CONFIG.items():
            f.write(f"  {key}: {val}\n")
        
        f.write(f"\nResults:\n")
        f.write(f"  Centralized Accuracy: {centralized_accuracy:.4f} ({centralized_accuracy*100:.2f}%)\n")
        f.write(f"  Federated Accuracy:   {federated_accuracy:.4f} ({federated_accuracy*100:.2f}%)\n")
        f.write(f"  Difference: {results_summary['accuracy_difference']:+.4f} ({results_summary['accuracy_difference']*100:+.2f}%)\n")
        
        f.write(f"\nGlobal Rounds Accuracy History:\n")
        for round_num, acc in zip(server.history['global_rounds'], server.history['global_accuracies']):
            f.write(f"  Round {round_num}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    print(f"\n[Main] Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("✅ FEDERATED LEARNING TRAINING COMPLETE!")
    print("="*80)
    print("\nTo analyze results:")
    print(f"  1. Check plots in: {CONFIG['output_dir']}/")
    print(f"  2. Review summary: {results_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
