"""
Complete All Missing Experiments for Research Tables
- Federated training with round-by-round logging
- UCI-HAR only training
- Personal/Custom dataset only training
"""

import os
import sys
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add federated directory to path
sys.path.append('federated')

from federated.data_utils import load_all_data
from federated.model_utils import create_cnn_model, compile_model
from federated.server import FederatedServer
from federated.client import ClientManager
from federated.data_utils import partition_data_non_iid

# Configuration
DATASET_DIR = "dataset"
PERSONAL_PATH = "dataset/Personal Dataset/har_dataset.csv"
OUTPUT_DIR = "federated/results"

print("="*80)
print("🚀 RUNNING ALL MISSING EXPERIMENTS")
print("="*80)


# =============================================================================
# EXPERIMENT 1: Federated Training with Round-by-Round Logging
# =============================================================================

def run_federated_with_logging():
    """Run federated training and log accuracy at specific rounds"""
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT 1: Federated Training (Round-by-Round Logging)")
    print("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test, _ = load_all_data(DATASET_DIR, PERSONAL_PATH)
    
    # Configuration
    config = {
        'num_clients': 5,
        'num_global_rounds': 75,
        'local_epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    # Partition data
    clients_data = partition_data_non_iid(X_train, y_train, num_clients=config['num_clients'], seed=42)
    
    # Initialize server and clients
    server = FederatedServer(learning_rate=config['learning_rate'])
    client_manager = ClientManager(clients_data, batch_size=config['batch_size'], learning_rate=config['learning_rate'])
    
    # Track specific rounds
    target_rounds = [1, 5, 10, 20, 31, 50, 55, 65, 75]
    round_results = {}
    
    print(f"\n[FL] Training for {config['num_global_rounds']} rounds...")
    print(f"[FL] Will log accuracy at rounds: {target_rounds}\n")
    
    # Federated training loop
    for global_round in range(1, config['num_global_rounds'] + 1):
        
        # Server sends global weights
        global_weights = server.get_global_weights()
        client_manager.set_global_weights(global_weights)
        
        # Clients train locally
        client_metrics = client_manager.train_all_clients(epochs=config['local_epochs'], verbose=0)
        
        # Server aggregates
        client_weights = client_manager.get_client_weights()
        client_sizes = client_manager.get_client_sizes()
        server.aggregate_weights(client_weights, client_sizes=client_sizes)
        
        # Evaluate
        round_summary = server.complete_round(X_test, y_test, client_metrics=client_metrics)
        
        # Log specific rounds
        if global_round in target_rounds:
            accuracy = round_summary['global_accuracy']
            round_results[global_round] = accuracy
            print(f"  Round {global_round:2d}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "federated_round_by_round.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': config,
            'target_rounds': target_rounds,
            'accuracies': {str(k): float(v) for k, v in round_results.items()}
        }, f, indent=4)
    
    print(f"\n✅ Results saved to: {output_file}")
    return round_results


# =============================================================================
# EXPERIMENT 2: UCI-HAR Dataset Only
# =============================================================================

def train_uci_only():
    """Train model on UCI-HAR dataset only"""
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT 2: UCI-HAR Dataset Only")
    print("="*80)
    
    from train_combined_model import load_uci_dataset_4_activities
    
    # Load only UCI dataset
    X_train, y_train, X_test, y_test = load_uci_dataset_4_activities()
    
    print(f"\n[UCI] Training samples: {X_train.shape[0]}")
    print(f"[UCI] Test samples: {X_test.shape[0]}")
    
    # Convert labels to one-hot
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)
    
    # Create and compile model
    model = create_cnn_model(input_shape=(128, 6), num_classes=4)
    model = compile_model(model, learning_rate=0.001)
    
    # Callbacks
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'uci_only_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print("\n[UCI] Training model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    train_acc = history.history['accuracy'][-1]
    
    print(f"\n[UCI] Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    results = {
        'dataset': 'UCI-HAR Only',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(accuracy),
        'epochs_trained': len(history.history['accuracy'])
    }
    
    output_file = os.path.join(OUTPUT_DIR, "uci_only_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to: {output_file}")
    return results


# =============================================================================
# EXPERIMENT 3: Personal/Custom Dataset Only
# =============================================================================

def train_personal_only():
    """Train model on Personal dataset only"""
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT 3: Personal/Custom Dataset Only")
    print("="*80)
    
    from train_combined_model import load_personal_dataset
    
    # Load only personal dataset
    X_personal, y_personal = load_personal_dataset()
    
    print(f"\n[Personal] Total samples: {X_personal.shape[0]}")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_personal, y_personal,
        test_size=0.2,
        random_state=42,
        stratify=y_personal
    )
    
    print(f"[Personal] Training samples: {X_train.shape[0]}")
    print(f"[Personal] Test samples: {X_test.shape[0]}")
    
    # Convert labels to one-hot
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)
    
    # Create and compile model
    model = create_cnn_model(input_shape=(128, 6), num_classes=4)
    model = compile_model(model, learning_rate=0.001)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'personal_only_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print("\n[Personal] Training model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    train_acc = history.history['accuracy'][-1]
    
    print(f"\n[Personal] Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    results = {
        'dataset': 'Personal/Custom Only',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(accuracy),
        'epochs_trained': len(history.history['accuracy'])
    }
    
    output_file = os.path.join(OUTPUT_DIR, "personal_only_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to: {output_file}")
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("\n🎯 Starting all experiments...")
    print("   This will take approximately 30-45 minutes.\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Federated round-by-round
    try:
        print("\n" + "🔵"*40)
        federated_results = run_federated_with_logging()
        all_results['federated'] = federated_results
        print("🔵"*40)
    except Exception as e:
        print(f"❌ Federated experiment failed: {e}")
    
    # Experiment 2: UCI-HAR only
    try:
        print("\n" + "🟢"*40)
        uci_results = train_uci_only()
        all_results['uci_only'] = uci_results
        print("🟢"*40)
    except Exception as e:
        print(f"❌ UCI-only experiment failed: {e}")
    
    # Experiment 3: Personal only
    try:
        print("\n" + "🟡"*40)
        personal_results = train_personal_only()
        all_results['personal_only'] = personal_results
        print("🟡"*40)
    except Exception as e:
        print(f"❌ Personal-only experiment failed: {e}")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved in: {OUTPUT_DIR}/")
    print(f"📄 Summary file: {summary_file}\n")
