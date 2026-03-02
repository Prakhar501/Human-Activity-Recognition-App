"""
Data Utilities for Federated Learning
- Loads and preprocesses datasets (identical to train_combined_model.py)
- Partitions combined dataset across N clients
- Manages test set for global evaluation
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


# ============================================================================
# CONFIGURATION (Match train_combined_model.py exactly)
# ============================================================================

# Dataset paths - automatically adjust based on script location
# Get the parent directory (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Dataset paths - relative to project root
UCI_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
PERSONAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "Personal Dataset", "har_dataset.csv")

# 4 Activities (merged from 6)
ACTIVITIES_4 = ['WALKING', 'SITTING', 'STANDING', 'LAYING']


# ============================================================================
# DATASET LOADING (Replicates train_combined_model.py logic)
# ============================================================================

def load_uci_dataset_4_activities(dataset_dir=UCI_DATASET_DIR):
    """
    Load UCI HAR Dataset and merge walking variants into one class.
    Returns: X_train, y_train, X_test, y_test (scaled, merged to 4 classes)
    """
    
    print("[Dataset] Loading UCI HAR Dataset...")
    
    def load_signals(file_paths):
        signals = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                signals.append([list(map(float, line.split())) for line in f])
        return np.transpose(signals, (1, 2, 0))
    
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z'
    ]
    
    # Load training data
    train_path = os.path.join(dataset_dir, 'UCI HAR Dataset', 'train', 'Inertial Signals')
    X_train = load_signals([
        os.path.join(train_path, f'{signal}_train.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(dataset_dir, 'UCI HAR Dataset', 'train', 'y_train.txt')) as f:
        y_train = np.array([int(line.strip()) - 1 for line in f])
    
    # Load test data
    test_path = os.path.join(dataset_dir, 'UCI HAR Dataset', 'test', 'Inertial Signals')
    X_test = load_signals([
        os.path.join(test_path, f'{signal}_test.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(dataset_dir, 'UCI HAR Dataset', 'test', 'y_test.txt')) as f:
        y_test = np.array([int(line.strip()) - 1 for line in f])
    
    # Merge walking activities (0, 1, 2 -> 0)
    print("[Dataset] Merging WALKING variants (UP/DOWN → WALKING)...")
    y_train_merged = y_train.copy()
    y_test_merged = y_test.copy()
    
    # Original: 0=Walking, 1=Walking_Up, 2=Walking_Down, 3=Sitting, 4=Standing, 5=Laying
    # New: 0=Walking, 1=Sitting, 2=Standing, 3=Laying
    for i in range(len(y_train_merged)):
        if y_train_merged[i] in [0, 1, 2]:
            y_train_merged[i] = 0
        elif y_train_merged[i] == 3:
            y_train_merged[i] = 1
        elif y_train_merged[i] == 4:
            y_train_merged[i] = 2
        elif y_train_merged[i] == 5:
            y_train_merged[i] = 3
    
    for i in range(len(y_test_merged)):
        if y_test_merged[i] in [0, 1, 2]:
            y_test_merged[i] = 0
        elif y_test_merged[i] == 3:
            y_test_merged[i] = 1
        elif y_test_merged[i] == 4:
            y_test_merged[i] = 2
        elif y_test_merged[i] == 5:
            y_test_merged[i] = 3
    
    # Scale to mobile sensor ranges (matches train_combined_model.py)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    for i in range(X_train.shape[0]):
        X_train_scaled[i, :, :3] = X_train[i, :, :3] * 9.81  # Accelerometer
        X_train_scaled[i, :, 3:] = X_train[i, :, 3:] * 3.0   # Gyroscope
    
    for i in range(X_test.shape[0]):
        X_test_scaled[i, :, :3] = X_test[i, :, :3] * 9.81
        X_test_scaled[i, :, 3:] = X_test[i, :, 3:] * 3.0
    
    print(f"[Dataset] UCI loaded: Train={X_train_scaled.shape[0]}, Test={X_test_scaled.shape[0]}")
    
    return X_train_scaled, y_train_merged, X_test_scaled, y_test_merged


def load_personal_dataset(personal_path=PERSONAL_DATASET_PATH, use_fraction=0.70):
    """
    Load and preprocess personal dataset CSV.
    Creates sequences of 128 timesteps with stride 64.
    
    Args:
        personal_path: Path to CSV file
        use_fraction: Fraction of data to use (default 0.70 = 70%)
    
    Returns: X, y (shuffled)
    """
    
    print(f"[Dataset] Loading Personal Dataset from {personal_path}...")
    
    df = pd.read_csv(personal_path)
    print(f"[Dataset] Loaded {len(df)} raw samples")
    
    # Normalize activity labels
    df['label'] = df['label'].str.upper().str.strip()
    
    # Create sequences of 128 timesteps
    SEQUENCE_LENGTH = 128
    sequences = []
    labels = []
    
    # Group by activity and create sequences
    for activity in df['label'].unique():
        activity_data = df[df['label'] == activity]
        sensor_data = activity_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']].values
        
        # Create overlapping windows with stride 64
        for i in range(0, len(sensor_data) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH // 2):
            sequence = sensor_data[i:i + SEQUENCE_LENGTH]
            if len(sequence) == SEQUENCE_LENGTH:
                sequences.append(sequence)
                
                # Map activity to label (0-3)
                if activity == 'WALKING':
                    labels.append(0)
                elif activity == 'SITTING':
                    labels.append(1)
                elif activity == 'STANDING':
                    labels.append(2)
                elif activity == 'LAYING':
                    labels.append(3)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # SHUFFLE FIRST (critical for fair sampling)
    print("[Dataset] Shuffling personal dataset...")
    X, y = shuffle(X, y, random_state=42)
    
    # USE ONLY FRACTION OF DATA
    total_samples = len(X)
    num_samples_to_use = int(total_samples * use_fraction)
    X = X[:num_samples_to_use]
    y = y[:num_samples_to_use]
    
    print(f"[Dataset] Personal: {len(X)} sequences created (using {use_fraction*100:.0f}% of {total_samples} total)")
    
    return X, y


def combine_datasets(X_uci_train, y_uci_train, X_uci_test, y_uci_test, X_personal, y_personal):
    """
    Combine UCI and Personal datasets (same logic as train_combined_model.py).
    Returns: X_train, y_train, X_test, y_test (both categorical), y_test_raw
    """
    
    print("\n[Dataset] Combining datasets...")
    
    # Split personal dataset into train/test (80/20)
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
        X_personal, y_personal, test_size=0.2, random_state=42, stratify=y_personal
    )
    
    print(f"[Dataset] Personal split: Train={len(X_p_train)}, Test={len(X_p_test)}")
    
    # Combine training data
    X_train_combined = np.concatenate([X_uci_train, X_p_train], axis=0)
    y_train_combined = np.concatenate([y_uci_train, y_p_train], axis=0)
    
    # Combine test data
    X_test_combined = np.concatenate([X_uci_test, X_p_test], axis=0)
    y_test_combined = np.concatenate([y_uci_test, y_p_test], axis=0)
    
    # SHUFFLE combined data (critical)
    print("[Dataset] Shuffling combined dataset...")
    X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined, random_state=42)
    X_test_combined, y_test_combined = shuffle(X_test_combined, y_test_combined, random_state=42)
    
    print(f"[Dataset] Combined training set: {len(X_train_combined)} samples")
    print(f"[Dataset] Combined test set: {len(X_test_combined)} samples")
    
    # Convert to categorical (for model training)
    y_train_cat = tf.keras.utils.to_categorical(y_train_combined, num_classes=4)
    y_test_cat = tf.keras.utils.to_categorical(y_test_combined, num_classes=4)
    
    return X_train_combined, y_train_cat, X_test_combined, y_test_cat, y_test_combined


def load_all_data(dataset_dir=UCI_DATASET_DIR, personal_path=PERSONAL_DATASET_PATH):
    """
    Load and combine all datasets in one call.
    Returns: X_train, y_train, X_test, y_test, y_test_raw
    """
    
    print("=" * 80)
    print("[LOADING] COMBINING DATASETS (Federated Learning)")
    print("=" * 80)
    
    # Load UCI
    X_uci_train, y_uci_train, X_uci_test, y_uci_test = load_uci_dataset_4_activities(dataset_dir)
    
    # Load Personal
    X_personal, y_personal = load_personal_dataset(personal_path)
    
    # Combine
    X_train, y_train, X_test, y_test, y_test_raw = combine_datasets(
        X_uci_train, y_uci_train, X_uci_test, y_uci_test, X_personal, y_personal
    )
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Datasets loaded and combined")
    print("=" * 80)
    print(f"Training set shape: {X_train.shape} (labels: one-hot)")
    print(f"Test set shape: {X_test.shape} (labels: one-hot)")
    print("\n")
    
    return X_train, y_train, X_test, y_test, y_test_raw


# ============================================================================
# CLIENT DATA PARTITIONING (Federated Learning)
# ============================================================================

def partition_data_non_iid(X, y, num_clients, seed=42):
    """
    Partition dataset across clients EQUALLY (IID - Independent and Identically Distributed).
    
    Args:
        X: Training data (N, 128, 6)
        y: Training labels (N, 4) - categorical
        num_clients: Number of clients
        seed: Random seed for reproducibility
    
    Returns:
        List of (X_client, y_client) tuples, one per client
    """
    
    np.random.seed(seed)
    
    # Get indices and shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Create equal partitions
    clients_data = []
    samples_per_client = len(X) // num_clients
    
    print(f"[Partitioning] Distributing {len(X)} samples across {num_clients} clients")
    print(f"[Partitioning] Samples per client: {samples_per_client}")
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        
        # Last client gets remaining samples
        if client_id == num_clients - 1:
            end_idx = len(X)
        else:
            end_idx = start_idx + samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        clients_data.append((X_client, y_client))
        
        # Log distribution
        y_raw = np.argmax(y_client, axis=1)
        print(f"  Client {client_id}: {len(X_client)} samples", end="")
        print(f" | Classes: {np.unique(y_raw, return_counts=True)[1]}")
    
    return clients_data


def get_client_batch(X_client, y_client, batch_size=32):
    """
    Create batches from client data.
    
    Args:
        X_client: Client's training data
        y_client: Client's labels
        batch_size: Batch size
    
    Yields:
        (X_batch, y_batch) tuples
    """
    
    num_samples = len(X_client)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X_client[batch_indices], y_client[batch_indices]


if __name__ == "__main__":
    # Test data loading
    print("Testing data utilities...")
    X_train, y_train, X_test, y_test, y_test_raw = load_all_data()
    
    # Test partitioning
    clients_data = partition_data_non_iid(X_train, y_train, num_clients=5)
    print(f"\n[SUCCESS] Created {len(clients_data)} client datasets")
