"""
Train Combined HAR Model - UCI + Personal Dataset
Merge both datasets and train one model with 4 activities
Compare accuracy with original 6-activity model
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Configuration
UCI_DATASET_DIR = "d:/har app-2/Human-Activity-Recognition-App/dataset"
PERSONAL_DATASET_PATH = "d:/har app-2/Human-Activity-Recognition-App/dataset/Personal Dataset/har_dataset.csv"
OUTPUT_DIR = "d:/har app-2/Human-Activity-Recognition-App/trained_model"
OLD_MODEL_PATH = "d:/har app-2/Human-Activity-Recognition-App/trained_model/har_model_cnn.h5"

# 4 Activities (merged)
ACTIVITIES_4 = ['WALKING', 'SITTING', 'STANDING', 'LAYING']
ACTIVITY_MAP = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 0,  # Merge to WALKING
    'WALKING_DOWNSTAIRS': 0,  # Merge to WALKING
    'SITTING': 1,
    'STANDING': 2,
    'LAYING': 3
}

print("=" * 80)
print("🚀 COMBINED HAR MODEL TRAINING (UCI + Personal Dataset)")
print("=" * 80)
print(f"Activities: {ACTIVITIES_4}")
print("=" * 80)


def load_uci_dataset_4_activities():
    """Load UCI dataset and merge walking variants into one class"""
    
    print("\n[1/6] Loading UCI HAR Dataset...")
    
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
    train_path = os.path.join(UCI_DATASET_DIR, 'UCI HAR Dataset', 'train', 'Inertial Signals')
    X_train = load_signals([
        os.path.join(train_path, f'{signal}_train.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(UCI_DATASET_DIR, 'UCI HAR Dataset', 'train', 'y_train.txt')) as f:
        y_train = np.array([int(line.strip()) - 1 for line in f])
    
    # Load test data
    test_path = os.path.join(UCI_DATASET_DIR, 'UCI HAR Dataset', 'test', 'Inertial Signals')
    X_test = load_signals([
        os.path.join(test_path, f'{signal}_test.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(UCI_DATASET_DIR, 'UCI HAR Dataset', 'test', 'y_test.txt')) as f:
        y_test = np.array([int(line.strip()) - 1 for line in f])
    
    # Merge walking activities (0, 1, 2 -> 0)
    print("   ✓ Merging WALKING_UPSTAIRS and WALKING_DOWNSTAIRS into WALKING...")
    y_train_merged = y_train.copy()
    y_test_merged = y_test.copy()
    
    # Map: 0=Walking, 1=Walking_Up, 2=Walking_Down, 3=Sitting, 4=Standing, 5=Laying
    # New: 0=Walking, 1=Sitting, 2=Standing, 3=Laying
    for i in range(len(y_train_merged)):
        if y_train_merged[i] in [0, 1, 2]:  # All walking variants
            y_train_merged[i] = 0
        elif y_train_merged[i] == 3:  # Sitting
            y_train_merged[i] = 1
        elif y_train_merged[i] == 4:  # Standing
            y_train_merged[i] = 2
        elif y_train_merged[i] == 5:  # Laying
            y_train_merged[i] = 3
    
    for i in range(len(y_test_merged)):
        if y_test_merged[i] in [0, 1, 2]:  # All walking variants
            y_test_merged[i] = 0
        elif y_test_merged[i] == 3:  # Sitting
            y_test_merged[i] = 1
        elif y_test_merged[i] == 4:  # Standing
            y_test_merged[i] = 2
        elif y_test_merged[i] == 5:  # Laying
            y_test_merged[i] = 3
    
    # Scale to mobile sensor ranges
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    for i in range(X_train.shape[0]):
        X_train_scaled[i, :, :3] = X_train[i, :, :3] * 9.81  # Accelerometer
        X_train_scaled[i, :, 3:] = X_train[i, :, 3:] * 3.0   # Gyroscope
    
    for i in range(X_test.shape[0]):
        X_test_scaled[i, :, :3] = X_test[i, :, :3] * 9.81
        X_test_scaled[i, :, 3:] = X_test[i, :, 3:] * 3.0
    
    print(f"   ✓ UCI Dataset loaded: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"   ✓ Activities distribution:")
    for act_idx, act_name in enumerate(ACTIVITIES_4):
        count = np.sum(y_train_merged == act_idx)
        print(f"      {act_name}: {count} samples")
    
    return X_train_scaled, y_train_merged, X_test_scaled, y_test_merged


def load_personal_dataset():
    """Load and preprocess personal dataset CSV"""
    
    print("\n[2/6] Loading Personal Dataset...")
    
    df = pd.read_csv(PERSONAL_DATASET_PATH)
    print(f"   ✓ Loaded {len(df)} raw samples")
    
    # Normalize activity labels
    df['label'] = df['label'].str.upper().str.strip()
    
    print(f"   ✓ Activities found: {df['label'].unique()}")
    print(f"   ✓ Distribution:")
    for activity in df['label'].unique():
        count = len(df[df['label'] == activity])
        print(f"      {activity}: {count} samples")
    
    # Create sequences of 128 timesteps
    SEQUENCE_LENGTH = 128
    sequences = []
    labels = []
    
    # Group by activity and create sequences
    for activity in df['label'].unique():
        activity_data = df[df['label'] == activity]
        
        # Extract sensor columns
        sensor_data = activity_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']].values
        
        # Create overlapping windows
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
    
    # 🎲 IMPORTANT: SHUFFLE THE DATA
    print("\n   🎲 Shuffling personal dataset to avoid bias...")
    X, y = shuffle(X, y, random_state=42)
    
    print(f"\n   ✓ Created {len(X)} sequences (128 timesteps each)")
    print(f"   ✓ Shape: {X.shape}")
    print(f"   ✓ Shuffled activities distribution:")
    for act_idx, act_name in enumerate(ACTIVITIES_4):
        count = np.sum(y == act_idx)
        print(f"      {act_name}: {count} sequences")
    
    return X, y


def combine_datasets(X_uci_train, y_uci_train, X_uci_test, y_uci_test, X_personal, y_personal):
    """Combine UCI and Personal datasets"""
    
    print("\n[3/6] Combining datasets...")
    
    # Split personal dataset into train/test (80/20)
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
        X_personal, y_personal, test_size=0.2, random_state=42, stratify=y_personal
    )
    
    print(f"   Personal Dataset Split:")
    print(f"      Train: {len(X_p_train)} samples")
    print(f"      Test: {len(X_p_test)} samples")
    
    # Combine training data
    X_train_combined = np.concatenate([X_uci_train, X_p_train], axis=0)
    y_train_combined = np.concatenate([y_uci_train, y_p_train], axis=0)
    
    # Combine test data
    X_test_combined = np.concatenate([X_uci_test, X_p_test], axis=0)
    y_test_combined = np.concatenate([y_uci_test, y_p_test], axis=0)
    
    # 🎲 SHUFFLE COMBINED DATA
    print("\n   🎲 Shuffling combined dataset...")
    X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined, random_state=42)
    X_test_combined, y_test_combined = shuffle(X_test_combined, y_test_combined, random_state=42)
    
    print(f"\n   ✓ Combined Training Set: {len(X_train_combined)} samples")
    print(f"   ✓ Combined Test Set: {len(X_test_combined)} samples")
    print(f"\n   📊 Final Distribution (Training):")
    for act_idx, act_name in enumerate(ACTIVITIES_4):
        count = np.sum(y_train_combined == act_idx)
        percentage = (count / len(y_train_combined)) * 100
        print(f"      {act_name}: {count} samples ({percentage:.1f}%)")
    
    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train_combined, num_classes=4)
    y_test_cat = keras.utils.to_categorical(y_test_combined, num_classes=4)
    
    return X_train_combined, y_train_cat, X_test_combined, y_test_cat, y_test_combined


def create_cnn_model(input_shape=(128, 6), num_classes=4):
    """Create CNN model for 4 activities"""
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Normalization(),
        
        # Conv Block 1
        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 2
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 3
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 4
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


def evaluate_old_model(X_test, y_test):
    """Evaluate old 6-activity model on 4-activity test data"""
    
    print("\n[4/6] Evaluating OLD model (6 activities)...")
    
    if not os.path.exists(OLD_MODEL_PATH):
        print("   ⚠️ Old model not found, skipping comparison")
        return None
    
    try:
        old_model = keras.models.load_model(OLD_MODEL_PATH)
        
        # Map 4-class labels to 6-class for prediction
        # We'll just test on merged data
        y_test_6class = y_test.copy()
        # Keep walking=0, sitting=3, standing=4, laying=5
        for i in range(len(y_test_6class)):
            if y_test_6class[i] == 1:  # Sitting
                y_test_6class[i] = 3
            elif y_test_6class[i] == 2:  # Standing
                y_test_6class[i] = 4
            elif y_test_6class[i] == 3:  # Laying
                y_test_6class[i] = 5
        
        y_test_6class_cat = keras.utils.to_categorical(y_test_6class, num_classes=6)
        
        loss, accuracy = old_model.evaluate(X_test, y_test_6class_cat, verbose=0)
        print(f"   📊 Old Model Accuracy: {accuracy*100:.2f}%")
        
        return accuracy * 100
        
    except Exception as e:
        print(f"   ⚠️ Error evaluating old model: {e}")
        return None


def train_combined_model(X_train, y_train, X_test, y_test):
    """Train new combined model"""
    
    print("\n[5/6] Training NEW Combined Model (4 activities)...")
    
    # Create model
    model = create_cnn_model(num_classes=4)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n   Model Architecture:")
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'har_model_combined_4act.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
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
    print("\n   🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Final evaluation
    print("\n   📊 Final Evaluation on Test Set:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   ✅ NEW Combined Model Accuracy: {accuracy*100:.2f}%")
    
    return model, history, accuracy * 100


def convert_to_tflite(model):
    """Convert model to TFLite"""
    
    print("\n[6/6] Converting to TFLite...")
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    tflite_path = os.path.join(OUTPUT_DIR, 'har_model_combined_4act.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"   ✅ TFLite model saved: {tflite_path}")
    
    return tflite_path


def main():
    """Main training pipeline"""
    
    # Load datasets
    X_uci_train, y_uci_train, X_uci_test, y_uci_test = load_uci_dataset_4_activities()
    X_personal, y_personal = load_personal_dataset()
    
    # Combine
    X_train, y_train, X_test, y_test, y_test_raw = combine_datasets(
        X_uci_train, y_uci_train, X_uci_test, y_uci_test, X_personal, y_personal
    )
    
    # Evaluate old model
    old_accuracy = evaluate_old_model(X_test, y_test_raw)
    
    # Train new model
    model, history, new_accuracy = train_combined_model(X_train, y_train, X_test, y_test)
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(model)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 ACCURACY COMPARISON:")
    if old_accuracy:
        print(f"   Old Model (6 activities): {old_accuracy:.2f}%")
    print(f"   New Combined Model (4 activities): {new_accuracy:.2f}%")
    if old_accuracy:
        improvement = new_accuracy - old_accuracy
        print(f"   Improvement: {improvement:+.2f}%")
    
    print(f"\n📁 Output Files:")
    print(f"   H5 Model: {OUTPUT_DIR}/har_model_combined_4act.h5")
    print(f"   TFLite: {tflite_path}")
    
    print(f"\n🎯 Activities (4):")
    for i, act in enumerate(ACTIVITIES_4):
        print(f"   {i}: {act}")
    
    print("\n✅ Now update the Flutter app to use the new model!")
    print("=" * 80)


if __name__ == "__main__":
    main()
