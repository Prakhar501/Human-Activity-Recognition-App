#!/usr/bin/env python3
"""
Quick test script for centralized baseline with production model setup
"""

import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

# Add federated directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'federated'))

from federated.data_utils import load_all_data
from federated.model_utils import create_cnn_model, compile_model

print("="*80)
print("[BASELINE TEST] Production Model Setup")
print("="*80)

# Load data
print("\n[1/3] Loading datasets...")
X_train, y_train, X_test, y_test, _ = load_all_data()
print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Create model  
print("\n[2/3] Creating model...")
model = create_cnn_model(input_shape=(128, 6), num_classes=4)
model = compile_model(model, learning_rate=0.001)

# Create callbacks
print("\n[3/3] Training with production setup (50 epochs + callbacks)...")
print("   Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\n")

os.makedirs('federated/results', exist_ok=True)

checkpoint = keras.callbacks.ModelCheckpoint(
    'federated/results/baseline_test_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=0  # Minimal output
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*80)
print("\n[SUCCESS] BASELINE TRAINING COMPLETE")
print("="*80)
print(f"   Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Final Loss:    {loss:.6f}")
print(f"   Epochs Trained: {len(history.history['accuracy'])}")
print(f"   Early Stopped:  {early_stop.stopped_epoch > 0}")
print("="*80)

# Compare with production
print("\n[COMPARISON]:")
print(f"   Production Combined Model: 83.22%")
print(f"   New Baseline (this run):   {accuracy*100:.2f}%")
print(f"   Difference:                {(accuracy*100) - 83.22:+.2f}%")

if accuracy > 0.80:
    print("\n[SUCCESS] Baseline is now production-level (~83%)!")
else:
    print(f"\n[NOTE] Baseline is still lower than expected ({accuracy*100:.2f}% < 83%)")
