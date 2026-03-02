# Centralized Baseline Fix - March 2, 2026

## Problem Identified

The centralized baseline model was achieving only **66% accuracy** while the production combined model achieves **83% accuracy**. This created an unfair comparison for federated learning experiments.

## Root Cause

The centralized baseline in `federated_train.py` was missing critical training optimizations:

### Old Baseline (66% accuracy)
- ❌ No callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
- ❌ Training epochs based on federated config (num_rounds × local_epochs)
- ❌ Learning rate from federated config (variable)
- ❌ Batch size from federated config (variable)

### Production Model (83% accuracy)
- ✅ ModelCheckpoint (saves best model)
- ✅ EarlyStopping (patience=15, prevents overfitting)
- ✅ ReduceLROnPlateau (adaptive learning rate)
- ✅ Fixed 50 epochs with callbacks
- ✅ Fixed learning rate: 0.001
- ✅ Fixed batch size: 32

## Solution Implemented

Updated `compute_centralized_baseline()` function in `federated_train.py` to match the production model exactly:

### Changes Made

1. **Added Keras Callbacks** (lines 131-152):
   ```python
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
   ```

2. **Fixed Training Parameters**:
   - Epochs: **50** (not variable based on federated config)
   - Batch size: **32** (consistent with production)
   - Learning rate: **0.001** (consistent with production)
   - Callbacks: All three production callbacks included

3. **Added Keras Import**:
   ```python
   from tensorflow import keras
   ```

## Expected Results

With these changes, the centralized baseline should now achieve:
- **~83% accuracy** (matching production combined model)
- Fair comparison with federated learning
- Better insight into whether FL actually improves accuracy

## Next Steps

1. **Run New Baseline Test**:
   ```bash
   cd federated
   python federated_train.py
   ```

2. **Compare Results**:
   - Old baseline: 66%
   - New baseline: ~83% (expected)
   - Federated: 90% (from previous run)

3. **Re-run Hyperparameter Optimization** (optional):
   Since the baseline is now stronger, you may want to re-run optimization to see if federated learning can still outperform the properly-trained centralized model.

## Files Modified

- `federated/federated_train.py` (lines 15, 96-173)

## Validation

The code changes:
- ✅ Match production model architecture exactly
- ✅ Use same data preprocessing
- ✅ Use same training callbacks
- ✅ Use same hyperparameters
- ✅ Provide fair baseline for FL comparison
