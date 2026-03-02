"""
Retrain HAR Model - Pure TFLite Compatible (No GRU/LSTM)
Uses only CNN layers which are fully supported in TFLite
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
DATASET_DIR = "d:/HAR app/dataset"
OUTPUT_DIR = "d:/HAR app/trained_model"

# Activities
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

print("=" * 70)
print("Mobile HAR Model Training (Pure TFLite Compatible - CNN Only)")
print("=" * 70)


def load_mobile_compatible_data():
    """Load only accelerometer and gyroscope data (6 features)"""
    
    print("\n[1/4] Loading mobile-compatible dataset...")
    
    def load_signals(file_paths):
        """Load signal data from multiple files"""
        signals = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                signals.append([list(map(float, line.split())) for line in f])
        return np.transpose(signals, (1, 2, 0))
    
    # Only use body accelerometer and gyroscope (6 signals)
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z'
    ]
    
    # Load training data
    train_path = os.path.join(DATASET_DIR, 'UCI HAR Dataset', 'train', 'Inertial Signals')
    X_train = load_signals([
        os.path.join(train_path, f'{signal}_train.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(DATASET_DIR, 'UCI HAR Dataset', 'train', 'y_train.txt')) as f:
        y_train = np.array([int(line.strip()) - 1 for line in f])
    
    # Load test data
    test_path = os.path.join(DATASET_DIR, 'UCI HAR Dataset', 'test', 'Inertial Signals')
    X_test = load_signals([
        os.path.join(test_path, f'{signal}_test.txt') 
        for signal in signal_types
    ])
    
    with open(os.path.join(DATASET_DIR, 'UCI HAR Dataset', 'test', 'y_test.txt')) as f:
        y_test = np.array([int(line.strip()) - 1 for line in f])
    
    print(f"   ✓ Training: {X_train.shape} (samples, timesteps, features)")
    print(f"   ✓ Test: {X_test.shape}")
    print(f"   ✓ Using 6 features: Accel(x,y,z) + Gyro(x,y,z)")
    
    # Apply mobile-like preprocessing
    print("\n[2/4] Applying mobile-like preprocessing...")
    X_train = apply_mobile_preprocessing(X_train)
    X_test = apply_mobile_preprocessing(X_test)
    
    print(f"   ✓ Applied scaling to realistic mobile ranges")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=6)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=6)
    
    return X_train, y_train_cat, X_test, y_test_cat, y_test


def apply_mobile_preprocessing(X):
    """Scale to realistic mobile sensor ranges"""
    X_processed = X.copy()
    
    for i in range(X.shape[0]):
        # Accelerometer (first 3 features) - scale to m/s²
        for j in range(3):
            X_processed[i, :, j] = X[i, :, j] * 9.81
        
        # Gyroscope (last 3 features) - scale to rad/s
        for j in range(3, 6):
            X_processed[i, :, j] = X[i, :, j] * 3.0
    
    return X_processed


def create_cnn_model(input_shape=(128, 6), num_classes=6):
    """
    Create CNN-only model (fully TFLite compatible)
    No GRU/LSTM - only Conv1D, Dense, Dropout, BatchNorm, Pooling
    """
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Normalization layer
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
        layers.GlobalAveragePooling1D(),  # Instead of Flatten to reduce params
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model():
    """Train the CNN model"""
    
    # Load data
    X_train, y_train, X_test, y_test, y_test_raw = load_mobile_compatible_data()
    
    print(f"\n[3/4] Creating and training CNN model...")
    
    # Create model
    model = create_cnn_model(input_shape=(128, 6), num_classes=6)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n   Model Summary:")
    model.summary()
    
    # Adapt normalization layer
    print(f"\n   Adapting normalization layer...")
    norm_layer = model.layers[0]
    norm_layer.adapt(X_train)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'har_model_cnn.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n   Training started...")
    print(f"   Architecture: Pure CNN (TFLite compatible)")
    print(f"   Batch size: 64")
    print(f"   Max epochs: 100")
    print(f"   " + "=" * 60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print(f"\n[4/4] Evaluating model...")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Detailed evaluation
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"\n   Per-class Accuracy:")
    for i, activity in enumerate(ACTIVITIES):
        mask = y_test_raw == i
        if mask.sum() > 0:
            class_acc = (predicted_classes[mask] == i).mean()
            print(f"      {activity:20s}: {class_acc*100:.1f}%")
    
    # Save final model
    model.save(os.path.join(OUTPUT_DIR, 'har_model_cnn_final.h5'))
    print(f"\n✅ Model saved to: {OUTPUT_DIR}")
    print(f"   - har_model_cnn.h5 (best checkpoint)")
    print(f"   - har_model_cnn_final.h5 (final)")
    
    return model, history


def convert_to_tflite(model_path):
    """Convert to TFLite (should work without Flex ops)"""
    
    print(f"\n" + "=" * 70)
    print("Converting to TFLite (Pure CNN - No Flex Ops)")
    print("=" * 70)
    
    model = keras.models.load_model(model_path)
    
    # Convert to TFLite - standard conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # NO Flex ops needed!
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(OUTPUT_DIR, 'har_model_cnn.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✓ TFLite model saved: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    
    # Test the TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✓ Input shape: {input_details[0]['shape']}")
    print(f"✓ Output shape: {output_details[0]['shape']}")
    
    # Test inference
    test_input = np.random.randn(1, 128, 6).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_idx = np.argmax(output[0])
    confidence = output[0][predicted_idx] * 100
    
    print(f"✓ Test prediction: {ACTIVITIES[predicted_idx]} ({confidence:.1f}%)")
    print(f"\n✅ Pure TFLite model ready! No Flex delegate needed!")
    print(f"   Copy: {tflite_path}")
    print(f"   To: flutter_har_app/assets/models/har_model.tflite")
    
    return tflite_path


if __name__ == '__main__':
    try:
        # Train CNN model
        model, history = train_model()
        
        # Convert to TFLite
        best_model_path = os.path.join(OUTPUT_DIR, 'har_model_cnn.h5')
        tflite_path = convert_to_tflite(best_model_path)
        
        print(f"\n" + "=" * 70)
        print("✅ ALL DONE!")
        print("=" * 70)
        print(f"\nPure CNN model - No Flex ops needed!")
        print(f"Should work perfectly in Flutter with standard TFLite!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
