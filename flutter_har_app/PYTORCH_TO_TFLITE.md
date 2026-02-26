# PyTorch to TensorFlow Lite Model Conversion Guide

## Why TFLite Instead of PyTorch?

**Problem:** `pytorch_mobile` Flutter package is outdated and incompatible with modern Android Gradle Plugin (AGP 8.1+).

**Solution:** Convert PyTorch model to TensorFlow Lite format, which has better Flutter support.

## Current Status

✅ **App now runs in DEMO MODE** - Shows simulated predictions based on sensor data patterns
✅ **Sensors working** - Collects accelerometer, gyroscope data
⚠️ **ML Model** - Need to convert PyTorch model to TFLite format

## Demo Mode

App will work WITHOUT actual ML model, using intelligent simulation:
- Analyzes sensor data movement patterns
- Simulates realistic activity predictions
- Shows confidence scores
- Perfect for testing UI and sensors

## Steps to Convert PyTorch Model to TFLite

### Option 1: Using ONNX (Recommended)

```python
# Install required packages
pip install torch onnx onnx-tf tensorflow

# Step 1: Load your PyTorch model
import torch
model = torch.load('har_model.ptl', map_location='cpu')
model.eval()

# Step 2: Export to ONNX
dummy_input = torch.randn(1, 128, 9)  # [batch, sequence, features]
torch.onnx.export(
    model,
    dummy_input,
    'har_model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Step 3: Convert ONNX to TensorFlow
from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load('har_model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('har_model_tf')

# Step 4: Convert TensorFlow to TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('har_model_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('har_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Option 2: Using PyTorch Mobile Export

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Load model
model = torch.load('har_model.ptl')
model.eval()

# Trace the model
example_input = torch.randn(1, 128, 9)
traced_script_module = torch.jit.trace(model, example_input)

# Optimize for mobile
optimized_model = optimize_for_mobile(traced_script_module)
optimized_model._save_for_lite_interpreter("har_model_mobile.ptl")

# Then convert to ONNX and follow Option 1
```

### Option 3: Train New Model in TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras

# Define similar architecture
model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(128, 9)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')  # 6 activities
])

# Train on UCI HAR dataset
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ... train the model ...

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('har_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## After Conversion

1. **Place model file:**
   ```
   flutter_har_app/assets/models/har_model.tflite
   ```

2. **Rebuild app:**
   ```bash
   flutter clean
   flutter pub get
   flutter run -d <device-id>
   ```

3. **Model will auto-load** and app will switch from DEMO MODE to REAL MODE

## Verification

Check logs for:
- ✅ "HAR Model loaded successfully" → Real ML predictions
- ⚠️ "Running in DEMO MODE" → Simulated predictions

## Model Requirements

- **Input:** `[1, 128, 9]` - Batch of 128 timesteps with 9 sensor features
  - Features: `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, linear_x, linear_y, linear_z]`
- **Output:** `[1, 6]` - Probabilities for 6 activities
  - Activities: `[WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING]`

## Benefits of TFLite

✅ Better Flutter support (`tflite_flutter` package actively maintained)
✅ Smaller model size
✅ Faster inference on mobile devices
✅ Hardware acceleration support (GPU, NNAPI)
✅ Compatible with latest Android Gradle Plugin

## Testing Without Model

The app is ready to install RIGHT NOW without any ML model:
1. It will use sensor data to make intelligent guesses
2. Movement patterns → Activity prediction
3. Realistic confidence scores
4. Full UI functionality

You can test and use the app immediately while working on model conversion!

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [ONNX Documentation](https://onnx.ai/)
- [PyTorch to ONNX](https://pytorch.org/docs/stable/onnx.html)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
