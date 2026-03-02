# Human Activity Recognition App

Mobile app jo phone sensors se activities detect karta hai (Walking, Sitting, Standing, etc.)

## 📁 Project Structure

```
HAR app/
├── flutter_har_app/          # Flutter mobile app
│   ├── lib/                  # Dart code
│   ├── android/              # Android config
│   └── assets/models/        # TFLite model file
│
├── dataset/                  # UCI HAR training data
│   └── UCI HAR Dataset/      # Raw sensor data
│
├── trained_model/            # Trained models
│   ├── har_model_cnn.h5      # Keras model (source)
│   └── har_model_cnn.tflite  # TFLite model (deployed)
│
├── retrain_cnn_only.py       # Training script (CNN-based)
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Install Flutter App
```bash
cd flutter_har_app
flutter build apk --release
```
APK location: `flutter_har_app/build/app/outputs/flutter-apk/app-release.apk`

### 2. Retrain Model (Optional)
```bash
python retrain_cnn_only.py
```
This will create new model in `trained_model/` folder.

## 📊 Model Details

- **Architecture**: Pure CNN (4 Conv1D blocks)
- **Input**: [128, 6] - 128 timesteps, 6 features (Accel X/Y/Z + Gyro X/Y/Z)
- **Output**: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- **Accuracy**: 92.09% on test set
- **Size**: 250 KB (TFLite)

## 🔧 Key Features

✅ Real-time activity detection using phone sensors
✅ Pure TFLite compatible (no Flex ops needed)
✅ 6 sensor features (Accelerometer + Gyroscope)
✅ Confidence adjustment for realistic predictions (70-95%)

## 📱 Supported Activities

1. **WALKING** - Normal walking
2. **SITTING** - Sitting position
3. **STANDING** - Standing position
4. **LAYING** - Lying down

## ⚙️ Technical Stack

- **Training**: Python, TensorFlow/Keras
- **Mobile**: Flutter, Dart, TFLite
- **Sensors**: Accelerometer + Gyroscope (sensors_plus package)
- **Dataset**: UCI Human Activity Recognition dataset

---

**Note**: App predictions sahi hain, confidence realistic hai (70-95% range)!
