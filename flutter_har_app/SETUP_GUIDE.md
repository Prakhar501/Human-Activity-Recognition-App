# Flutter HAR App - Installation & Usage Guide

## Prerequisites

### 1. Install Flutter
```bash
# Download Flutter SDK from: https://flutter.dev/docs/get-started/install
# Add Flutter to your PATH
```

### 2. Verify Flutter Installation
```bash
flutter doctor
```

## Setup Steps

### 1. Navigate to Project
```bash
cd "d:\HAR app\flutter_har_app"
```

### 2. Install Dependencies
```bash
flutter pub get
```

### 3. Check Connected Devices
```bash
flutter devices
```

### 4. Run the App
```bash
# On connected device
flutter run

# On specific device
flutter run -d <device-id>

# Release mode
flutter run --release
```

## Building APK

### Debug APK
```bash
flutter build apk --debug
```

### Release APK
```bash
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

## Common Issues & Solutions

### 1. Model Not Loading
**Problem:** PyTorch model file not found

**Solution:**
- Verify model file exists: `assets/models/mobilenet_gru_uci_har_mobile.ptl`
- Check `pubspec.yaml` has correct assets path
- Run `flutter clean` and `flutter pub get`

### 2. Sensor Permissions Not Working
**Problem:** Sensors not accessible

**Solution:**
- Grant permissions from device Settings > Apps > HAR App > Permissions
- Use physical device (emulator has limited sensor support)

### 3. Build Errors
**Problem:** Gradle build fails

**Solution:**
```bash
flutter clean
cd android
./gradlew clean
cd ..
flutter pub get
flutter build apk
```

### 4. Package Not Found
**Problem:** `pytorch_mobile` package error

**Solution:**
```bash
flutter pub cache repair
flutter pub get
```

## Testing

### Run on Android Device
1. Enable USB Debugging on your phone
2. Connect via USB
3. Run: `flutter run`

### Run on Android Emulator
1. Open Android Studio
2. Start AVD (Android Virtual Device)
3. Run: `flutter run`

## Project Structure

```
flutter_har_app/
├── lib/
│   ├── main.dart                   # Main entry point & UI
│   ├── models/
│   │   └── har_model.dart         # ML model wrapper
│   └── services/
│       └── sensor_data_collector.dart  # Sensor handling
├── assets/
│   └── models/
│       └── *.ptl                  # PyTorch model
├── android/                        # Android-specific code
└── pubspec.yaml                   # Project configuration
```

## Usage Instructions

1. **Launch App:** Open app on your device

2. **Wait for Model Load:** App will automatically load the ML model

3. **Select True Activity:** Choose your current activity from dropdown (for accuracy tracking)

4. **Start Detection:** Press "Start Detection" button

5. **Perform Activity:** Move around while app detects your activity

6. **View Results:**
   - Current detected activity
   - Confidence percentage
   - All probabilities
   - Real-time accuracy

7. **Stop Detection:** Press "Stop Detection" when done

## Performance Tips

- Use physical device for better sensor accuracy
- Keep device stable for sitting/standing detection
- Move naturally for walking activities
- Ensure good battery level (sensors consume power)

## Troubleshooting Commands

```bash
# Clean build
flutter clean

# Update dependencies
flutter pub upgrade

# Check for issues
flutter doctor -v

# View device logs
flutter logs

# Install on device
flutter install
```

## Requirements

- **Flutter:** 3.0.0 or higher
- **Dart:** 3.0.0 or higher
- **Android:** API 26 (Android 8.0) or higher
- **Device Storage:** ~50MB for app + model

## Permissions Required

- Body Sensors
- Activity Recognition
- High Sampling Rate Sensors (Android 12+)

## Next Steps

- Test app thoroughly on different devices
- Adjust sensor sampling rates if needed
- Customize UI colors/themes
- Add more features (data export, history, etc.)

## Support

For issues or questions, refer to:
- Flutter documentation: https://flutter.dev/docs
- PyTorch Mobile: https://pytorch.org/mobile
- Sensors Plus: https://pub.dev/packages/sensors_plus
