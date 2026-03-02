# Flutter HAR App

## Human Activity Recognition App - Flutter Version

Ye app **Flutter** me banaya gaya hai jo real-time human activity recognition karta hai using phone sensors aur PyTorch machine learning model.

### 🎯 Features

- **6 Activities Detect Karta Hai:**
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING

- **Real-time Detection:** Phone ke sensors se live data collect karke activity predict karta hai
- **Accuracy Tracking:** User apni actual activity select kar sakta hai aur app ki accuracy check kar sakta hai
- **Beautiful UI:** Material Design 3 with Card-based layout

### 📱 Technology Stack

- **Framework:** Flutter 3.x
- **Language:** Dart
- **ML Framework:** PyTorch Mobile
- **Sensors:** sensors_plus package
- **Permissions:** permission_handler

### 🔧 Dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  sensors_plus: ^4.0.2
  permission_handler: ^11.3.0
  pytorch_mobile: ^0.2.2
  provider: ^6.1.1
```

### 📁 Project Structure

```
flutter_har_app/
├── lib/
│   ├── main.dart                    # Main UI
│   ├── models/
│   │   └── har_model.dart          # PyTorch model handler
│   └── services/
│       └── sensor_data_collector.dart  # Sensor data collection
├── assets/
│   └── models/
│       └── mobilenet_gru_uci_har_mobile.ptl  # ML model
├── android/
│   └── app/
│       ├── build.gradle            # Android build config
│       └── src/main/
│           └── AndroidManifest.xml # Permissions
└── pubspec.yaml                    # Flutter config
```

### 🚀 Installation & Setup

1. **Install Flutter:**
   ```bash
   # Download Flutter SDK from https://flutter.dev
   ```

2. **Clone/Navigate to project:**
   ```bash
   cd "d:\HAR app\flutter_har_app"
   ```

3. **Install dependencies:**
   ```bash
   flutter pub get
   ```

4. **Connect Android device or start emulator**

5. **Run app:**
   ```bash
   flutter run
   ```

### 🔐 Required Permissions

- Body Sensors
- Activity Recognition
- High Sampling Rate Sensors (Android 12+)

### 📊 How It Works

1. **Sensors** se data collect hota hai (Accelerometer, Gyroscope, Linear Acceleration)
2. **128 samples** ka window banaya jata hai
3. **PyTorch model** ko input diya jata hai
4. Model **6 activities ke probabilities** calculate karta hai
5. **Highest probability** wali activity display hoti hai

### 🔄 Differences from Java Version

| Feature | Java (Original) | Flutter (This) |
|---------|----------------|----------------|
| Language | Java | Dart |
| UI Framework | XML Layouts | Flutter Widgets |
| Sensors | Android SensorManager | sensors_plus package |
| ML Integration | PyTorch Android | pytorch_mobile package |
| State Management | Activity callbacks | StatefulWidget |
| Async Operations | Handler/Looper | Future/async-await |

### ⚠️ Important Notes

1. **pytorch_mobile** package ka implementation actual Java version se thoda different ho sakta hai
2. Model loading aur inference code ko test karna zaroori hai
3. iOS support ke liye additional configuration chahiye hoga
4. Sensor sampling rate Flutter me automatically managed hoti hai

### 🐛 Troubleshooting

**Model not loading:**
- Check if `.ptl` file is in `assets/models/` folder
- Verify `pubspec.yaml` me assets path correctly defined hai

**Sensors not working:**
- Permissions granted hain ya nahi check karein
- Physical device pe test karein (emulator me sensors limited hote hain)

**Build errors:**
- `flutter clean` run karein
- `flutter pub get` dubara run karein

### 📝 Next Steps (Optional Improvements)

- [ ] Add data visualization (charts/graphs)
- [ ] Add activity history tracking
- [ ] Export data to CSV
- [ ] Add background service for continuous monitoring
- [ ] Add iOS support
- [ ] Improve model loading with better error handling
- [ ] Add settings page for customization

### 👨‍💻 Original Java App

Original Java/Android version: `d:\HAR app\Human-Activity-Recognition-App`

### 📄 License

Same as original project.

---

**Made with ❤️ in Flutter**
