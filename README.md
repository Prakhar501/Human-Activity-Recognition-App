# 📱 Human Activity Recognition App

**Real-time activity detection using phone sensors + Privacy-preserving Federated Learning**

---

## 🎯 What is This?

Mobile app jo phone ke sensors (Accelerometer + Gyroscope) use karke detect karta hai ki aap kya kar rahe ho:

- 🚶 **WALKING** - Chal rahe ho
- 🪑 **SITTING** - Baithe ho
- 🧍 **STANDING** - Khade ho
- 🛏️ **LAYING** - Lete ho

**Accuracy:** 95%+ real devices pe  
**Model Size:** 255 KB (bahut chhota!)  
**Speed:** <100ms (real-time)

---

## ⚡ Quick Start

### **1. Install App on Phone**

```bash
cd flutter_har_app
flutter build apk --release
flutter install
```

**APK Location:** `flutter_har_app/build/app/outputs/flutter-apk/app-release.apk`

### **2. Start Federated Learning Server** (Optional)

```bash
cd federated
python flask_server.py
```

Server starts at: `http://YOUR_IP:5000`

### **3. Use App**

- Open app → Press **Start** button
- App will show your current activity
- Try walking, sitting, standing, laying down

---

## 📊 Features

```
✅ Real-time Activity Detection
✅ 95%+ Accuracy
✅ Lightweight Model (255 KB)
✅ Privacy-Preserving Federated Learning
✅ Cross-Platform (Flutter)
✅ Offline Predictions
✅ Material Design 3 UI
```

---

## 📁 Project Structure

```
Human-Activity-Recognition-App/
│
├── 📱 flutter_har_app/        # Mobile app (Flutter)
├── 🤖 trained_model/          # TFLite models
├── 📊 dataset/                # Training data
├── 🔄 federated/              # FL server + clients
├── � results_and_metrics/    # All experimental results & metrics
├── 🐍 *.py                    # Training scripts
│
├── 📄 README.md               # This file (quick start)
└── 📘 PROJECT_GUIDE.md        # Complete documentation
```

---

## 📊 Results & Metrics

**All experimental results, accuracy metrics, and performance data:**

📁 **[results_and_metrics/](results_and_metrics/)** - Complete results folder
- 📊 **RESEARCH_TABLES.md** - All research tables with complete data
- 📈 **Federated Results** - Round-by-round accuracy, model comparisons
- 🎯 **Model Performance** - Classification reports, confusion matrices
- 📉 **Visualizations** - Performance graphs and comparison charts

**Quick Stats:**
- Federated Learning: **89.75%** accuracy
- Centralized Model: **91.81%** accuracy
- UCI-HAR Only: **93.28%** accuracy
- Personal Data Only: **93.94%** accuracy

---

## 📚 Documentation

🔥 **Complete Guide:** [PROJECT_GUIDE.md](PROJECT_GUIDE.md)

Isme milega:
- ✅ Step-by-step installation
- ✅ Architecture details
- ✅ Training process
- ✅ Federated Learning explanation
- ✅ Troubleshooting
- ✅ Testing results

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Mobile** | Flutter 3.x, Dart |
| **ML Model** | TensorFlow, Keras, TFLite |
| **FL Server** | Python Flask, FedAvg |
| **Sensors** | Accelerometer + Gyroscope |
| **Dataset** | UCI HAR (10K) + Personal (33K) |

---

## 🚀 Current Status

```
✅ Model: Trained & Optimized (83% validation, 95% real-world)
✅ App: Deployed & Working
✅ FL: Implemented & Tested
✅ Predictions: Real-time (<100ms)
✅ Privacy: Data stays on device

STATUS: PRODUCTION READY 🎉
```

---

## 🔥 Quick Commands

```bash
# Install app
flutter build apk --release && flutter install

# Start FL server
python federated/flask_server.py

# Train model
python train_combined_model.py

# View logs
adb logcat -s flutter:I
```

---

## 📞 Need Help?

- **Complete guide:** [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- **Installation issues:** Check troubleshooting section in guide
- **FL not working:** Check firewall settings (port 5000)

---

## 🏆 Key Achievements

- ✅ **43,067 samples** trained (mixed real-world + lab data)
- ✅ **95%+ accuracy** on actual devices
- ✅ **255KB model** (mobile-optimized)
- ✅ **Federated Learning** working with FedAvg
- ✅ **Privacy-preserving** - data never leaves device

---

**Version:** 1.0.0  
**Date:** March 3, 2026  
**Status:** ✅ Production Ready

*For detailed documentation, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md)*
