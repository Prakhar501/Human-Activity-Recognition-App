# Flutter HAR App - Device Installation Guide

## Current Status

✅ Original Android/Gradle app deleted
✅ Flutter app ready with all dependencies installed
❌ No Android device currently connected

## Available Devices:
- Windows Desktop
- Chrome Browser
- Edge Browser
- Android Emulator (not starting - environment issues)

## Options to Install App:

### Option 1: Connect Physical Android Device (RECOMMENDED)

1. **Enable Developer Options on your Android phone:**
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times
   - Go back to Settings > Developer Options
   - Enable "USB Debugging"

2. **Connect via USB:**
   - Connect phone to PC with USB cable
   - Allow USB debugging when prompted on phone
   - Select "File Transfer" or "PTP" mode

3. **Verify connection:**
   ```bash
   cd "d:\HAR app\flutter_har_app"
   flutter devices
   ```

4. **Install app:**
   ```bash
   flutter run
   # or
   flutter install
   ```

### Option 2: Fix Android Emulator

1. **Check Android SDK location:**
   - Open Android Studio
   - Go to File > Settings > Appearance & Behavior > System Settings > Android SDK
   - Note the SDK location

2. **Verify emulator installation:**
   - In Android Studio, go to AVD Manager
   - Start "Medium Phone API 36.1" manually

3. **Then run:**
   ```bash
   cd "d:\HAR app\flutter_har_app"
   flutter devices
   flutter run
   ```

### Option 3: Install via ADB (if device is connected but not detected)

```bash
# Check ADB devices
adb devices

# Install APK directly (build first)
cd "d:\HAR app\flutter_har_app"
flutter build apk --release
adb install build\app\outputs\flutter-apk\app-release.apk
```

### Option 4: Run on Windows Desktop (for testing UI only)

⚠️ **Note:** Sensors won't work on Windows, but you can test the UI

```bash
cd "d:\HAR app\flutter_har_app"
flutter run -d windows
```

## Quick Commands:

```powershell
# Navigate to project
cd "d:\HAR app\flutter_har_app"

# Check connected devices
flutter devices

# Run on specific device
flutter run -d <device-id>

# Build APK and install manually
flutter build apk --release
# Then transfer APK to phone and install

# Clean if needed
flutter clean
flutter pub get
```

## Troubleshooting:

### Device not detected?
- Try different USB port
- Try different USB cable
- Restart ADB: `adb kill-server` then `adb start-server`
- Reinstall USB drivers

### Emulator not starting?
- Open Android Studio
- Go to AVD Manager
- Delete and recreate emulator
- Or download a new system image

### Build errors?
```bash
flutter clean
flutter pub get
flutter doctor -v
```

## Ready to Install?

Once you connect your Android device:

```bash
cd "d:\HAR app\flutter_har_app"
flutter run
```

The app will automatically install and launch on your device!
