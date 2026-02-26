@echo off
echo ========================================
echo   Flutter HAR App - Device Installer
echo ========================================
echo.

cd /d "d:\HAR app\flutter_har_app"

echo Checking for connected devices...
echo.
flutter devices
echo.

echo ========================================
echo.
echo Options:
echo   1. Install on connected device (run app)
echo   2. Build APK only
echo   3. Check device connection
echo   4. Clean and rebuild
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting app installation...
    flutter run
) else if "%choice%"=="2" (
    echo.
    echo Building APK...
    flutter build apk --release
    echo.
    echo APK built successfully!
    echo Location: build\app\outputs\flutter-apk\app-release.apk
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    flutter devices
    echo.
    pause
) else if "%choice%"=="4" (
    echo.
    echo Cleaning project...
    flutter clean
    echo Installing dependencies...
    flutter pub get
    echo.
    echo Done! Now try installing again.
    echo.
    pause
) else (
    echo Exiting...
    exit
)
