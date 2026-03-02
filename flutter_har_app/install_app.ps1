# Flutter HAR App - Quick Installer Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Flutter HAR App - Device Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "d:\HAR app\flutter_har_app"

Write-Host "Checking for connected devices..." -ForegroundColor Yellow
Write-Host ""
flutter devices
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Options:" -ForegroundColor Green
Write-Host "  1. Install on connected device (run app)"
Write-Host "  2. Build APK only"
Write-Host "  3. Check device connection"
Write-Host "  4. Clean and rebuild"
Write-Host "  5. Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Starting app installation..." -ForegroundColor Green
        flutter run
    }
    "2" {
        Write-Host ""
        Write-Host "Building APK..." -ForegroundColor Yellow
        flutter build apk --release
        Write-Host ""
        Write-Host "APK built successfully!" -ForegroundColor Green
        Write-Host "Location: build\app\outputs\flutter-apk\app-release.apk" -ForegroundColor Cyan
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "3" {
        Write-Host ""
        flutter devices
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "4" {
        Write-Host ""
        Write-Host "Cleaning project..." -ForegroundColor Yellow
        flutter clean
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        flutter pub get
        Write-Host ""
        Write-Host "Done! Now try installing again." -ForegroundColor Green
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    default {
        Write-Host "Exiting..." -ForegroundColor Red
        exit
    }
}
