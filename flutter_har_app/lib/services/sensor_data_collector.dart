import 'dart:async';
import 'package:sensors_plus/sensors_plus.dart';

typedef SensorDataCallback = void Function(List<List<double>> sensorData);

class SensorDataCollector {
  static const String tag = "SensorCollector";

  // Data buffers (only accel + gyro = 6 features)
  final List<List<double>> _accelerometerData = [];
  final List<List<double>> _gyroscopeData = [];

  final int windowSize = 128; // 128 samples per window
  bool _isCollecting = false;

  // Rate limiting
  DateTime _lastProcessTime = DateTime.now();
  static const Duration minProcessInterval = Duration(seconds: 2);

  // Stream subscriptions
  StreamSubscription<AccelerometerEvent>? _accelSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroSubscription;

  // Callback
  SensorDataCallback? onDataCollected;

  SensorDataCollector({this.onDataCollected});

  /// Initialize sensors (optional in Flutter - sensors_plus handles this)
  void initializeSensors() {
    print('$tag: Sensors initialized successfully');
  }

  /// Start collecting sensor data
  void startCollecting() {
    if (_isCollecting) {
      print('$tag: Already collecting');
      return;
    }

    // Clear old data
    _accelerometerData.clear();
    _gyroscopeData.clear();

    _lastProcessTime = DateTime.now();
    _isCollecting = true;

    // Subscribe to accelerometer
    _accelSubscription = accelerometerEventStream().listen((AccelerometerEvent event) {
      if (_isCollecting) {
        _accelerometerData.add([event.x, event.y, event.z]);
        _checkAndProcessData();
      }
    });

    // Subscribe to gyroscope
    _gyroSubscription = gyroscopeEventStream().listen((GyroscopeEvent event) {
      if (_isCollecting) {
        _gyroscopeData.add([event.x, event.y, event.z]);
      }
    });

    print('$tag: Started collecting sensor data');
  }

  /// Stop collecting sensor data
  void stopCollecting() {
    if (!_isCollecting) {
      return;
    }

    _accelSubscription?.cancel();
    _gyroSubscription?.cancel();

    _accelSubscription = null;
    _gyroSubscription = null;

    _isCollecting = false;
    print('$tag: Stopped collecting sensor data');
  }

  /// Check if we have enough data and process it
  void _checkAndProcessData() {
    // Check if BOTH sensors have enough data AND enough time has passed
    if (_accelerometerData.length >= windowSize &&
        _gyroscopeData.length >= windowSize) {
      
      final currentTime = DateTime.now();
      if (currentTime.difference(_lastProcessTime) >= minProcessInterval) {
        _processDataWindow();
        _lastProcessTime = currentTime;

        // Use sliding window: keep 50% overlap (64 samples) for smoother transitions
        // This helps stabilize static activity detection
        int keepSamples = windowSize ~/ 2; // Keep 64 samples
        if (_accelerometerData.length > keepSamples) {
          _accelerometerData.removeRange(0, _accelerometerData.length - keepSamples);
        }
        if (_gyroscopeData.length > keepSamples) {
          _gyroscopeData.removeRange(0, _gyroscopeData.length - keepSamples);
        }
      }
    }
    
    // Remove excess data if buffer grows too large (prevents memory issues)
    if (_accelerometerData.length > windowSize * 2) {
      _accelerometerData.removeRange(0, _accelerometerData.length - windowSize);
    }
    if (_gyroscopeData.length > windowSize * 2) {
      _gyroscopeData.removeRange(0, _gyroscopeData.length - windowSize);
    }
  }

  /// Process the data window and create input for model
  void _processDataWindow() {
    try {
      // Trim to exactly windowSize samples from each sensor
      List<List<double>> accelTrimmed = _accelerometerData.sublist(0, windowSize);
      List<List<double>> gyroTrimmed = _gyroscopeData.sublist(0, windowSize);
      
      // Combine sensor data into a single array (6 features total)
      List<List<double>> combinedData = [];

      for (int i = 0; i < windowSize; i++) {
        List<double> row = [];
        
        // Add accelerometer data (3 values) - guaranteed to exist
        row.addAll(accelTrimmed[i]);
        
        // Add gyroscope data (3 values) - guaranteed to exist
        row.addAll(gyroTrimmed[i]);

        combinedData.add(row);
      }

      // Validate combined data dimensions
      if (combinedData.length != windowSize || combinedData[0].length != 6) {
        print('$tag: ERROR - Invalid data shape: [${combinedData.length}, ${combinedData[0].length}]');
        return;
      }
      
      // Log data stats
      print('$tag: Processed window [128x6] - Accel: ${accelTrimmed.length}, '
            'Gyro: ${gyroTrimmed.length}');

      // Callback with combined data
      if (onDataCollected != null) {
        onDataCollected!(combinedData);
      }
    } catch (e) {
      print('$tag: Error processing data window: $e');
    }
  }

  /// Get current status
  bool get isCollecting => _isCollecting;

  /// Dispose and cleanup
  void dispose() {
    stopCollecting();
  }
}
