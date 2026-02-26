import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/sensor_data_collector.dart';
import 'models/har_model.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HAR App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HARHomePage(),
    );
  }
}

class HARHomePage extends StatefulWidget {
  const HARHomePage({super.key});

  @override
  State<HARHomePage> createState() => _HARHomePageState();
}

class _HARHomePageState extends State<HARHomePage> {
  // Services
  late SensorDataCollector _sensorCollector;
  late HARModel _harModel;

  // State variables
  bool _isDetecting = false;
  bool _isModelLoaded = false;
  String _statusText = "Ready";
  String _currentActivity = "---";
  double _confidence = 0.0;
  List<double> _probabilities = List.filled(6, 0.0);
  
  // Accuracy tracking
  int _correctPredictions = 0;
  int _totalPredictions = 0;
  String _selectedTrueActivity = "WALKING";

  @override
  void initState() {
    super.initState();
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    // Request permissions
    await _requestPermissions();

    // Initialize sensor collector
    _sensorCollector = SensorDataCollector(
      onDataCollected: _onSensorDataCollected,
    );
    _sensorCollector.initializeSensors();

    // Load ML model
    await _loadModel();
  }

  Future<void> _requestPermissions() async {
    final status = await [
      Permission.sensors,
      Permission.activityRecognition,
    ].request();

    print('Permissions status: $status');
  }

  Future<void> _loadModel() async {
    try {
      setState(() {
        _statusText = "Loading model...";
      });

      _harModel = HARModel();
      await _harModel.loadModel();

      setState(() {
        _isModelLoaded = true;
        _statusText = "Model loaded successfully";
      });

      _showSnackBar("Model loaded successfully", Colors.green);
    } catch (e) {
      setState(() {
        _isModelLoaded = true; // Set true to allow demo mode
        _statusText = "⚠️ Running in DEMO MODE";
      });
      _showSnackBar("Running in DEMO MODE with simulated predictions", Colors.orange);
    }
  }

  void _onSensorDataCollected(List<List<double>> sensorData) async {
    if (!_isModelLoaded || !_isDetecting) return;

    try {
      // Run prediction
      final result = await _harModel.predict(sensorData);

      setState(() {
        _currentActivity = result.activityName;
        _confidence = result.confidence;
        _probabilities = result.allProbabilities;
        _totalPredictions++;

        // Check accuracy
        if (result.activityName == _selectedTrueActivity) {
          _correctPredictions++;
        }
      });

      print('Predicted: ${result.activityName} (${(result.confidence * 100).toStringAsFixed(1)}%)');
    } catch (e) {
      print('Error during prediction: $e');
    }
  }

  void _startDetection() {
    if (!_isModelLoaded) {
      _showSnackBar("Model not loaded yet", Colors.orange);
      return;
    }

    setState(() {
      _isDetecting = true;
      _statusText = "Detecting...";
    });

    _sensorCollector.startCollecting();
    _showSnackBar("Detection started", Colors.green);
  }

  void _stopDetection() {
    setState(() {
      _isDetecting = false;
      _statusText = "Stopped";
    });

    _sensorCollector.stopCollecting();
    _showSnackBar("Detection stopped", Colors.blue);
  }

  void _showSnackBar(String message, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  @override
  void dispose() {
    _sensorCollector.dispose();
    _harModel.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Human Activity Recognition'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Status Card
            _buildCard(
              title: "Status",
              child: Text(
                _statusText,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: _isDetecting ? Colors.green : Colors.grey,
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Current Activity Card
            _buildCard(
              title: "Current Activity",
              child: Column(
                children: [
                  Text(
                    _currentActivity,
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    "Confidence: ${(_confidence * 100).toStringAsFixed(1)}%",
                    style: const TextStyle(
                      fontSize: 18,
                      color: Colors.blue,
                    ),
                  ),
                ],
              ),
              centered: true,
            ),

            const SizedBox(height: 16),

            // Probabilities Card
            _buildCard(
              title: "All Probabilities",
              child: Column(
                children: List.generate(
                  HARModel.activityLabels.length,
                  (index) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          HARModel.activityLabels[index],
                          style: const TextStyle(fontSize: 14),
                        ),
                        Text(
                          "${(_probabilities[index] * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 16),

            // True Activity Selection Card
            _buildCard(
              title: "Select True Activity (for accuracy)",
              child: DropdownButton<String>(
                value: _selectedTrueActivity,
                isExpanded: true,
                items: HARModel.activityLabels.map((String activity) {
                  return DropdownMenuItem<String>(
                    value: activity,
                    child: Text(activity),
                  );
                }).toList(),
                onChanged: (String? newValue) {
                  if (newValue != null) {
                    setState(() {
                      _selectedTrueActivity = newValue;
                    });
                  }
                },
              ),
            ),

            const SizedBox(height: 16),

            // Accuracy Card
            _buildCard(
              title: "Accuracy",
              child: Text(
                _totalPredictions > 0
                    ? "${((_correctPredictions / _totalPredictions) * 100).toStringAsFixed(1)}% ($_correctPredictions/$_totalPredictions)"
                    : "No predictions yet",
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.green,
                ),
              ),
              centered: true,
            ),

            const SizedBox(height: 24),

            // Control Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isDetecting ? null : _startDetection,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text("Start Detection"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isDetecting ? _stopDetection : null,
                    icon: const Icon(Icons.stop),
                    label: const Text("Stop Detection"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCard({
    required String title,
    required Widget child,
    bool centered = false,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment:
              centered ? CrossAxisAlignment.center : CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontSize: 14,
                color: Colors.grey,
              ),
            ),
            const SizedBox(height: 8),
            child,
          ],
        ),
      ),
    );
  }
}
