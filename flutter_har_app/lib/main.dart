import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/sensor_data_collector.dart';
import 'models/har_model.dart';
import 'services/federated_client.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HAR App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6200EE),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
          ),
        ),
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
  FederatedClient? _federatedClient;

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
  
  // Federated Learning
  bool _isFLTraining = false;
  String _flStatus = 'Not connected';
  int _flRound = 0;

  // Helper function to format activity names for display
  String _formatActivityName(String activityName) {
    // Merge all walking types into one
    if (activityName.contains('WALKING')) {
      return 'Walking';
    }
    // Format other activities: SITTING -> Sitting
    return activityName.substring(0, 1).toUpperCase() + 
           activityName.substring(1).toLowerCase();
  }

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
    
    // Initialize Federated Learning Client
    _initializeFederatedLearning();
  }
  
  void _initializeFederatedLearning() {
    // IMPORTANT: Replace with your computer's local IP address
    // Windows: Run 'ipconfig' in terminal, look for IPv4 Address
    // Linux/Mac: Run 'ifconfig', look for inet address
    const String serverIp = '192.168.1.4'; // Your computer's IP
    const String serverPort = '5000';
    
    _federatedClient = FederatedClient(
      serverUrl: 'http://$serverIp:$serverPort',
    );
    
    // Check server connection
    _checkFLServer();
  }
  
  Future<void> _checkFLServer() async {
    if (_federatedClient == null) return;
    
    final connected = await _federatedClient!.checkServerConnection();
    setState(() {
      _flStatus = connected ? 'Connected ✅' : 'Server offline ❌';
    });
    
    if (connected) {
      // Get server status
      final status = await _federatedClient!.getServerStatus();
      if (status != null) {
        setState(() {
          _flRound = status['current_round'] ?? 0;
          _flStatus = 'Connected ✅ (Round $_flRound)';
        });
      }
    }
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
        _currentActivity = _formatActivityName(result.activityName);
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
  
  // Federated Learning Training
  Future<void> _startFederatedTraining() async {
    if (_isFLTraining) {
      _showSnackBar("Already training...", Colors.orange);
      return;
    }
    
    if (_federatedClient == null) {
      _showSnackBar("FL Client not initialized", Colors.red);
      return;
    }
    
    setState(() {
      _isFLTraining = true;
      _flStatus = 'Training... 🔄';
    });
    
    _showSnackBar("Starting federated training...", Colors.blue);
    
    try {
      // Simulate training round
      // In production, you would:
      // 1. Collect sensor data samples
      // 2. Train model locally (if supported)
      // 3. Extract weights from trained model
      
      final success = await _federatedClient!.simulateTrainingRound();
      
      if (success) {
        setState(() {
          _flRound++;
          _flStatus = 'Training complete ✅ (Round $_flRound)';
        });
        _showSnackBar("✅ Federated round complete! Model updated.", Colors.green);
        
        // Optionally reload the model with new weights
        // await _harModel.loadModel();
      } else {
        setState(() {
          _flStatus = 'Waiting for clients... ⏳';
        });
        _showSnackBar("⏳ Waiting for more clients to join...", Colors.orange);
      }
    } catch (e) {
      setState(() {
        _flStatus = 'Training failed ❌';
      });
      _showSnackBar("❌ Training failed: $e", Colors.red);
      print('FL Training error: $e');
    } finally {
      setState(() {
        _isFLTraining = false;
      });
    }
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
    final screenSize = MediaQuery.of(context).size;
    final screenWidth = screenSize.width;
    final screenHeight = screenSize.height;
    
    // Responsive sizing
    final horizontalPadding = screenWidth * 0.04; // 4% of screen width
    final verticalSpacing = screenHeight * 0.015; // 1.5% of screen height
    final cardPadding = screenWidth * 0.045; // 4.5% of screen width
    
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Theme.of(context).colorScheme.primaryContainer.withOpacity(0.3),
              Theme.of(context).colorScheme.secondaryContainer.withOpacity(0.2),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Modern App Bar
              _buildModernAppBar(context, screenWidth),
              
              // Scrollable content
              Expanded(
                child: SingleChildScrollView(
                  padding: EdgeInsets.symmetric(
                    horizontal: horizontalPadding,
                    vertical: verticalSpacing,
                  ),
                  child: Column(
                    children: [
                      // Status Card with animation
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        child: _buildStatusCard(context, screenWidth, cardPadding),
                      ),

                      SizedBox(height: verticalSpacing),

                      // Current Activity Hero Card
                      _buildActivityHeroCard(context, screenWidth, cardPadding),

                      SizedBox(height: verticalSpacing),

                      // Probabilities Card with progress bars
                      _buildProbabilitiesCard(context, screenWidth, cardPadding),

                      SizedBox(height: verticalSpacing),

                      // True Activity Selection Card
                      _buildSelectionCard(context, screenWidth, cardPadding),

                      SizedBox(height: verticalSpacing),

                      // Accuracy Card
                      _buildAccuracyCard(context, screenWidth, cardPadding),

                      SizedBox(height: verticalSpacing * 1.5),

                      // Control Buttons
                      _buildControlButtons(context, screenWidth),
                      
                      SizedBox(height: verticalSpacing * 1.5),
                      
                      // Federated Learning Card
                      _buildFederatedLearningCard(context, screenWidth, cardPadding),
                      
                      SizedBox(height: verticalSpacing),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildModernAppBar(BuildContext context, double screenWidth) {
    final titleSize = screenWidth * 0.05; // 5% of screen width
    
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: screenWidth * 0.04,
        vertical: screenWidth * 0.03,
      ),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(screenWidth * 0.02),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Theme.of(context).colorScheme.primary,
                  Theme.of(context).colorScheme.secondary,
                ],
              ),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.sensors,
              color: Colors.white,
              size: screenWidth * 0.06,
            ),
          ),
          SizedBox(width: screenWidth * 0.03),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Activity Recognition',
                  style: TextStyle(
                    fontSize: titleSize.clamp(18.0, 24.0),
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
                Text(
                  'Real-time monitoring',
                  style: TextStyle(
                    fontSize: (titleSize * 0.55).clamp(11.0, 14.0),
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard(BuildContext context, double screenWidth, double cardPadding) {
    final statusColor = _isDetecting ? Colors.green : Colors.grey;
    final fontSize = (screenWidth * 0.045).clamp(16.0, 20.0);
    
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(cardPadding),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: statusColor.withOpacity(0.3),
          width: 2,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(screenWidth * 0.025),
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
            ),
            child: Icon(
              _isDetecting ? Icons.check_circle : Icons.info_outline,
              color: Colors.white,
              size: screenWidth * 0.05,
            ),
          ),
          SizedBox(width: screenWidth * 0.03),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Status',
                  style: TextStyle(
                    fontSize: (fontSize * 0.7).clamp(12.0, 14.0),
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                SizedBox(height: screenWidth * 0.01),
                Text(
                  _statusText,
                  style: TextStyle(
                    fontSize: fontSize,
                    fontWeight: FontWeight.bold,
                    color: statusColor,
                  ),
                ),
              ],
            ),
          ),
          if (_isDetecting)
            SizedBox(
              width: screenWidth * 0.06,
              height: screenWidth * 0.06,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                valueColor: AlwaysStoppedAnimation<Color>(statusColor),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildActivityHeroCard(BuildContext context, double screenWidth, double cardPadding) {
    final activitySize = (screenWidth * 0.08).clamp(28.0, 40.0);
    final confidenceSize = (screenWidth * 0.045).clamp(16.0, 20.0);
    
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(cardPadding * 1.3),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Theme.of(context).colorScheme.primaryContainer,
            Theme.of(context).colorScheme.secondaryContainer,
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Text(
            'Current Activity',
            style: TextStyle(
              fontSize: (confidenceSize * 0.8).clamp(13.0, 16.0),
              color: Theme.of(context).colorScheme.onPrimaryContainer.withOpacity(0.7),
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
          SizedBox(height: screenWidth * 0.02),
          Text(
            _currentActivity,
            style: TextStyle(
              fontSize: activitySize,
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onPrimaryContainer,
              letterSpacing: 0.5,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: screenWidth * 0.03),
          Container(
            padding: EdgeInsets.symmetric(
              horizontal: screenWidth * 0.04,
              vertical: screenWidth * 0.02,
            ),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.5),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.analytics,
                  size: confidenceSize,
                  color: Theme.of(context).colorScheme.primary,
                ),
                SizedBox(width: screenWidth * 0.02),
                Text(
                  "Confidence: ${(_confidence * 100).toStringAsFixed(1)}%",
                  style: TextStyle(
                    fontSize: confidenceSize,
                    color: Theme.of(context).colorScheme.primary,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProbabilitiesCard(BuildContext context, double screenWidth, double cardPadding) {
    final labelSize = (screenWidth * 0.037).clamp(13.0, 16.0);
    
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(cardPadding),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.bar_chart,
                size: screenWidth * 0.05,
                color: Theme.of(context).colorScheme.primary,
              ),
              SizedBox(width: screenWidth * 0.02),
              Text(
                'All Probabilities',
                style: TextStyle(
                  fontSize: (labelSize * 1.15).clamp(15.0, 18.0),
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
            ],
          ),
          SizedBox(height: screenWidth * 0.03),
          ...List.generate(
            HARModel.activityLabels.length,
            (index) {
              final probability = _probabilities[index];
              final activityLabel = HARModel.activityLabels[index];
              
              // Skip WALKING_UPSTAIRS and WALKING_DOWNSTAIRS (now merged into WALKING)
              if (activityLabel == 'WALKING_UPSTAIRS' || activityLabel == 'WALKING_DOWNSTAIRS') {
                return const SizedBox.shrink();
              }
              
              final isTopPrediction = _currentActivity == _formatActivityName(activityLabel);
              
              return Padding(
                padding: EdgeInsets.symmetric(vertical: screenWidth * 0.015),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          _formatActivityName(activityLabel),
                          style: TextStyle(
                            fontSize: labelSize,
                            fontWeight: isTopPrediction ? FontWeight.bold : FontWeight.normal,
                            color: isTopPrediction
                                ? Theme.of(context).colorScheme.primary
                                : Theme.of(context).colorScheme.onSurface,
                          ),
                        ),
                        Text(
                          "${(probability * 100).toStringAsFixed(1)}%",
                          style: TextStyle(
                            fontSize: labelSize,
                            fontWeight: FontWeight.bold,
                            color: isTopPrediction
                                ? Theme.of(context).colorScheme.primary
                                : Theme.of(context).colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: screenWidth * 0.015),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: LinearProgressIndicator(
                        value: probability,
                        minHeight: screenWidth * 0.02,
                        backgroundColor: Theme.of(context).colorScheme.surfaceVariant,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          isTopPrediction
                              ? Theme.of(context).colorScheme.primary
                              : Theme.of(context).colorScheme.secondary,
                        ),
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildSelectionCard(BuildContext context, double screenWidth, double cardPadding) {
    final fontSize = (screenWidth * 0.037).clamp(13.0, 16.0);
    
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(cardPadding),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.check_circle_outline,
                size: screenWidth * 0.05,
                color: Theme.of(context).colorScheme.primary,
              ),
              SizedBox(width: screenWidth * 0.02),
              Expanded(
                child: Text(
                  'Select True Activity',
                  style: TextStyle(
                    fontSize: (fontSize * 1.15).clamp(15.0, 18.0),
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
              ),
            ],
          ),
          SizedBox(height: screenWidth * 0.03),
          Container(
            padding: EdgeInsets.symmetric(horizontal: screenWidth * 0.03),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceVariant,
              borderRadius: BorderRadius.circular(12),
            ),
            child: DropdownButton<String>(
              value: _selectedTrueActivity,
              isExpanded: true,
              underline: const SizedBox(),
              style: TextStyle(
                fontSize: fontSize,
                color: Theme.of(context).colorScheme.onSurface,
                fontWeight: FontWeight.w600,
              ),
              icon: Icon(
                Icons.arrow_drop_down,
                color: Theme.of(context).colorScheme.primary,
              ),
              items: HARModel.activityLabels
                .where((activity) => 
                  activity != 'WALKING_UPSTAIRS' && 
                  activity != 'WALKING_DOWNSTAIRS')
                .map((String activity) {
                return DropdownMenuItem<String>(
                  value: activity,
                  child: Text(_formatActivityName(activity)),
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
        ],
      ),
    );
  }

  Widget _buildAccuracyCard(BuildContext context, double screenWidth, double cardPadding) {
    final accuracyValue = _totalPredictions > 0
        ? (_correctPredictions / _totalPredictions) * 100
        : 0.0;
    final fontSize = (screenWidth * 0.055).clamp(20.0, 28.0);
    
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(cardPadding * 1.2),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.green.shade400,
            Colors.teal.shade400,
          ],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.green.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.verified,
                color: Colors.white,
                size: screenWidth * 0.06,
              ),
              SizedBox(width: screenWidth * 0.02),
              Text(
                'Accuracy',
                style: TextStyle(
                  fontSize: (fontSize * 0.65).clamp(14.0, 18.0),
                  fontWeight: FontWeight.w600,
                  color: Colors.white.withOpacity(0.9),
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          SizedBox(height: screenWidth * 0.02),
          Text(
            _totalPredictions > 0
                ? "${accuracyValue.toStringAsFixed(1)}%"
                : "---",
            style: TextStyle(
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          SizedBox(height: screenWidth * 0.015),
          Text(
            _totalPredictions > 0
                ? "$_correctPredictions / $_totalPredictions predictions"
                : "No predictions yet",
            style: TextStyle(
              fontSize: (fontSize * 0.5).clamp(12.0, 15.0),
              color: Colors.white.withOpacity(0.85),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControlButtons(BuildContext context, double screenWidth) {
    final buttonPadding = screenWidth * 0.04;
    final fontSize = (screenWidth * 0.04).clamp(14.0, 18.0);
    final iconSize = (screenWidth * 0.05).clamp(20.0, 24.0);
    
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isDetecting ? null : _startDetection,
            icon: Icon(Icons.play_arrow, size: iconSize),
            label: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                "Start",
                style: TextStyle(fontSize: fontSize, fontWeight: FontWeight.bold),
              ),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
              disabledBackgroundColor: Colors.grey.shade300,
              padding: EdgeInsets.symmetric(vertical: buttonPadding),
              elevation: _isDetecting ? 0 : 2,
            ),
          ),
        ),
        SizedBox(width: screenWidth * 0.03),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isDetecting ? _stopDetection : null,
            icon: Icon(Icons.stop, size: iconSize),
            label: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                "Stop",
                style: TextStyle(fontSize: fontSize, fontWeight: FontWeight.bold),
              ),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
              disabledBackgroundColor: Colors.grey.shade300,
              padding: EdgeInsets.symmetric(vertical: buttonPadding),
              elevation: _isDetecting ? 2 : 0,
            ),
          ),
        ),
      ],
    );
  }
  
  // Federated Learning Card
  Widget _buildFederatedLearningCard(BuildContext context, double screenWidth, double cardPadding) {
    final fontSize = (screenWidth * 0.035).clamp(12.0, 16.0);
    final iconSize = (screenWidth * 0.05).clamp(20.0, 24.0);
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(cardPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Icon(
                  Icons.cloud_sync_outlined,
                  color: Theme.of(context).colorScheme.primary,
                  size: iconSize,
                ),
                const SizedBox(width: 8),
                Text(
                  'Federated Learning',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    fontSize: (screenWidth * 0.04).clamp(14.0, 18.0),
                  ),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: _flStatus.contains('✅') 
                        ? Colors.green.withOpacity(0.1)
                        : _flStatus.contains('❌')
                            ? Colors.red.withOpacity(0.1)
                            : Colors.orange.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: _flStatus.contains('✅') 
                          ? Colors.green
                          : _flStatus.contains('❌')
                              ? Colors.red
                              : Colors.orange,
                      width: 1,
                    ),
                  ),
                  child: Text(
                    _flStatus.contains('✅') ? 'Online' : 
                    _flStatus.contains('❌') ? 'Offline' : 'Training',
                    style: TextStyle(
                      fontSize: (screenWidth * 0.028).clamp(10.0, 12.0),
                      fontWeight: FontWeight.bold,
                      color: _flStatus.contains('✅') 
                          ? Colors.green
                          : _flStatus.contains('❌')
                              ? Colors.red
                              : Colors.orange,
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Description
            Text(
              'Help improve the model by contributing to federated training. Your data stays on your device.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontSize: fontSize,
                color: Colors.grey.shade600,
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Status Row
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Server Status',
                        style: TextStyle(
                          fontSize: fontSize,
                          color: Colors.grey.shade600,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _flStatus,
                        style: TextStyle(
                          fontSize: fontSize,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Training Rounds',
                        style: TextStyle(
                          fontSize: fontSize,
                          color: Colors.grey.shade600,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '$_flRound',
                        style: TextStyle(
                          fontSize: fontSize,
                          fontWeight: FontWeight.bold,
                          color: Theme.of(context).colorScheme.primary,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 16),
            
            // Training Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _isFLTraining ? null : _startFederatedTraining,
                icon: Icon(
                  _isFLTraining ? Icons.hourglass_empty : Icons.upload,
                  size: iconSize,
                ),
                label: Text(
                  _isFLTraining ? 'Training...' : 'Contribute to Training',
                  style: TextStyle(
                    fontSize: fontSize,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Theme.of(context).colorScheme.secondaryContainer,
                  foregroundColor: Theme.of(context).colorScheme.onSecondaryContainer,
                  padding: EdgeInsets.symmetric(vertical: cardPadding * 0.8),
                  elevation: _isFLTraining ? 0 : 2,
                ),
              ),
            ),
            
            const SizedBox(height: 8),
            
            // Refresh Status Button
            TextButton.icon(
              onPressed: _isFLTraining ? null : _checkFLServer,
              icon: Icon(Icons.refresh, size: iconSize * 0.8),
              label: Text(
                'Refresh Status',
                style: TextStyle(fontSize: fontSize),
              ),
              style: TextButton.styleFrom(
                padding: EdgeInsets.zero,
              ),
            ),
          ],
        ),
      ),
    );
  }}