import 'dart:convert';
import 'dart:math';
import 'package:http/http.dart' as http;

/// Federated Learning Client for HAR App
/// Communicates with Python Flask server for federated training
class FederatedClient {
  final String serverUrl;
  final String clientId;
  
  // Training state
  bool isTraining = false;
  int currentRound = 0;
  List<Map<String, dynamic>> trainingHistory = [];
  
  FederatedClient({
    required this.serverUrl,
    String? clientId,
  }) : clientId = clientId ?? _generateClientId();
  
  /// Generate unique client ID based on timestamp
  static String _generateClientId() {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final random = Random().nextInt(1000);
    return 'flutter_client_${timestamp}_$random';
  }
  
  /// Check server health
  Future<bool> checkServerConnection() async {
    try {
      final response = await http.get(
        Uri.parse('$serverUrl/health'),
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print('✅ Connected to FL Server: ${data['service']}');
        return true;
      }
      return false;
    } catch (e) {
      print('❌ Server connection failed: $e');
      return false;
    }
  }
  
  /// Get server status
  Future<Map<String, dynamic>?> getServerStatus() async {
    try {
      final response = await http.get(
        Uri.parse('$serverUrl/status'),
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      return null;
    } catch (e) {
      print('❌ Failed to get server status: $e');
      return null;
    }
  }
  
  /// Extract dummy weights from local model
  /// TODO: Replace with actual weight extraction from TFLite model
  List<double> _extractModelWeights() {
    // Simulated weights - in production, extract from actual model
    // This would require custom TFLite implementation
    print('⚙️ Extracting model weights (simulated)...');
    
    // Generate dummy weights (in real app, get from interpreter)
    final random = Random(DateTime.now().millisecondsSinceEpoch);
    return List.generate(1000, (i) => random.nextDouble() * 2 - 1);
  }
  
  /// Upload local model weights to server
  Future<Map<String, dynamic>?> uploadWeights({
    required int numSamples,
    List<double>? customWeights,
  }) async {
    try {
      print('📤 Uploading weights to server...');
      
      // Extract or use provided weights
      final weights = customWeights ?? _extractModelWeights();
      
      final response = await http.post(
        Uri.parse('$serverUrl/upload_weights'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'client_id': clientId,
          'weights': weights,
          'num_samples': numSamples,
          'timestamp': DateTime.now().toIso8601String(),
        }),
      ).timeout(const Duration(seconds: 30));
      
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        print('✅ Weights uploaded successfully');
        print('📊 Server response: ${result['message']}');
        
        if (result['aggregated'] == true) {
          print('🎯 Aggregation completed! Downloading global model...');
        }
        
        return result;
      } else {
        print('❌ Upload failed: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('❌ Error uploading weights: $e');
      return null;
    }
  }
  
  /// Download global model weights from server
  Future<List<double>?> downloadGlobalWeights() async {
    try {
      print('📥 Downloading global weights from server...');
      
      final response = await http.get(
        Uri.parse('$serverUrl/get_global_weights'),
      ).timeout(const Duration(seconds: 30));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final weights = List<double>.from(data['weights']);
        print('✅ Global weights downloaded (Round ${data['round']})');
        return weights;
      } else if (response.statusCode == 404) {
        print('⏳ Global weights not yet available');
        return null;
      } else {
        print('❌ Download failed: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('❌ Error downloading weights: $e');
      return null;
    }
  }
  
  /// Update local model with global weights
  /// TODO: Implement actual model update logic
  void updateLocalModel(List<double> globalWeights) {
    print('🔄 Updating local model with global weights...');
    // In production: update TFLite interpreter with new weights
    // This requires custom TFLite implementation
    print('✅ Local model updated (simulated)');
  }
  
  /// Participate in a complete federated learning round
  Future<bool> participateInFederatedRound({
    required int numLocalSamples,
    List<double>? precomputedWeights,
  }) async {
    if (isTraining) {
      print('⚠️ Already participating in training');
      return false;
    }
    
    try {
      isTraining = true;
      print('\n${'='*60}');
      print('🚀 Starting Federated Learning Round');
      print('${'='*60}');
      print('👤 Client ID: $clientId');
      print('📊 Local samples: $numLocalSamples');
      
      // Step 1: Check server connection
      print('\n📡 Step 1/4: Checking server connection...');
      if (!await checkServerConnection()) {
        throw Exception('Cannot connect to FL server');
      }
      
      // Step 2: Upload weights
      print('\n📤 Step 2/4: Uploading local weights...');
      final uploadResult = await uploadWeights(
        numSamples: numLocalSamples,
        customWeights: precomputedWeights,
      );
      
      if (uploadResult == null) {
        throw Exception('Failed to upload weights');
      }
      
      // Step 3: Wait for aggregation if needed
      if (uploadResult['aggregated'] != true) {
        print('\n⏳ Step 3/4: Waiting for other clients...');
        print('   ${uploadResult['clients_waiting']}/${uploadResult['clients_needed'] + uploadResult['clients_waiting']} clients ready');
        print('   ℹ️ Global model will be available after aggregation');
        
        // Store partial result
        trainingHistory.add({
          'timestamp': DateTime.now(),
          'round': currentRound,
          'status': 'waiting',
          'uploaded': true,
        });
        
        return false; // Not complete yet
      }
      
      // Step 4: Download global model
      print('\n📥 Step 4/4: Downloading global model...');
      await Future.delayed(const Duration(seconds: 2)); // Give server time to aggregate
      
      final globalWeights = await downloadGlobalWeights();
      if (globalWeights == null) {
        print('⚠️ Could not download global model, but upload was successful');
        return true; // Still consider it success
      }
      
      // Step 5: Update local model
      print('\n🔄 Step 5/4: Updating local model...');
      updateLocalModel(globalWeights);
      
      // Success!
      currentRound++;
      trainingHistory.add({
        'timestamp': DateTime.now(),
        'round': currentRound,
        'status': 'completed',
        'num_samples': numLocalSamples,
      });
      
      print('\n${'='*60}');
      print('✅ Federated Learning Round Complete!');
      print('${'='*60}');
      print('🎯 Round: $currentRound');
      print('📊 Model updated successfully\n');
      
      return true;
      
    } catch (e) {
      print('\n❌ Federated learning failed: $e\n');
      trainingHistory.add({
        'timestamp': DateTime.now(),
        'round': currentRound,
        'status': 'failed',
        'error': e.toString(),
      });
      return false;
    } finally {
      isTraining = false;
    }
  }
  
  /// Simulate local training and participate
  /// Use this when you don't have actual training implemented
  Future<bool> simulateTrainingRound() async {
    print('🎭 Simulating local training...');
    
    // Simulate some processing time
    await Future.delayed(const Duration(seconds: 2));
    
    // Participate with random weights
    return await participateInFederatedRound(
      numLocalSamples: Random().nextInt(100) + 50, // 50-150 samples
    );
  }
  
  /// Get training statistics
  Map<String, dynamic> getStatistics() {
    final completed = trainingHistory.where((h) => h['status'] == 'completed').length;
    final failed = trainingHistory.where((h) => h['status'] == 'failed').length;
    
    return {
      'client_id': clientId,
      'total_rounds': currentRound,
      'completed_rounds': completed,
      'failed_rounds': failed,
      'is_training': isTraining,
      'history': trainingHistory,
    };
  }
  
  /// Reset client state
  void reset() {
    isTraining = false;
    currentRound = 0;
    trainingHistory.clear();
    print('🔄 Client state reset');
  }
}
