#!/usr/bin/env python3
"""
Federated Learning Server for Flutter HAR App
Uses Flask to receive weights from mobile clients and perform FedAvg aggregation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from threading import Lock
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile clients

# Global state
client_weights = {}
global_weights = None
weights_lock = Lock()
round_number = 0
REQUIRED_CLIENTS = 1  # Minimum clients needed for aggregation (set to 1 for testing)
MAX_WAIT_TIME = 300  # 5 minutes max wait

class FederatedServer:
    def __init__(self):
        self.clients_history = []
        self.aggregation_history = []
    
    def add_client_weights(self, client_id, weights, num_samples):
        """Add client weights to the pool"""
        with weights_lock:
            client_weights[client_id] = {
                'weights': np.array(weights, dtype=np.float32),
                'num_samples': num_samples,
                'timestamp': time.time()
            }
            
            self.clients_history.append({
                'client_id': client_id,
                'round': round_number,
                'num_samples': num_samples,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✅ Client {client_id} uploaded weights ({num_samples} samples)")
            logger.info(f"📊 Progress: {len(client_weights)}/{REQUIRED_CLIENTS} clients")
            
            return len(client_weights)
    
    def aggregate_weights(self):
        """Perform FedAvg aggregation"""
        global client_weights, global_weights, round_number
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Starting FedAvg Aggregation - Round {round_number + 1}")
        logger.info(f"{'='*60}")
        
        with weights_lock:
            if len(client_weights) == 0:
                logger.warning("⚠️ No client weights available")
                return False
            
            # Calculate total samples across all clients
            total_samples = sum(c['num_samples'] for c in client_weights.values())
            logger.info(f"📊 Total training samples: {total_samples}")
            
            # Get weight shape from first client
            first_client = list(client_weights.values())[0]
            weight_shape = first_client['weights'].shape
            
            # Initialize aggregated weights
            aggregated = np.zeros(weight_shape, dtype=np.float32)
            
            # Weighted averaging (FedAvg formula)
            logger.info(f"\n📐 Computing weighted average:")
            for client_id, client_data in client_weights.items():
                weight = client_data['num_samples'] / total_samples
                contribution = weight * client_data['weights']
                aggregated += contribution
                
                logger.info(f"  • {client_id}: {client_data['num_samples']} samples "
                          f"(weight: {weight:.4f})")
            
            global_weights = aggregated
            round_number += 1
            
            # Store aggregation history
            self.aggregation_history.append({
                'round': round_number,
                'num_clients': len(client_weights),
                'total_samples': total_samples,
                'timestamp': datetime.now().isoformat()
            })
            
            # Clear client weights for next round
            num_clients = len(client_weights)
            client_weights.clear()
        
        logger.info(f"\n✅ Aggregation Complete!")
        logger.info(f"🎯 Updated global model (Round {round_number})")
        logger.info(f"👥 Aggregated from {num_clients} clients")
        logger.info(f"{'='*60}\n")
        
        return True
    
    def get_statistics(self):
        """Get server statistics"""
        return {
            'current_round': round_number,
            'clients_waiting': len(client_weights),
            'required_clients': REQUIRED_CLIENTS,
            'has_global_weights': global_weights is not None,
            'total_clients_served': len(self.clients_history),
            'total_rounds_completed': len(self.aggregation_history),
            'recent_aggregations': self.aggregation_history[-5:] if self.aggregation_history else []
        }

# Initialize server
fed_server = FederatedServer()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Federated Learning Server',
        'version': '1.0.0'
    })

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    """Receive weights from mobile client"""
    try:
        data = request.json
        
        # Validate request
        if not all(k in data for k in ['client_id', 'weights', 'num_samples']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        client_id = data['client_id']
        weights = data['weights']
        num_samples = data['num_samples']
        
        # Add client weights
        num_clients = fed_server.add_client_weights(client_id, weights, num_samples)
        
        # Check if we should aggregate
        should_aggregate = num_clients >= REQUIRED_CLIENTS
        
        if should_aggregate:
            success = fed_server.aggregate_weights()
            
            return jsonify({
                'status': 'success',
                'message': 'Weights uploaded and aggregated',
                'round': round_number,
                'aggregated': True,
                'global_weights_available': True
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'Weights uploaded, waiting for more clients',
                'round': round_number,
                'aggregated': False,
                'clients_waiting': num_clients,
                'clients_needed': REQUIRED_CLIENTS - num_clients
            })
    
    except Exception as e:
        logger.error(f"❌ Error uploading weights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_global_weights', methods=['GET'])
def get_global_weights():
    """Send global weights to client"""
    try:
        if global_weights is None:
            return jsonify({
                'error': 'No global weights available yet',
                'message': 'Server is waiting for initial client uploads'
            }), 404
        
        return jsonify({
            'status': 'success',
            'weights': global_weights.tolist(),
            'round': round_number,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Error sending global weights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get server status and statistics"""
    stats = fed_server.get_statistics()
    return jsonify(stats)

@app.route('/trigger_aggregation', methods=['POST'])
def trigger_aggregation():
    """Manually trigger aggregation (admin endpoint)"""
    try:
        if len(client_weights) == 0:
            return jsonify({
                'error': 'No client weights available',
                'clients_waiting': 0
            }), 400
        
        success = fed_server.aggregate_weights()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Manual aggregation completed',
                'round': round_number
            })
        else:
            return jsonify({
                'error': 'Aggregation failed'
            }), 500
    
    except Exception as e:
        logger.error(f"❌ Error in manual aggregation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset server state (admin endpoint)"""
    global client_weights, global_weights, round_number
    
    with weights_lock:
        client_weights.clear()
        global_weights = None
        round_number = 0
    
    logger.info("🔄 Server state reset")
    
    return jsonify({
        'status': 'success',
        'message': 'Server state reset'
    })

@app.route('/')
def home():
    """Home page with server info"""
    stats = fed_server.get_statistics()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated Learning Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .stat {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
            .label {{ font-weight: bold; color: #666; }}
            .value {{ color: #0066cc; }}
            .status {{ padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
            .status.online {{ background: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Federated Learning Server</h1>
            <p><span class="status online">ONLINE</span></p>
            
            <h2>📊 Server Statistics</h2>
            <div class="stat">
                <span class="label">Current Round:</span>
                <span class="value">{stats['current_round']}</span>
            </div>
            <div class="stat">
                <span class="label">Clients Waiting:</span>
                <span class="value">{stats['clients_waiting']} / {stats['required_clients']}</span>
            </div>
            <div class="stat">
                <span class="label">Global Model Available:</span>
                <span class="value">{'Yes' if stats['has_global_weights'] else 'No'}</span>
            </div>
            <div class="stat">
                <span class="label">Total Clients Served:</span>
                <span class="value">{stats['total_clients_served']}</span>
            </div>
            <div class="stat">
                <span class="label">Rounds Completed:</span>
                <span class="value">{stats['total_rounds_completed']}</span>
            </div>
            
            <h2>📡 API Endpoints</h2>
            <ul>
                <li><code>POST /upload_weights</code> - Upload client weights</li>
                <li><code>GET /get_global_weights</code> - Download global model</li>
                <li><code>GET /status</code> - Server statistics (JSON)</li>
                <li><code>GET /health</code> - Health check</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🚀 Federated Learning Server Starting...")
    logger.info("="*60)
    logger.info(f"📡 Minimum clients per round: {REQUIRED_CLIENTS}")
    logger.info(f"⏱️  Max wait time: {MAX_WAIT_TIME}s")
    logger.info(f"🌐 Server will run on: http://0.0.0.0:5000")
    logger.info(f"📱 Mobile clients should connect to: http://YOUR_IP:5000")
    logger.info("="*60)
    logger.info("\n💡 Tip: Use 'ipconfig' (Windows) or 'ifconfig' (Linux/Mac)")
    logger.info("   to find your local IP address for mobile testing.\n")
    
    # Run Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
