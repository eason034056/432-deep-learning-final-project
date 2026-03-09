"""
Flask backend server for mmWave Human Identification GUI

This server provides REST API endpoints for:
- Uploading raw mesh data
- Configuring preprocessing
- Starting model training
- Monitoring training progress
- Generating reports
- Visualizing results
- Downloading trained models
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.training_manager import TrainingManager
from backend.utils import allowed_file, get_project_root, load_config, save_config

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(get_project_root(), 'data', 'raw')
app.config['PROCESSED_FOLDER'] = os.path.join(get_project_root(), 'data', 'processed')
app.config['RESULTS_FOLDER'] = os.path.join(get_project_root(), 'results')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Training manager
training_manager = TrainingManager()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Frontend Routes ====================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, 'index.html')


# ==================== API Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config = load_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        new_config = request.json
        save_config(new_config)
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Upload raw mesh files (.ply, .obj)
    
    Expects: multipart/form-data with files
    Returns: List of uploaded filenames and status
    """
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
                logger.info(f"Uploaded file: {filename}")
            else:
                errors.append(f"Invalid file: {file.filename}")
        
        return jsonify({
            'success': True,
            'uploaded': uploaded_files,
            'errors': errors,
            'count': len(uploaded_files)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/data/files', methods=['GET'])
def list_data_files():
    """List all uploaded data files"""
    try:
        raw_dir = app.config['UPLOAD_FOLDER']
        files = []
        
        if os.path.exists(raw_dir):
            for filename in os.listdir(raw_dir):
                if allowed_file(filename):
                    filepath = os.path.join(raw_dir, filename)
                    files.append({
                        'name': filename,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    Preprocess raw mesh data
    
    Expects JSON with preprocessing parameters:
    - num_points: int (default: 200)
    - normalize: bool
    - augmentation settings
    """
    try:
        params = request.json or {}
        
        # Update config with preprocessing params
        config = load_config()
        config['data'].update(params.get('data', {}))
        config['augmentation'].update(params.get('augmentation', {}))
        config['data'].update(params.get('data', {}))
        config['augmentation'].update(params.get('augmentation', {}))
        save_config(config)
        
        # Log detailed parameters
        logger.info("Preprocessing started with parameters:")
        logger.info(f"  Num Points: {config['data'].get('num_points')}")
        logger.info(f"  Samples/Mesh: {config['data'].get('samples_per_mesh')}")
        logger.info(f"  Normalize Center: {config['data'].get('normalize_center')}")
        logger.info(f"  Normalize Scale: {config['data'].get('normalize_scale')}")
        logger.info(f"  Augmentation: {json.dumps(config['augmentation'], indent=2)}")
        
        # Start preprocessing as a background job
        job_id = training_manager.start_preprocessing(config)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Preprocessing started'
        })
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Start model training
    
    Expects JSON with:
    - model_type: 'mlp', 'cnn1d', or 'pointnet'
    - hyperparameters: batch_size, learning_rate, num_epochs, etc.
    """
    try:
        params = request.json or {}
        model_type = params.get('model_type', 'pointnet2')
        
        # Validate model type (classification + autoencoder)
        valid_models = ['mlp', 'cnn1d', 'pointnet', 'pointnet2', 'mlp_ae', 'pointnet2_ae']
        if model_type not in valid_models:
            return jsonify({
                'success': False,
                'error': f'Invalid model type: {model_type}. Valid: {valid_models}'
            }), 400
        
        # Update config with training params
        config = load_config()
        config['model']['type'] = model_type
        config['training'].update(params.get('training', {}))
        save_config(config)
        
        # Start training as a background job
        job_id = training_manager.start_training(model_type, config)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'model_type': model_type,
            'message': 'Training started'
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """Get training job status and metrics"""
    try:
        status = training_manager.get_status(job_id)
        
        if status is None:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        return jsonify({
            'success': True,
            **status
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/logs/<job_id>', methods=['GET'])
def get_training_logs(job_id):
    """Get training logs for a job"""
    try:
        logs = training_manager.get_logs(job_id)
        
        if logs is None:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        return jsonify({
            'success': True,
            'logs': logs
        })
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training/jobs', methods=['GET'])
def list_training_jobs():
    """List all training jobs"""
    try:
        jobs = training_manager.list_jobs()
        return jsonify({
            'success': True,
            'jobs': jobs
        })
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate/<job_id>', methods=['POST'])
def evaluate_model(job_id):
    """
    Evaluate a trained model
    
    Expects JSON with:
    - metrics: list of metrics to compute
    """
    try:
        results = training_manager.evaluate_model(job_id)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Job not found or not completed'
            }), 404
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/report/generate/<job_id>', methods=['POST'])
def generate_report(job_id):
    """Generate comprehensive evaluation report"""
    try:
        report_path = training_manager.generate_report(job_id)
        
        if report_path is None:
            return jsonify({
                'success': False,
                'error': 'Job not found or not completed'
            }), 404
        
        return jsonify({
            'success': True,
            'report_path': report_path,
            'message': 'Report generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/visualization/<job_id>', methods=['GET'])
def get_visualizations(job_id):
    """Get training visualizations and plots"""
    try:
        viz_data = training_manager.get_visualizations(job_id)
        
        if viz_data is None:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        
        return jsonify({
            'success': True,
            **viz_data
        })
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/model/<job_id>', methods=['GET'])
def download_model(job_id):
    """Download trained model checkpoint"""
    try:
        model_path = training_manager.get_model_path(job_id)
        
        if model_path is None or not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f'model_{job_id}.pth'
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/report/<job_id>', methods=['GET'])
def download_report(job_id):
    """Download evaluation report"""
    try:
        report_path = training_manager.get_report_path(job_id)
        
        if report_path is None or not os.path.exists(report_path):
            return jsonify({
                'success': False,
                'error': 'Report not found'
            }), 404
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f'report_{job_id}.json'
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
