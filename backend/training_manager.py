"""
Training Manager for background model training and job management

Handles:
- Starting training jobs in background threads
- Tracking training progress
- Managing job queue
- Collecting and storing metrics
- Evaluating trained models
"""

import os
import json
import uuid
import threading
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional, List
from queue import Queue
import traceback

from backend.train_integration import (
    train_model,
    train_ae_model,
    evaluate_model as eval_model,
    evaluate_ae_model,
    preprocess_faust_dataset,
    _is_autoencoder,
)
from backend.utils import get_project_root, load_config

logger = logging.getLogger(__name__)


class TrainingJob:
    """Represents a single training job"""
    
    def __init__(self, job_id: str, model_type: str, config: dict):
        self.job_id = job_id
        self.model_type = model_type
        self.config = deepcopy(config)
        self.status = 'queued'  # queued, running, completed, failed
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = config['training']['num_epochs']
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.logs = []
        self.error = None
        self.start_time = None
        self.end_time = None
        self.model_path = None
        self.report_path = None
        self.evaluation_results = None
        self.best_val_acc = 0.0
    
    def log(self, message: str):
        """Add a log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        logger.info(f"Job {self.job_id}: {message}")
    
    def update_progress(self, epoch: int, metrics: dict):
        """Update training progress"""
        self.current_epoch = epoch
        self.progress = (epoch / self.total_epochs) * 100

        # Store metrics
        if 'train_loss' in metrics:
            self.metrics['train_loss'].append(metrics['train_loss'])
        if 'train_acc' in metrics:
            self.metrics['train_acc'].append(metrics['train_acc'])
        if 'val_loss' in metrics:
            self.metrics['val_loss'].append(metrics['val_loss'])
        if 'val_acc' in metrics:
            self.metrics['val_acc'].append(metrics['val_acc'])
            self.best_val_acc = max(self.best_val_acc, metrics['val_acc'])
        else:
            # Autoencoder: no accuracy, append 0 for chart compatibility
            self.metrics['train_acc'].append(0.0)
            self.metrics['val_acc'].append(0.0)
    
    def to_dict(self) -> dict:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'model_type': self.model_type,
            'config': self.config,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'metrics': self.metrics,
            'logs': self.logs,
            'best_val_acc': self.best_val_acc,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'model_path': self.model_path,
            'report_path': self.report_path,
            'evaluation_results': self.evaluation_results,
        }


class TrainingManager:
    """Manages training jobs and background execution"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue = Queue()
        self.lock = threading.Lock()
        self.jobs_file = os.path.join(get_project_root(), 'results', 'jobs.json')
        
        # Load persisted jobs
        self._load_jobs()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("TrainingManager initialized")

    def _save_jobs(self):
        """Save jobs to disk"""
        try:
            jobs_data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            with open(self.jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def _load_jobs(self):
        """Load jobs from disk"""
        if not os.path.exists(self.jobs_file):
            return
            
        try:
            with open(self.jobs_file, 'r') as f:
                jobs_data = json.load(f)
                
            for job_id, data in jobs_data.items():
                job_config = data.get('config') or load_config()
                job = TrainingJob(job_id, data['model_type'], job_config)
                job.status = data['status']
                job.progress = data['progress']
                job.current_epoch = data['current_epoch']
                job.metrics = data['metrics']
                job.logs = data.get('logs', [])
                job.best_val_acc = data.get('best_val_acc', 0.0)
                job.model_path = data.get('model_path')
                job.report_path = data.get('report_path')
                job.evaluation_results = data.get('evaluation_results')
                job.error = data.get('error')
                
                if data.get('start_time'):
                    job.start_time = datetime.fromisoformat(data['start_time'])
                if data.get('end_time'):
                    job.end_time = datetime.fromisoformat(data['end_time'])

                if job.report_path is None:
                    report_path = self._build_report_path(job_id)
                    if os.path.exists(report_path):
                        job.report_path = report_path
                
                self.jobs[job_id] = job
                
            logger.info(f"Loaded {len(self.jobs)} jobs from disk")
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
    
    def start_preprocessing(self, config: dict) -> None:
        """Run preprocessing synchronously."""
        try:
            logger.info("Starting preprocessing")
            preprocess_faust_dataset(config)
            logger.info("Preprocessing completed")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def start_training(self, model_type: str, config: dict) -> str:
        """
        Start a new training job
        
        Args:
            model_type: Model type ('mlp', 'cnn1d', 'pointnet', 'mlp_ae', or 'pointnet_ae')
            config: Configuration dictionary
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        # Create training job
        job = TrainingJob(job_id, model_type, config)
        
        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job_id)
            self._save_jobs()
        
        job.log(f"Training job created for model: {model_type}")
        
        return job_id
    
    def _worker(self):
        """Background worker thread that processes training jobs"""
        logger.info("Training worker thread started")
        
        while True:
            try:
                # Get next job from queue
                job_id = self.job_queue.get()
                
                with self.lock:
                    job = self.jobs.get(job_id)
                
                if job is None:
                    continue
                
                # Run training
                self._run_training(job)
                
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
    
    def _run_training(self, job: TrainingJob):
        """Run training for a job"""
        try:
            job.status = 'running'
            job.start_time = datetime.now()
            job.log("Starting training...")
            
            # Prepare callback for progress updates
            def progress_callback(epoch, metrics):
                job.update_progress(epoch, metrics)
                if _is_autoencoder(job.model_type):
                    job.log(f"Epoch {epoch}/{job.total_epochs} - "
                           f"Train Loss: {metrics.get('train_loss', 0):.6f}, "
                           f"Val Loss: {metrics.get('val_loss', 0):.6f}")
                else:
                    job.log(f"Epoch {epoch}/{job.total_epochs} - "
                           f"Train Loss: {metrics.get('train_loss', 0):.4f}, "
                           f"Train Acc: {metrics.get('train_acc', 0):.4f}, "
                           f"Val Loss: {metrics.get('val_loss', 0):.4f}, "
                           f"Val Acc: {metrics.get('val_acc', 0):.4f}")

            # Call training function with callback
            if _is_autoencoder(job.model_type):
                model_path = self._train_ae_with_callback(
                    job.model_type,
                    job.config,
                    progress_callback
                )
            else:
                model_path = self._train_with_callback(
                    job.model_type,
                    job.config,
                    progress_callback
                )
            
            job.model_path = model_path
            job.status = 'completed'
            job.end_time = datetime.now()
            job.progress = 100.0
            job.log("Training completed successfully!")
            self._save_jobs()
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.end_time = datetime.now()
            job.log(f"Training failed: {e}")
            logger.error(f"Training job {job.job_id} failed: {e}\n{traceback.format_exc()}")
            self._save_jobs()
    
    def _train_with_callback(self, model_type: str, config: dict, callback):
        """
        Train model with progress callback
        
        This is a wrapper around the existing train.py script
        We'll modify train.py to accept a callback parameter
        """
        # Set output directory for this specific job
        results_dir = os.path.join(get_project_root(), 'results', 'checkpoints', model_type)
        os.makedirs(results_dir, exist_ok=True)
        
        # Train model (this will be integrated with modified train.py)
        model_path = train_model(
            model_type=model_type,
            config=config,
            progress_callback=callback
        )
        
        return model_path

    def _train_ae_with_callback(self, model_type: str, config: dict, callback):
        """Train autoencoder with progress callback."""
        results_dir = os.path.join(get_project_root(), 'results', 'checkpoints', model_type)
        os.makedirs(results_dir, exist_ok=True)
        model_path = train_ae_model(
            model_type=model_type,
            config=config,
            progress_callback=callback
        )
        return model_path

    def get_status(self, job_id: str) -> Optional[dict]:
        """Get job status"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            d = job.to_dict()
            d['model_type'] = job.model_type
            return d
    
    def get_logs(self, job_id: str) -> Optional[List[str]]:
        """Get job logs"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return job.logs
    
    def list_jobs(self) -> List[dict]:
        """List all jobs"""
        with self.lock:
            jobs = list(self.jobs.values())

        jobs.sort(
            key=lambda job: (
                job.end_time or job.start_time or datetime.min,
                job.job_id,
            ),
            reverse=True,
        )
        return [job.to_dict() for job in jobs]
    
    def evaluate_model(self, job_id: str) -> Optional[dict]:
        """Evaluate a trained model"""
        with self.lock:
            job = self.jobs.get(job_id)

        if job is None or job.status != 'completed':
            return None

        try:
            if _is_autoencoder(job.model_type):
                return evaluate_ae_model(
                    model_type=job.model_type,
                    checkpoint_path=job.model_path,
                    config=job.config
                )
            return eval_model(
                model_type=job.model_type,
                checkpoint_path=job.model_path,
                config=job.config
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def generate_report(self, job_id: str) -> Optional[dict]:
        """Generate evaluation report"""
        with self.lock:
            job = self.jobs.get(job_id)
        
        if job is None or job.status != 'completed':
            return None
        
        try:
            # Generate report (JSON format; PDF/HTML not implemented)
            report_path = self._build_report_path(job_id)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Evaluate and save results
            results = self.evaluate_model(job_id)
            if results is None:
                return None
            
            report_data = {
                'job_id': job_id,
                'model_type': job.model_type,
                'training_summary': job.to_dict(),
                'evaluation_results': results,
                'generated_at': datetime.now().isoformat(),
                'report_path': report_path,
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            with self.lock:
                job.report_path = report_path
                job.evaluation_results = results
                self._save_jobs()

            return report_data
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def get_visualizations(self, job_id: str) -> Optional[dict]:
        """Get visualization data for training curves"""
        with self.lock:
            job = self.jobs.get(job_id)
        
        if job is None:
            return None
        
        return {
            'metrics': job.metrics,
            'epochs': list(range(1, len(job.metrics['train_loss']) + 1))
        }
    
    def get_model_path(self, job_id: str) -> Optional[str]:
        """Get path to trained model"""
        with self.lock:
            job = self.jobs.get(job_id)
        
        if job is None or job.model_path is None:
            return None
        
        return job.model_path
    
    def get_report_path(self, job_id: str) -> Optional[str]:
        """Get path to evaluation report"""
        with self.lock:
            job = self.jobs.get(job_id)

        report_path = job.report_path if job is not None else self._build_report_path(job_id)
        
        if os.path.exists(report_path):
            return report_path
        
        return None

    def _build_report_path(self, job_id: str) -> str:
        return os.path.join(
            get_project_root(),
            'results',
            'reports',
            f'report_{job_id}.json'
        )
