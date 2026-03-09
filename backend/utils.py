"""
Utility functions for the backend server

Includes:
- File handling utilities
- Configuration management
- Path utilities
- Validation functions
"""

import os
import yaml
from pathlib import Path
from typing import Optional

# Allowed file extensions for mesh uploads
ALLOWED_EXTENSIONS = {'.ply', '.obj', '.stl', '.off'}


def get_project_root() -> str:
    """
    Get the project root directory
    
    Returns:
        Absolute path to project root
    """
    # Backend is in project_root/backend, so go up one level
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def allowed_file(filename: str) -> bool:
    """
    Check if filename has an allowed extension
    
    Args:
        filename: Name of the file
        
    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_config_path() -> str:
    """Get path to config.yaml"""
    return os.path.join(get_project_root(), 'config.yaml')


def load_config() -> dict:
    """
    Load configuration from config.yaml
    
    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict) -> None:
    """
    Save configuration to config.yaml
    
    Args:
        config: Configuration dictionary to save
    """
    config_path = get_config_path()
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def format_bytes(bytes: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_model_checkpoint_path(model_type: str, job_id: Optional[str] = None) -> str:
    """
    Get path for model checkpoint
    
    Args:
        model_type: Model type ('mlp', 'cnn1d', 'pointnet')
        job_id: Optional job ID for unique naming
        
    Returns:
        Path to checkpoint file
    """
    results_dir = os.path.join(get_project_root(), 'results', 'checkpoints', model_type)
    ensure_dir(results_dir)
    
    if job_id:
        filename = f'model_{job_id}.pth'
    else:
        filename = 'model_best.pth'
    
    return os.path.join(results_dir, filename)


def validate_config(config: dict) -> tuple[bool, Optional[str]]:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
        
    Returns:
        (is_valid, error_message)
    """
    required_keys = ['data', 'training', 'model']
    
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"
    
    # Validate data config
    if 'num_points' not in config['data']:
        return False, "Missing data.num_points"
    
    if config['data']['num_points'] <= 0:
        return False, "data.num_points must be positive"
    
    # Validate training config
    training_keys = ['batch_size', 'num_epochs', 'learning_rate']
    for key in training_keys:
        if key not in config['training']:
            return False, f"Missing training.{key}"
    
    # Validate model config
    if 'type' not in config['model']:
        return False, "Missing model.type"
    
    if config['model']['type'] not in ['mlp', 'cnn1d', 'pointnet']:
        return False, f"Invalid model type: {config['model']['type']}"
    
    return True, None
