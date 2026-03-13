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
