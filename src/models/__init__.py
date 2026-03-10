"""
Models package for point cloud classification.

Available models:
- MLPBaseline: Simple MLP baseline
- CNN1DModel: 1D convolution model
- TinyPointNet: PointNet-based model (best performance)
- MLPAutoencoder, PointNetAutoencoder: Compression autoencoders
"""

from .mlp import MLPBaseline, DeepMLPBaseline
from .cnn1d import CNN1DModel, ResidualCNN1D
from .pointnet_tiny import TinyPointNet, TNet, PointNetBackbone
from .autoencoder import (
    MLPAutoencoder,
    PointNetAutoencoder,
    chamfer_distance,
    create_autoencoder_from_config,
    get_autoencoder_config,
)

__all__ = [
    'MLPBaseline',
    'DeepMLPBaseline',
    'CNN1DModel',
    'ResidualCNN1D',
    'TinyPointNet',
    'TNet',
    'PointNetBackbone',
    'MLPAutoencoder',
    'PointNetAutoencoder',
    'chamfer_distance',
    'create_autoencoder_from_config',
    'get_autoencoder_config',
]

