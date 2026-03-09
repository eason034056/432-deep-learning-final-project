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
from .mmidnet import MMIDNet
from .autoencoder import MLPAutoencoder, PointNetAutoencoder, chamfer_distance

__all__ = [
    'MLPBaseline',
    'DeepMLPBaseline',
    'CNN1DModel',
    'ResidualCNN1D',
    'TinyPointNet',
    'TNet',
    'PointNetBackbone',
    'MMIDNet',
    'MLPAutoencoder',
    'PointNetAutoencoder',
    'chamfer_distance'
]

