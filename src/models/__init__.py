"""
Models package for point cloud classification.

Available models:
- MLPBaseline: Simple MLP baseline
- CNN1DModel: 1D convolution model
- TinyPointNet: PointNet-based model (best performance)
- PointNet2SSG: PointNet++ with hierarchical local features
"""

from .mlp import MLPBaseline, DeepMLPBaseline
from .cnn1d import CNN1DModel, ResidualCNN1D
from .pointnet_tiny import TinyPointNet, TNet, PointNetBackbone
from .pointnet2 import PointNet2SSG
from .mmidnet import MMIDNet
from .autoencoder import MLPAutoencoder, PointNet2Autoencoder, chamfer_distance

__all__ = [
    'MLPBaseline',
    'DeepMLPBaseline',
    'CNN1DModel',
    'ResidualCNN1D',
    'TinyPointNet',
    'TNet',
    'PointNetBackbone',
    'PointNet2SSG',
    'MMIDNet',
    'MLPAutoencoder',
    'PointNet2Autoencoder',
    'chamfer_distance'
]

