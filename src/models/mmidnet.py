"""
MMIDNet Model for Point Cloud Classification.

MMIDNet (Millimeter-wave Identification Network) is designed for human identification
using point cloud data. This implementation is adapted for the FAUST dataset:
- Non-temporal data: Bi-LSTM and TimeDistributed removed
- 3 channels (x, y, z) only: no velocity or SNR

Architecture (from Human_identifiaction.pdf Section 6.4):
    Input → Transform Block (T-Net) → Residual CNN Block → Global Max Pooling → Dense Block → Output

Key components:
1. Transform Block: T-Net predicts 3×3 matrix, applies to x,y,z for transformation invariance
2. Residual CNN Block: Three residual groups (128, 256, 512 channels) for feature extraction
3. Global Max Pooling: Permutation invariance
4. Dense Block: 512 → 256 → 64 → num_classes with dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .pointnet_tiny import TNet


class TransformBlock(nn.Module):
    """
    Transformation Block for spatial alignment of point clouds.
    
    Applies T-Net to predict a 3×3 affine transformation matrix and applies it
    to the x, y, z coordinates. For FAUST we only have 3 channels, so no split/concatenation.
    
    Input: (B, N, 3) or (B, 3, N)
    Output: (B, 3, N) - transformed coordinates ready for Conv1D
    """

    def __init__(self):
        super(TransformBlock, self).__init__()
        self.tnet = TNet(k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to point cloud coordinates.
        
        Args:
            x: input tensor, shape (B, 3, N) for Conv1D format
            
        Returns:
            transformed: shape (B, 3, N)
        """
        T = self.tnet(x)  # (B, 3, 3)
        # Apply: X' = X^T @ T, then transpose back
        # x: (B, 3, N) -> transpose to (B, N, 3)
        x_t = x.transpose(1, 2)  # (B, N, 3)
        x_transformed = torch.bmm(x_t, T)  # (B, N, 3) @ (B, 3, 3) = (B, N, 3)
        return x_transformed.transpose(1, 2)  # (B, 3, N)


class ResidualCNNBlock(nn.Module):
    """
    Residual CNN Block with three residual groups.
    
    Each group has two Conv1D layers (kernel_size=1) with BatchNorm and ReLU.
    Skip connection: output = ReLU(main_path(x) + skip_path(x))
    
    Channel progression: 3 → 128 → 256 → 512
    """

    def __init__(self, dropout: float = 0.3):
        super(ResidualCNNBlock, self).__init__()
        self.dropout = dropout

        # Group 1: 3 → 128
        self.res1 = self._make_residual_group(3, 128)
        self.drop1 = nn.Dropout(dropout)

        # Group 2: 128 → 256
        self.res2 = self._make_residual_group(128, 256)
        self.drop2 = nn.Dropout(dropout)

        # Group 3: 256 → 512
        self.res3 = self._make_residual_group(256, 512)

    def _make_residual_group(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create residual block: two Conv1D layers with skip connection.
        
        main: Conv1D(in, out) -> BN -> ReLU -> Conv1D(out, out) -> BN
        skip: Identity if in==out, else Conv1D(in, out) -> BN
        """
        main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels == out_channels:
            skip_path = nn.Identity()
        else:
            skip_path = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

        class ResGroup(nn.Module):
            def __init__(self, main, skip):
                super().__init__()
                self.main = main
                self.skip = skip

            def forward(self, x):
                return F.relu(self.main(x) + self.skip(x))

        return ResGroup(main_path, skip_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through three residual groups."""
        x = self.res1(x)  # (B, 128, N)
        x = self.drop1(x)
        x = self.res2(x)  # (B, 256, N)
        x = self.drop2(x)
        x = self.res3(x)  # (B, 512, N)
        return x


class MMIDNet(nn.Module):
    """
    MMIDNet for FAUST point cloud classification.
    
    Full pipeline:
        Input (B, N, 3) → Transform Block → Residual CNN → Global Max Pool → Dense Block → (B, K)
    
    Adapted from original MMIDNet (mmWave radar, temporal data):
    - Removed Bi-LSTM (no temporal dimension)
    - Removed TimeDistributed
    - Input: 3 channels only (x, y, z)
    """

    def __init__(
        self,
        num_points: int = 200,
        num_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        """
        Initialize MMIDNet.
        
        Args:
            num_points: number of points per cloud (200)
            num_channels: input channels (3 for x, y, z)
            num_classes: output classes (10 for FAUST)
            dropout: dropout rate for dense block
            
        Example:
            >>> model = MMIDNet(num_classes=10)
            >>> x = torch.randn(32, 200, 3)
            >>> out = model(x)
            >>> out.shape
            torch.Size([32, 10])
        """
        super(MMIDNet, self).__init__()
        self.num_points = num_points
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.transform_block = TransformBlock()
        self.residual_cnn = ResidualCNNBlock(dropout=dropout)

        # Dense Block: 512 → 256 → 64 → num_classes
        self.dense_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MMIDNet.
        
        Args:
            x: input point cloud, shape (B, N, C)
               where B=batch_size, N=num_points, C=3
               
        Returns:
            logits: class scores, shape (B, num_classes)
        """
        # Transpose to (B, C, N) for Conv1D
        x = x.transpose(1, 2)  # (B, 3, N)

        # Transform Block
        x = self.transform_block(x)  # (B, 3, N)

        # Residual CNN Block
        x = self.residual_cnn(x)  # (B, 512, N)

        # Global Max Pooling: (B, 512, N) → (B, 512)
        x = torch.max(x, dim=2)[0]

        # Dense Block
        logits = self.dense_block(x)  # (B, num_classes)

        return logits

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
