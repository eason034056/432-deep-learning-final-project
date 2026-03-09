"""
1D-CNN Model for Point Cloud Classification.

This model applies 1D convolutions across the point dimension to capture
local spatial patterns. It's inspired by the residual CNN component in MMIDNet.

Key improvements over MLP:
- Captures local neighborhoods through convolutions
- Reduces parameters via weight sharing
- More robust to point ordering (when combined with sorting)

Architecture flow:
    Input (B, N, 3) → Sort by z → Transpose → Conv1D layers → GlobalMaxPool → Dense
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CNN1DModel(nn.Module):
    """
    1D-CNN model for point cloud classification.
    
    This model treats the point cloud as a 1D sequence after sorting.
    It applies Conv1D filters to capture local patterns.
    
    Architecture:
        Input (B, N, 3) → Sort → Conv1D(64) → BN → ReLU
                               → Conv1D(128) → BN → ReLU
                               → GlobalMaxPooling
                               → Dense(128) → Dropout → Dense(num_classes)
    
    Sorting strategy:
    - Sort points by z-coordinate to establish a consistent order
    - This makes the model more robust to permutation
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 conv_channels: Tuple[int, ...] = (64, 128, 256),
                 fc_dims: Tuple[int, ...] = (128,),
                 dropout: float = 0.3,
                 kernel_size: int = 1):
        """
        Initialize 1D-CNN model.
        
        Args:
            num_points: number of points per point cloud (200)
            num_channels: input channels (3 for x,y,z)
            num_classes: output classes (10 subjects)
            conv_channels: tuple of output channels for each conv layer
            fc_dims: tuple of dimensions for fully connected layers
            dropout: dropout rate for regularization
            kernel_size: size of convolution kernel (1 for pointwise, 3 for local)
            
        Explanation of parameters:
        - conv_channels=(64, 128, 256): 
            * First conv: 3 → 64 channels (learn 64 different features)
            * Second conv: 64 → 128 channels (combine features)
            * Third conv: 128 → 256 channels (high-level features)
        
        - kernel_size=1: 
            * Pointwise convolution (like in PointNet)
            * Processes each point independently
            * Shared MLP across all points
        
        Example:
            >>> model = CNN1DModel(num_points=200, num_classes=10)
            >>> x = torch.randn(32, 200, 3)
            >>> out = model(x)
            >>> out.shape
            torch.Size([32, 10])
        """
        super(CNN1DModel, self).__init__()
        
        self.num_points = num_points
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        
        # Build convolutional layers
        # Conv1D in PyTorch expects input: (B, C_in, L)
        # where B=batch, C_in=channels, L=length (number of points)
        
        conv_layers = []
        in_channels = num_channels  # Start with 3 (x, y, z)
        
        for out_channels in conv_channels:
            # Conv1d layer: applies filters across the point dimension
            # If kernel_size=1: pointwise convolution (no neighbor interaction)
            # If kernel_size=3: local neighborhood convolution
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Keep same length
                )
            )
            
            # BatchNorm1d: normalize across channel dimension
            # Helps training stability
            conv_layers.append(nn.BatchNorm1d(out_channels))
            
            # ReLU: non-linear activation
            conv_layers.append(nn.ReLU(inplace=True))
            
            in_channels = out_channels  # Update for next layer
        
        # Combine all conv layers into a sequential module
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global Max Pooling: reduce (B, C, N) → (B, C)
        # Takes maximum value across all points for each channel
        # This achieves permutation invariance: order doesn't matter for max
        # (Though we sort first for better local patterns)
        
        # Fully connected layers for classification
        fc_layers = []
        in_dim = conv_channels[-1]  # Last conv output channels (256)
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(p=dropout))
            in_dim = fc_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Final classification layer
        self.classifier = nn.Linear(in_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: input point cloud, shape (B, N, 3)
               where B=batch_size, N=num_points (200), C=channels (3)
               
        Returns:
            logits: class scores, shape (B, num_classes)
            
        Processing steps:
        1. Sort points by z-coordinate for consistent ordering
        2. Transpose to (B, C, N) for Conv1D
        3. Apply convolutional layers
        4. Global max pooling to get (B, C) features
        5. Fully connected layers for classification
        """
        batch_size = x.size(0)  # B
        
        # Step 1: Sort points by z-coordinate
        # This creates a canonical ordering that's consistent across samples
        # Sorting index based on z-values (x[:, :, 2] extracts z-coordinates)
        _, sorted_indices = torch.sort(x[:, :, 2], dim=1)  # (B, N)
        
        # Use sorted indices to reorder points
        # expand to (B, N, 3) for gathering
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        x_sorted = torch.gather(x, dim=1, index=sorted_indices_expanded)  # (B, N, 3)
        
        # Step 2: Transpose to (B, C, N) format for Conv1D
        # Conv1D expects: (batch, channels, sequence_length)
        # We have: (batch, points, channels)
        # So transpose dimensions 1 and 2
        x_transposed = x_sorted.transpose(1, 2)  # (B, 3, N)
        
        # Step 3: Apply convolutional layers
        # Each conv layer learns spatial patterns across points
        features = self.conv_layers(x_transposed)  # (B, 3, N) → (B, 256, N)
        
        # Step 4: Global max pooling
        # Take maximum value for each channel across all points
        # This makes the representation permutation-invariant
        pooled, _ = torch.max(features, dim=2)  # (B, 256, N) → (B, 256)
        
        # Step 5: Fully connected layers
        fc_features = self.fc_layers(pooled)  # (B, 256) → (B, 128)
        
        # Step 6: Classification
        logits = self.classifier(fc_features)  # (B, 128) → (B, 10)
        
        return logits
    
    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualCNN1D(nn.Module):
    """
    1D-CNN with residual connections (inspired by MMIDNet).
    
    Residual connections help training deeper networks by allowing
    gradients to flow directly through skip connections.
    
    Architecture:
        Input → ResBlock1 → ResBlock2 → ResBlock3 → GlobalMaxPool → Dense
        
    where each ResBlock has:
        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 64,
                 dropout: float = 0.3):
        """
        Initialize residual CNN model.
        
        Args:
            num_points: number of points
            num_channels: input channels (3)
            num_classes: output classes (10)
            base_channels: base number of channels for residual blocks
            dropout: dropout rate
        """
        super(ResidualCNN1D, self).__init__()
        
        # Initial projection to base_channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(num_channels, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with increasing channels
        self.res_block1 = self._make_residual_block(base_channels, base_channels)
        self.res_block2 = self._make_residual_block(base_channels, base_channels * 2)
        self.res_block3 = self._make_residual_block(base_channels * 2, base_channels * 4)
        
        # Classification head
        final_channels = base_channels * 4  # 64 * 4 = 256
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a residual block with skip connection.
        
        If in_channels != out_channels, use 1x1 conv for dimension matching.
        
        Args:
            in_channels: input channel dimension
            out_channels: output channel dimension
            
        Returns:
            residual_block: nn.Module implementing residual connection
        """
        # Main path: two convolutions
        main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # Skip path: identity if same channels, else 1x1 conv
        if in_channels == out_channels:
            skip_path = nn.Identity()
        else:
            skip_path = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        
        # Wrap in a module that adds skip connection
        class ResBlock(nn.Module):
            def __init__(self, main, skip):
                super().__init__()
                self.main = main
                self.skip = skip
                
            def forward(self, x):
                # F(x) + x (or F(x) + projection(x))
                return F.relu(self.main(x) + self.skip(x))
        
        return ResBlock(main_path, skip_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        batch_size = x.size(0)
        
        # Sort by z-coordinate
        _, sorted_indices = torch.sort(x[:, :, 2], dim=1)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        x_sorted = torch.gather(x, dim=1, index=sorted_indices_expanded)
        
        # Transpose to (B, C, N)
        x_transposed = x_sorted.transpose(1, 2)  # (B, 3, N)
        
        # Initial convolution
        x = self.input_conv(x_transposed)  # (B, 64, N)
        
        # Residual blocks
        x = self.res_block1(x)  # (B, 64, N)
        x = self.res_block2(x)  # (B, 128, N)
        x = self.res_block3(x)  # (B, 256, N)
        
        # Global max pooling
        x, _ = torch.max(x, dim=2)  # (B, 256)
        
        # Classification
        logits = self.classifier(x)  # (B, 10)
        
        return logits
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Why CNN1D performs better than MLP:
#
# 1. **Local Pattern Recognition**:
#    Convolutions capture local neighborhoods, e.g., "shoulder region" or "torso shape"
#
# 2. **Parameter Efficiency**:
#    Weight sharing across points reduces parameters compared to MLP
#
# 3. **Partial Permutation Invariance**:
#    Sorting + GlobalMaxPooling provides some ordering robustness
#
# 4. **Hierarchical Features**:
#    Stacked convolutions learn from low-level (point pairs) to high-level (body parts)
#
# Expected performance: ~60-70% accuracy (from paper)
# Still limited by sorting dependency, which PointNet overcomes.

