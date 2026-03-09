"""
Tiny PointNet Model for Point Cloud Classification.

This is a simplified version of PointNet (https://arxiv.org/abs/1612.00593)
tailored for the human identification task.

Key innovations of PointNet:
1. Permutation Invariance: order of points doesn't matter
2. Transformation Invariance: learns to align point clouds via T-Net
3. Symmetric Function: max pooling aggregates features

Architecture:
    Input → [T-Net] → Shared MLP → Global Max Pooling → MLP → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TNet(nn.Module):
    """
    Transformation Network (T-Net) for learning spatial alignment.
    
    T-Net predicts a transformation matrix to canonically align the input.
    This makes the model invariant to rotation and translation.
    
    For input transformation:
    - Predicts 3×3 affine transformation matrix
    - Applies to (x,y,z) coordinates
    
    Architecture:
        Input (B, 3, N) → Conv1D layers → GlobalMax → FC → 3×3 matrix
    
    The predicted matrix is applied: X' = X @ T
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize T-Net.
        
        Args:
            k: dimension of transformation (3 for xyz, 64 for features)
               k=3: transform input coordinates
               k=64: transform feature space (optional, more advanced)
        
        Example:
            >>> tnet = TNet(k=3)
            >>> x = torch.randn(32, 3, 200)  # (B, 3, N)
            >>> T = tnet(x)  # (B, 3, 3)
            >>> x_transformed = torch.bmm(x.transpose(1,2), T)  # (B, N, 3)
        """
        super(TNet, self).__init__()
        
        self.k = k  # Dimension to transform (3 for xyz)
        
        # Feature extraction with Conv1D layers
        # Process each point independently (kernel_size=1)
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Fully connected layers to predict transformation matrix
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)  # Output k×k matrix
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict transformation matrix.
        
        Args:
            x: input tensor, shape (B, k, N)
               For coordinate transform: (B, 3, N)
               
        Returns:
            T: transformation matrix, shape (B, k, k)
               For coordinates: (B, 3, 3)
        """
        batch_size = x.size(0)
        
        # Feature extraction: (B, k, N) → (B, 1024, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)
        
        # Global max pooling: (B, 1024, N) → (B, 1024)
        # Aggregates information from all points
        x = torch.max(x, dim=2)[0]  # [0] gets max values (not indices)
        
        # Predict transformation via FC layers
        x = F.relu(self.bn_fc1(self.fc1(x)))  # (B, 512)
        x = F.relu(self.bn_fc2(self.fc2(x)))  # (B, 256)
        x = self.fc3(x)  # (B, k*k) = (B, 9) for k=3
        
        # Reshape to matrix: (B, k*k) → (B, k, k)
        T = x.view(batch_size, self.k, self.k)  # (B, 3, 3)
        
        # Add identity matrix to make optimization easier
        # This means we're predicting a residual transformation
        # T_final = I + T_predicted
        identity = torch.eye(self.k, device=x.device).unsqueeze(0)  # (1, k, k)
        identity = identity.repeat(batch_size, 1, 1)  # (B, k, k)
        T = T + identity
        
        return T


class PointNetBackbone(nn.Module):
    """
    PointNet feature extraction backbone.
    
    This module extracts permutation-invariant features from point clouds.
    
    Architecture:
        Input (B, 3, N) → [T-Net] → Shared MLP (64, 64, 128, 1024)
                                  → GlobalMaxPooling → Global Features (B, 1024)
    
    The output is a fixed-size feature vector for each point cloud,
    regardless of input point count.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 use_tnet: bool = True,
                 channel_dims: Tuple[int, ...] = (64, 128, 1024)):
        """
        Initialize PointNet backbone.
        
        Args:
            input_channels: number of input channels (3 for xyz)
            use_tnet: whether to use T-Net for transformation
            channel_dims: tuple of channel dimensions for shared MLP
            
        Example:
            >>> backbone = PointNetBackbone(input_channels=3, use_tnet=True)
            >>> x = torch.randn(32, 3, 200)
            >>> features = backbone(x)  # (32, 1024)
        """
        super(PointNetBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.use_tnet = use_tnet
        
        # T-Net for input transformation (optional)
        if use_tnet:
            self.tnet = TNet(k=3)
        
        # Shared MLP implemented as Conv1D with kernel_size=1
        # This applies the same MLP to each point independently
        conv_layers = []
        bn_layers = []
        
        in_channels = input_channels  # 3
        for out_channels in channel_dims:
            # Conv1d with kernel_size=1: pointwise convolution
            # Equivalent to applying same MLP to each point
            conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
            bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        # Store as ModuleList for proper parameter registration
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from point cloud.
        
        Args:
            x: input point cloud, shape (B, C, N)
               where B=batch, C=channels (3), N=num_points
               
        Returns:
            global_features: shape (B, channel_dims[-1])
                           e.g., (B, 1024)
        """
        batch_size, num_channels, num_points = x.size()
        
        # Step 1: Apply T-Net transformation if enabled
        if self.use_tnet:
            # Predict transformation matrix
            T = self.tnet(x)  # (B, 3, 3)
            
            # Apply transformation: X' = X^T @ T
            # Transpose x: (B, C, N) → (B, N, C)
            x = x.transpose(1, 2)  # (B, N, 3)
            
            # Batch matrix multiplication
            x = torch.bmm(x, T)  # (B, N, 3) @ (B, 3, 3) = (B, N, 3)
            
            # Transpose back: (B, N, C) → (B, C, N)
            x = x.transpose(1, 2)  # (B, 3, N)
        
        # Step 2: Shared MLP (Conv1D layers)
        # Process each point with the same network
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)  # Apply convolution
            x = bn(x)    # Batch normalization
            x = F.relu(x)  # Non-linearity
        
        # x is now (B, 1024, N) with rich features for each point
        
        # Step 3: Global Max Pooling
        # Take maximum across all points for each feature channel
        # This achieves permutation invariance: max([a,b,c]) = max([b,c,a])
        global_features = torch.max(x, dim=2)[0]  # (B, 1024, N) → (B, 1024)
        
        return global_features


class TinyPointNet(nn.Module):
    """
    Complete Tiny PointNet model for classification.
    
    This is a simplified PointNet suitable for the FAUST human identification task.
    
    Full pipeline:
        Point Cloud (B, N, 3) → Transpose → (B, 3, N)
                              → PointNet Backbone → Global Features (B, 1024)
                              → MLP Classifier → Logits (B, num_classes)
    
    Key properties:
    1. Permutation Invariant: order of points doesn't affect output
    2. Transformation Invariant: robust to rotation/translation (via T-Net)
    3. Efficient: only processes what's needed via max pooling
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 use_tnet: bool = True,
                 channel_dims: Tuple[int, ...] = (64, 128, 1024),
                 fc_dims: Tuple[int, ...] = (512, 256),
                 dropout: float = 0.3):
        """
        Initialize Tiny PointNet.
        
        Args:
            num_points: number of points per cloud (200)
            num_channels: input channels (3 for xyz)
            num_classes: output classes (10 subjects)
            use_tnet: whether to use T-Net transformation
            channel_dims: dimensions for backbone Conv1D layers
            fc_dims: dimensions for classifier FC layers
            dropout: dropout rate for regularization
            
        Example:
            >>> model = TinyPointNet(num_classes=10, use_tnet=True)
            >>> x = torch.randn(32, 200, 3)
            >>> out = model(x)
            >>> out.shape
            torch.Size([32, 10])
        """
        super(TinyPointNet, self).__init__()
        
        self.num_points = num_points
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Feature extraction backbone
        self.backbone = PointNetBackbone(
            input_channels=num_channels,
            use_tnet=use_tnet,
            channel_dims=channel_dims
        )
        
        # Classification head
        # Input: global features from backbone (channel_dims[-1])
        # Output: class logits (num_classes)
        classifier_layers = []
        in_dim = channel_dims[-1]  # 1024
        
        for fc_dim in fc_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ])
            in_dim = fc_dim
        
        # Final output layer
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TinyPointNet.
        
        Args:
            x: input point cloud, shape (B, N, C)
               where B=batch_size, N=num_points, C=num_channels
               
        Returns:
            logits: class scores, shape (B, num_classes)
            
        Example:
            >>> model = TinyPointNet(num_classes=10)
            >>> x = torch.randn(32, 200, 3)
            >>> logits = model(x)
            >>> probs = F.softmax(logits, dim=1)
            >>> predictions = torch.argmax(logits, dim=1)
        """
        batch_size = x.size(0)
        
        # Transpose to (B, C, N) format for Conv1D
        # Input: (B, N, C) = (B, 200, 3)
        # Output: (B, C, N) = (B, 3, 200)
        x = x.transpose(1, 2)
        
        # Extract global features via backbone
        # (B, 3, N) → (B, 1024)
        global_features = self.backbone(x)
        
        # Classify based on global features
        # (B, 1024) → (B, num_classes)
        logits = self.classifier(global_features)
        
        return logits
    
    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features without classification.
        
        Useful for visualization, feature analysis, or transfer learning.
        
        Args:
            x: input point cloud, shape (B, N, C)
            
        Returns:
            features: global features, shape (B, 1024)
        """
        x = x.transpose(1, 2)  # (B, N, C) → (B, C, N)
        features = self.backbone(x)  # (B, 1024)
        return features


# Why PointNet performs best:
#
# 1. **Permutation Invariance**:
#    Max pooling ensures f({p1, p2, p3}) = f({p3, p1, p2})
#    Order doesn't matter, matching the true nature of point clouds
#
# 2. **Transformation Invariance**:
#    T-Net learns to canonically align input before processing
#    Robust to rotation, translation, and pose variations
#
# 3. **Efficient Feature Learning**:
#    Shared MLP learns features for each point independently
#    Then aggregates globally via max pooling
#
# 4. **Low Parameter Count**:
#    Compared to MLP, PointNet has fewer parameters due to:
#    - Shared weights across points (Conv1D with kernel=1)
#    - No need for flattening high-dimensional tensors
#
# 5. **Geometric Understanding**:
#    Captures spatial relationships better than MLP
#    More robust than CNN1D's sorting-based approach
#
# Expected performance: ~75-85% accuracy (from roadmap)
# Best among the three models for this task.

