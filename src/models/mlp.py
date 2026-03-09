"""
MLP (Multi-Layer Perceptron) Baseline Model for Point Cloud Classification.

This is the simplest baseline that:
- Flattens the point cloud into a 1D vector
- Passes through fully connected layers
- Outputs class probabilities

Limitations:
- Order-dependent: changing point order changes the flattened vector
- No geometric understanding: treats 3D coordinates as arbitrary numbers
- High parameter count due to flattening

Purpose: Establish baseline to show that spatial structure matters
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline for point cloud classification.
    
    Architecture:
        Input (B, N, 3) → Flatten → Dense → BN → ReLU → Dropout
                                  → Dense → BN → ReLU → Dropout
                                  → Dense → Softmax
    
    where B = batch size, N = number of points (e.g., 200)
    
    This model serves as a sanity check:
    - If it works reasonably well: data contains identity information
    - If it performs poorly: need better feature extraction (CNN/PointNet)
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 dropout: float = 0.3):
        """
        Initialize MLP baseline model.
        
        Args:
            num_points: number of points per point cloud (default 200)
            num_channels: number of channels per point, typically 3 for (x,y,z)
            num_classes: number of output classes (10 subjects in FAUST)
            hidden_dims: tuple of hidden layer dimensions
            dropout: dropout rate for regularization (0.3 = drop 30% of neurons)
            
        Example:
            >>> model = MLPBaseline(num_points=200, num_classes=10)
            >>> x = torch.randn(32, 200, 3)  # Batch of 32 point clouds
            >>> out = model(x)
            >>> out.shape
            torch.Size([32, 10])  # 32 samples, 10 class scores
        """
        super(MLPBaseline, self).__init__()
        
        # Store parameters
        self.num_points = num_points  # 200
        self.num_channels = num_channels  # 3
        self.num_classes = num_classes  # 10
        
        # Calculate input dimension after flattening
        # Shape (B, 200, 3) → Flatten → (B, 600)
        input_dim = num_points * num_channels  # 200 * 3 = 600
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim  # 600
        
        # Add hidden layers with BatchNorm, ReLU, and Dropout
        for hidden_dim in hidden_dims:
            # Linear layer: y = xW^T + b
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch Normalization: normalize activations for stable training
            # Reduces internal covariate shift
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU activation: f(x) = max(0, x)
            # Introduces non-linearity
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout: randomly zero out neurons during training
            # Prevents overfitting by forcing redundancy
            layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim  # Update for next layer
        
        # Combine all hidden layers into a sequential module
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification layer: map to num_classes
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: input point cloud tensor, shape (B, N, C)
               where B=batch_size, N=num_points, C=num_channels
               
        Returns:
            out: class logits, shape (B, num_classes)
                 Apply softmax for probabilities: softmax(out, dim=1)
                 
        Example:
            >>> model = MLPBaseline(num_points=200, num_classes=10)
            >>> x = torch.randn(32, 200, 3)
            >>> logits = model(x)
            >>> probs = torch.softmax(logits, dim=1)
            >>> predicted_class = torch.argmax(probs, dim=1)
        """
        batch_size = x.size(0)  # B
        
        # Step 1: Flatten point cloud from (B, N, C) to (B, N*C)
        # This loses spatial structure but allows simple MLP processing
        x_flat = x.view(batch_size, -1)  # (B, 200, 3) → (B, 600)
        
        # Step 2: Pass through feature extraction layers
        features = self.feature_extractor(x_flat)  # (B, 600) → (B, 128)
        
        # Step 3: Classification head
        logits = self.classifier(features)  # (B, 128) → (B, 10)
        
        return logits
    
    def get_num_params(self) -> int:
        """
        Count total number of trainable parameters.
        
        Useful for comparing model complexity across architectures.
        
        Returns:
            num_params: total trainable parameters
            
        Example:
            >>> model = MLPBaseline()
            >>> print(f"Parameters: {model.get_num_params():,}")
            Parameters: 1,234,567
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepMLPBaseline(nn.Module):
    """
    Deeper MLP variant with more layers.
    
    This model has more capacity but also higher risk of overfitting.
    Useful for comparing shallow vs deep MLP performance.
    
    Architecture:
        Input → Dense(512) → BN → ReLU → Dropout
              → Dense(256) → BN → ReLU → Dropout
              → Dense(128) → BN → ReLU → Dropout
              → Dense(64) → BN → ReLU → Dropout
              → Dense(num_classes)
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 dropout: float = 0.3):
        """
        Initialize deep MLP model.
        
        Args:
            num_points: number of points per point cloud
            num_channels: number of channels per point
            num_classes: number of output classes
            dropout: dropout rate
        """
        super(DeepMLPBaseline, self).__init__()
        
        input_dim = num_points * num_channels  # 600
        
        # Define architecture with progressively smaller dimensions
        self.network = nn.Sequential(
            # Layer 1: 600 → 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # Layer 4: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # Output layer: 64 → num_classes
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten
        logits = self.network(x_flat)
        return logits
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Why MLP is expected to underperform:
#
# 1. **Order Dependency**: 
#    Point clouds are unordered sets. Swapping two points shouldn't change
#    the object identity, but MLP treats [p1, p2, p3] differently from [p2, p1, p3].
#
# 2. **No Spatial Understanding**:
#    MLP processes coordinates as arbitrary numbers. It doesn't understand
#    that these are 3D positions with geometric relationships.
#
# 3. **High Dimensionality**:
#    Flattening to 600-dim vector creates many parameters and makes training harder.
#
# 4. **No Local Feature Extraction**:
#    Cannot learn local patterns like "shoulder shape" or "limb proportions"
#    that are crucial for human identification.
#
# Expected performance: ~20-40% accuracy (from paper)
# This will be surpassed by CNN and PointNet models.

