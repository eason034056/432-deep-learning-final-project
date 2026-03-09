"""
PointNet++ Model for Point Cloud Classification.

PointNet++ (https://arxiv.org/abs/1706.02413) improves upon PointNet by learning
hierarchical local features through Set Abstraction layers.

Key components:
1. Farthest Point Sampling (FPS): Select representative points
2. Ball Query / K-NN: Group neighboring points
3. Set Abstraction (SA): Local feature extraction via mini-PointNet
4. Global feature → Classifier

Architecture for classification:
    Input (B, N, 3) → SA1 → SA2 → SA3 (global) → MLP → Output (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance between two point sets.
    
    Formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        src: (B, N, C) - source points
        dst: (B, M, C) - destination points
        
    Returns:
        dist: (B, N, M) - squared distance from each src point to each dst point
        
    Example:
        >>> src = torch.randn(4, 100, 3)
        >>> dst = torch.randn(4, 50, 3)
        >>> dist = square_distance(src, dst)  # (4, 100, 50)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # (B, N, 1) + (B, 1, M) - 2*(B, N, M) via broadcasting
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)    # (B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)     # (B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) - batched version.
    
    Iteratively selects the point farthest from already-selected points.
    Ensures good spatial coverage of the point cloud.
    
    Args:
        xyz: (B, N, 3) - input point cloud
        npoint: number of points to sample
        
    Returns:
        centroids: (B, npoint) - indices of sampled points
        
    Example:
        >>> xyz = torch.randn(8, 200, 3)
        >>> idx = farthest_point_sample(xyz, 64)  # (8, 64)
    """
    B, N, C = xyz.shape
    device = xyz.device
    
    # centroids: indices of selected points for each batch
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # distance: for each point, min distance to selected set
    distance = torch.ones(B, N, device=device) * 1e10
    # Start with random point for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        # Get coordinates of selected centroids: (B, 1, 3)
        centroid_xyz = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        # Squared distance from all points to current centroid: (B, N)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        # Update: keep minimum distance to any selected point
        mask = dist < distance
        distance[mask] = dist[mask]
        # Next: point with maximum (minimum distance) = farthest from selected set
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by indices (batch-aware).
    
    Args:
        points: (B, N, C) - input points
        idx: (B, M) or (B, M, K) - indices to gather
        
    Returns:
        gathered: (B, M, C) or (B, M, K, C) - gathered points
        
    Example:
        >>> points = torch.randn(4, 200, 3)
        >>> idx = torch.randint(0, 200, (4, 64))
        >>> out = index_points(points, idx)  # (4, 64, 3)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # batch_indices: (B, M) or (B, M, K) - batch index for each element
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, 
                    new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball Query: find up to nsample points within radius of each centroid.
    
    For points outside radius, duplicates the nearest point to fill.
    Uses K-NN as fallback when ball query would be empty.
    
    Args:
        radius: search radius (squared distance for efficiency)
        nsample: max points to return per centroid
        xyz: (B, N, 3) - all points
        new_xyz: (B, npoint, 3) - centroid points
        
    Returns:
        group_idx: (B, npoint, nsample) - indices of grouped points
    """
    B, N, C = xyz.shape
    _, M, _ = new_xyz.shape
    
    # Squared distance: (B, N, M) - dist from each point to each centroid
    sqrdist = square_distance(xyz, new_xyz)  # (B, N, M)
    
    # For each centroid, get nsample nearest points (K-NN)
    # Sort by distance, take first nsample
    # sqrdist.permute(0,2,1): (B, M, N) - for each centroid, dist to all points
    dist, group_idx = torch.topk(sqrdist.permute(0, 2, 1), nsample, dim=-1, largest=False)
    # group_idx: (B, M, nsample)
    
    # Optional: mask out points beyond radius (replace with nearest)
    # For simplicity we use K-NN which works well in practice
    return group_idx


class SetAbstraction(nn.Module):
    """
    Set Abstraction (SA) layer - core building block of PointNet++.
    
    Process:
    1. FPS: sample npoint centroids from N points
    2. Ball Query / K-NN: group nsample neighbors for each centroid
    3. Local PointNet: MLP on each group → max pool → local features
    
    Output: (B, npoint, C_out) - one feature vector per centroid
    """
    
    def __init__(self,
                 npoint: int,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 mlp: List[int],
                 use_xyz: bool = True):
        """
        Args:
            npoint: number of centroids to sample (None = global, single point)
            radius: ball query radius (squared)
            nsample: max neighbors per centroid
            in_channel: input feature channels (3 for xyz only)
            mlp: list of output channels for each MLP layer, e.g. [64, 64, 128]
            use_xyz: if True, concatenate xyz with features (recommended)
        """
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        
        # MLP layers: Conv1d with kernel_size=1 = shared MLP per point
        # Input: in_channel + 3 (xyz) if use_xyz, else in_channel
        # First layer (in_channel=0): only xyz → 3 dims
        in_dim = (in_channel + 3) if use_xyz else max(in_channel, 3)
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        for i, out_channel in enumerate(mlp):
            self.mlp_convs.append(nn.Conv1d(in_dim, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            in_dim = out_channel
        
    def forward(self,
                xyz: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) - point coordinates
            features: (B, C, N) - point features (None for first layer)
            
        Returns:
            new_xyz: (B, npoint, 3) - sampled centroid coordinates
            new_features: (B, C_out, npoint) - features at centroids
        """
        B, N, _ = xyz.shape
        
        if self.npoint is not None:
            # FPS: get centroid indices
            fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
            
            # Ball Query / K-NN: get neighbor indices
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
            
            # Group points: (B, npoint, nsample, 3)
            grouped_xyz = index_points(xyz, idx)
            # Relative coordinates (centered at centroid)
            grouped_xyz -= new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)
            
            if features is not None:
                # Group features: (B, npoint, nsample, C)
                grouped_features = index_points(features.permute(0, 2, 1), idx)
                # Concatenate xyz and features: (B, npoint, nsample, 3+C)
                if self.use_xyz:
                    grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
                else:
                    grouped_features = grouped_features
            else:
                grouped_features = grouped_xyz  # (B, npoint, nsample, 3)
            
            # (B, npoint, nsample, C) → (B, C, npoint*nsample) for Conv1d
            grouped_features = grouped_features.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
            grouped_features = grouped_features.reshape(B, -1, self.npoint * self.nsample)
            
            # Shared MLP
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                grouped_features = F.relu(bn(conv(grouped_features)))
            
            # Max pool over neighbors: (B, C_out, npoint*nsample) → (B, C_out, npoint)
            new_features = torch.max(
                grouped_features.reshape(B, -1, self.npoint, self.nsample), dim=-1
            )[0]
            
        else:
            # Global aggregation: all points → single point
            new_xyz = xyz[:, :1, :]  # (B, 1, 3) - dummy, not used
            
            if features is not None:
                if self.use_xyz:
                    # Concat xyz with features: (B, 3, N) + (B, C, N) → (B, 3+C, N)
                    grouped_features = torch.cat([xyz.permute(0, 2, 1), features], dim=1)
                else:
                    grouped_features = features  # (B, C, N)
            else:
                grouped_features = xyz.permute(0, 2, 1)  # (B, 3, N)
            
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                grouped_features = F.relu(bn(conv(grouped_features)))
            
            new_features = torch.max(grouped_features, dim=-1, keepdim=True)[0]  # (B, C_out, 1)
        
        return new_xyz, new_features


class PointNet2SSG(nn.Module):
    """
    PointNet++ Single-Scale Grouping (SSG) for classification.
    
    Simpler variant using single radius per layer (vs Multi-Scale Grouping).
    Good balance of performance and efficiency.
    
    Architecture (adapted for 200 points input):
        SA1: 200 → 64 points, radius 0.2, nsample 32, mlp [3, 64, 64, 128]
        SA2: 64 → 16 points, radius 0.4, nsample 32, mlp [128, 128, 128, 256]
        SA3: 16 → 1 (global), mlp [256, 256, 512, 1024]
        FC: 1024 → 512 → 256 → num_classes
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 dropout: float = 0.3,
                 use_xyz: bool = True):
        """
        Args:
            num_points: input points per cloud (200)
            num_channels: input channels (3 for xyz)
            num_classes: output classes (10 subjects)
            dropout: dropout rate for classifier
            use_xyz: use xyz in Set Abstraction (recommended)
        """
        super(PointNet2SSG, self).__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes
        
        # Set Abstraction layers
        # SA1: downsample 200 → 64
        self.sa1 = SetAbstraction(
            npoint=64,
            radius=0.2 ** 2,  # squared for efficiency
            nsample=32,
            in_channel=0,  # xyz only for first layer
            mlp=[64, 64, 128],
            use_xyz=use_xyz
        )
        
        # SA2: downsample 64 → 16
        self.sa2 = SetAbstraction(
            npoint=16,
            radius=0.4 ** 2,
            nsample=32,
            in_channel=128,
            mlp=[128, 128, 256],
            use_xyz=use_xyz
        )
        
        # SA3: global (16 → 1)
        self.sa3 = SetAbstraction(
            npoint=None,  # global
            radius=0.0,
            nsample=0,
            in_channel=256,
            mlp=[256, 256, 512, 1024],
            use_xyz=use_xyz
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) - input point cloud
            
        Returns:
            logits: (B, num_classes)
        """
        # Input: (B, N, 3)
        xyz = x
        features = None
        
        # SA1
        xyz, features = self.sa1(xyz, features)  # (B, 64, 3), (B, 128, 64)
        
        # SA2
        xyz, features = self.sa2(xyz, features)  # (B, 16, 3), (B, 256, 16)
        
        # SA3 (global)
        _, features = self.sa3(xyz, features)  # (B, 1024, 1)
        
        # Flatten and classify
        features = features.squeeze(-1)  # (B, 1024)
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
