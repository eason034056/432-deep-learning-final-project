"""
Point Cloud Autoencoder Models for Compression/Reconstruction.

Two architectures:
1. MLPAutoencoder: Flatten -> FC encoder -> latent -> FC decoder -> reshape
2. PointNet2Autoencoder: PointNet++ Set Abstraction encoder -> Folding decoder

Used for the second cognitive problem: Point Cloud Compression (Autoencoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import PointNet++ Set Abstraction for encoder
from .pointnet2 import SetAbstraction


class MLPAutoencoder(nn.Module):
    """
    MLP-based Point Cloud Autoencoder (Challenger).
    
    Architecture:
        Encoder: (B, N, 3) -> Flatten -> FC(600, 256) -> FC(256, 128) -> latent (B, latent_dim)
        Decoder: latent -> FC(128, 256) -> FC(256, 600) -> reshape -> (B, N, 3)
    
    Simple baseline - order-dependent, no geometric structure.
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 latent_dim: int = 128,
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 dropout: float = 0.1):
        super(MLPAutoencoder, self).__init__()
        self.num_points = num_points
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        input_dim = num_points * num_channels  # 600
        
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder (mirror structure)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) -> latent: (B, latent_dim)"""
        B = x.shape[0]
        x_flat = x.view(B, -1)
        return self.encoder(x_flat)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, N, 3)"""
        B = z.shape[0]
        out = self.decoder(z)
        return out.view(B, self.num_points, self.num_channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstructed, latent)"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class PointNet2Autoencoder(nn.Module):
    """
    PointNet++-based Point Cloud Autoencoder (Champion).
    
    Encoder: Set Abstraction layers (hierarchical) -> global feature (B, 1024)
    Decoder: Folding-style - latent + 2D grid -> MLP -> 3D points
    
    Permutation-invariant encoder, better geometric structure.
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 latent_dim: int = 1024,
                 dropout: float = 0.1,
                 use_xyz: bool = True):
        super(PointNet2Autoencoder, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Encoder: PointNet++ Set Abstraction (same as PointNet2SSG up to sa3)
        self.sa1 = SetAbstraction(
            npoint=64, radius=0.2 ** 2, nsample=32,
            in_channel=0, mlp=[64, 64, 128], use_xyz=use_xyz
        )
        self.sa2 = SetAbstraction(
            npoint=16, radius=0.4 ** 2, nsample=32,
            in_channel=128, mlp=[128, 128, 256], use_xyz=use_xyz
        )
        self.sa3 = SetAbstraction(
            npoint=None, radius=0.0, nsample=0,
            in_channel=256, mlp=[256, 256, 512, latent_dim], use_xyz=use_xyz
        )
        
        # Decoder: Folding - 2D grid + latent -> 3D points
        # Create fixed 2D grid (N, 2) - normalized to [-1, 1]
        grid_size = int(num_points ** 0.5)
        if grid_size * grid_size < num_points:
            grid_size += 1
        # Regular 2D grid, take first num_points
        g = torch.linspace(-1, 1, grid_size)
        xx, yy = torch.meshgrid(g, g, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:num_points]
        self.register_buffer('grid', grid.float())
        
        # Decoder MLP: (latent + 2) -> 3 per point
        # Input: for each of N points, concat [latent (1024), grid_xy (2)] -> 1026
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) -> latent: (B, latent_dim)"""
        xyz, features = x, None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        return features.squeeze(-1)  # (B, latent_dim)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, N, 3)"""
        B = z.shape[0]
        N = self.num_points
        # Expand: z (B, L) -> (B, N, L), grid (N, 2) -> (B, N, 2)
        z_expand = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        grid_expand = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        decoder_input = torch.cat([z_expand, grid_expand], dim=-1)  # (B, N, latent_dim+2)
        
        # MLP per point: (B, N, L+2) -> reshape to (B*N, L+2) -> (B*N, 3) -> (B, N, 3)
        out = self.decoder_mlp(decoder_input.view(B * N, -1))
        return out.view(B, N, 3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstructed, latent)"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor, 
                     reduce: str = 'mean') -> torch.Tensor:
    """
    Chamfer Distance between two point sets.
    
    CD = (1/|S1|) sum_p min_q ||p-q||^2 + (1/|S2|) sum_q min_p ||p-q||^2
    
    Args:
        pred: (B, N, 3) predicted point cloud
        target: (B, N, 3) target point cloud
        reduce: 'mean' or 'sum' over batch
        
    Returns:
        cd: scalar loss
    """
    # pred (B, N, 3), target (B, M, 3)
    # dist: (B, N, M) - pairwise squared distance
    diff = pred.unsqueeze(3) - target.unsqueeze(2)  # (B, N, M, 3)
    dist = (diff ** 2).sum(dim=-1)  # (B, N, M)
    
    # pred_to_target: for each pred point, min dist to target
    pred_to_target = dist.min(dim=2)[0]  # (B, N)
    target_to_pred = dist.min(dim=1)[0]  # (B, M)
    
    cd = pred_to_target.mean(dim=1) + target_to_pred.mean(dim=1)  # (B,)
    if reduce == 'mean':
        return cd.mean()
    return cd.sum()
