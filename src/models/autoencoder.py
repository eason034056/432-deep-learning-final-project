"""
Point Cloud Autoencoder Models for Compression/Reconstruction.

Architectures:
- MLPAutoencoder: Flatten -> FC encoder -> latent -> FC decoder -> reshape
- PointNetAutoencoder: PointNet backbone encoder -> Folding decoder (permutation-invariant)

Used for the second cognitive problem: Point Cloud Compression (Autoencoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .pointnet_tiny import PointNetBackbone


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


class PointNetAutoencoder(nn.Module):
    """
    PointNet-based Point Cloud Autoencoder (permutation-invariant).

    Encoder: PointNetBackbone (T-Net + Shared MLP + GlobalMaxPool) -> global feature (B, 1024)
    Decoder: Folding-style - latent + 2D grid -> MLP -> 3D points

    Permutation-invariant encoder, better geometric structure than MLP.
    """

    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 latent_dim: int = 1024,
                 dropout: float = 0.1,
                 use_tnet: bool = True,
                 channel_dims: Tuple[int, ...] = (64, 128, 1024)):
        super(PointNetAutoencoder, self).__init__()
        self.num_points = num_points
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        # Encoder: PointNet backbone
        self.encoder = PointNetBackbone(
            input_channels=num_channels,
            use_tnet=use_tnet,
            channel_dims=channel_dims
        )

        # Project latent if needed (backbone outputs channel_dims[-1], typically 1024)
        backbone_dim = channel_dims[-1]
        if latent_dim != backbone_dim:
            self.latent_proj = nn.Linear(backbone_dim, latent_dim)
        else:
            self.latent_proj = nn.Identity()

        # Decoder: Folding - 2D grid + latent -> 3D points
        grid_size = int(num_points ** 0.5)
        if grid_size * grid_size < num_points:
            grid_size += 1
        g = torch.linspace(-1, 1, grid_size)
        xx, yy = torch.meshgrid(g, g, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:num_points]
        self.register_buffer('grid', grid.float())

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
        # PointNetBackbone expects (B, C, N)
        x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        features = self.encoder(x)  # (B, 1024)
        return self.latent_proj(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, N, 3)"""
        B = z.shape[0]
        N = self.num_points
        z_expand = z.unsqueeze(1).expand(-1, N, -1)
        grid_expand = self.grid.unsqueeze(0).expand(B, -1, -1)
        decoder_input = torch.cat([z_expand, grid_expand], dim=-1)
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
    # Build all pairwise point differences:
    # pred.unsqueeze(2)   -> (B, N, 1, 3)
    # target.unsqueeze(1) -> (B, 1, M, 3)
    # broadcast result    -> (B, N, M, 3)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)  # (B, N, M)
    
    # pred_to_target: for each pred point, min dist to target
    pred_to_target = dist.min(dim=2)[0]  # (B, N)
    target_to_pred = dist.min(dim=1)[0]  # (B, M)
    
    cd = pred_to_target.mean(dim=1) + target_to_pred.mean(dim=1)  # (B,)
    if reduce == 'mean':
        return cd.mean()
    return cd.sum()
