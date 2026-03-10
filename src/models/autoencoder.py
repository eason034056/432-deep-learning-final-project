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
from typing import Any, Dict, Tuple

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
                 channel_dims: Tuple[int, ...] = (64, 128, 1024),
                 decoder_dims: Tuple[int, ...] = (512, 256, 128)):
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

        decoder_layers = []
        in_dim = latent_dim + 2
        for i, hidden_dim in enumerate(decoder_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if i < len(decoder_dims) - 1:
                decoder_layers.extend([
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
            else:
                decoder_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, 3))
        self.decoder_mlp = nn.Sequential(*decoder_layers)

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


def _require_section(config: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise KeyError(f"Missing required config section: {path}")
    return section


def _require_value(config: Dict[str, Any], key: str, path: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config value: {path}")
    return config[key]


def _to_tuple(value: Any, path: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{path} must be a non-empty list or tuple")
    return tuple(int(v) for v in value)


def get_autoencoder_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize autoencoder-specific configuration."""
    data_cfg = _require_section(config, 'data', 'data')
    ae_cfg = _require_section(config, 'autoencoder', 'autoencoder')
    common_cfg = _require_section(ae_cfg, 'common', 'autoencoder.common')
    mlp_cfg = _require_section(ae_cfg, 'mlp', 'autoencoder.mlp')
    pointnet_cfg = _require_section(ae_cfg, 'pointnet', 'autoencoder.pointnet')
    train_cfg = _require_section(ae_cfg, 'train', 'autoencoder.train')
    eval_cfg = _require_section(ae_cfg, 'eval', 'autoencoder.eval')
    early_stopping_cfg = _require_section(
        train_cfg, 'early_stopping', 'autoencoder.train.early_stopping'
    )

    num_points = int(_require_value(data_cfg, 'num_points', 'data.num_points'))
    num_channels = int(
        common_cfg.get('num_channels', data_cfg.get('num_channels'))
    )

    if num_channels <= 0:
        raise ValueError("autoencoder.common.num_channels must be positive")

    return {
        'common': {
            'num_points': num_points,
            'num_channels': num_channels,
        },
        'mlp': {
            'latent_dim': int(_require_value(mlp_cfg, 'latent_dim', 'autoencoder.mlp.latent_dim')),
            'hidden_dims': _to_tuple(_require_value(mlp_cfg, 'hidden_dims', 'autoencoder.mlp.hidden_dims'),
                                     'autoencoder.mlp.hidden_dims'),
            'dropout': float(_require_value(mlp_cfg, 'dropout', 'autoencoder.mlp.dropout')),
        },
        'pointnet': {
            'latent_dim': int(_require_value(pointnet_cfg, 'latent_dim', 'autoencoder.pointnet.latent_dim')),
            'channel_dims': _to_tuple(
                _require_value(pointnet_cfg, 'channel_dims', 'autoencoder.pointnet.channel_dims'),
                'autoencoder.pointnet.channel_dims'
            ),
            'decoder_dims': _to_tuple(
                pointnet_cfg.get('decoder_dims', (512, 256, 128)),
                'autoencoder.pointnet.decoder_dims'
            ),
            'dropout': float(_require_value(pointnet_cfg, 'dropout', 'autoencoder.pointnet.dropout')),
            'use_tnet': bool(_require_value(pointnet_cfg, 'use_tnet', 'autoencoder.pointnet.use_tnet')),
        },
        'train': {
            'batch_size': int(_require_value(train_cfg, 'batch_size', 'autoencoder.train.batch_size')),
            'learning_rate': float(
                _require_value(train_cfg, 'learning_rate', 'autoencoder.train.learning_rate')
            ),
            'weight_decay': float(
                _require_value(train_cfg, 'weight_decay', 'autoencoder.train.weight_decay')
            ),
            'num_epochs': int(_require_value(train_cfg, 'num_epochs', 'autoencoder.train.num_epochs')),
            'augment': bool(_require_value(train_cfg, 'augment', 'autoencoder.train.augment')),
            'early_stopping': {
                'enabled': bool(
                    _require_value(
                        early_stopping_cfg, 'enabled', 'autoencoder.train.early_stopping.enabled'
                    )
                ),
                'patience': int(
                    _require_value(
                        early_stopping_cfg, 'patience', 'autoencoder.train.early_stopping.patience'
                    )
                ),
                'min_delta': float(
                    _require_value(
                        early_stopping_cfg, 'min_delta', 'autoencoder.train.early_stopping.min_delta'
                    )
                ),
            },
        },
        'eval': {
            'batch_size': int(_require_value(eval_cfg, 'batch_size', 'autoencoder.eval.batch_size')),
            'num_samples': int(_require_value(eval_cfg, 'num_samples', 'autoencoder.eval.num_samples')),
        },
    }


def create_autoencoder_from_config(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """Build an autoencoder from the normalized project config."""
    ae_cfg = get_autoencoder_config(config)
    common_cfg = ae_cfg['common']

    if model_type == 'mlp_ae':
        mlp_cfg = ae_cfg['mlp']
        return MLPAutoencoder(
            num_points=common_cfg['num_points'],
            num_channels=common_cfg['num_channels'],
            latent_dim=mlp_cfg['latent_dim'],
            hidden_dims=mlp_cfg['hidden_dims'],
            dropout=mlp_cfg['dropout'],
        )

    if model_type == 'pointnet_ae':
        pointnet_cfg = ae_cfg['pointnet']
        return PointNetAutoencoder(
            num_points=common_cfg['num_points'],
            num_channels=common_cfg['num_channels'],
            latent_dim=pointnet_cfg['latent_dim'],
            dropout=pointnet_cfg['dropout'],
            use_tnet=pointnet_cfg['use_tnet'],
            channel_dims=pointnet_cfg['channel_dims'],
            decoder_dims=pointnet_cfg['decoder_dims'],
        )

    raise ValueError(f"Unknown model: {model_type}. Use mlp_ae or pointnet_ae")


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
    if reduce == 'sum':
        return cd.sum()
    if reduce == 'none':
        return cd
    raise ValueError("reduce must be one of: 'mean', 'sum', 'none'")
