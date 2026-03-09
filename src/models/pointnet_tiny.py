"""
Tiny PointNet for Point Cloud Classification.

Simplified PointNet: shared MLP + max pooling + FC classifier.
Input format: (B, N, 3) - batch, num_points, xyz.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class TNet(nn.Module):
    """Spatial transformer for 3x3 transformation (canonical alignment)."""
    
    def __init__(self, in_dim: int = 3, out_dim: int = 3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim * out_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> (B, C, N)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.out_dim, device=x.device).flatten().unsqueeze(0).expand(x.size(0), -1)
        x = x + iden
        return x.view(-1, self.out_dim, self.out_dim)


class PointNetBackbone(nn.Module):
    """Shared MLP + max pool backbone."""
    
    def __init__(self, channel_dims: Tuple[int, ...] = (64, 128, 1024)):
        super(PointNetBackbone, self).__init__()
        layers = []
        prev = 3
        for c in channel_dims:
            layers.append(nn.Conv1d(prev, c, 1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            prev = c
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        return x


class TinyPointNet(nn.Module):
    """
    Tiny PointNet for classification.
    Input: (B, N, 3), Output: (B, num_classes)
    """
    
    def __init__(self,
                 num_points: int = 200,
                 num_channels: int = 3,
                 num_classes: int = 10,
                 use_tnet: bool = True,
                 channel_dims: Tuple[int, ...] = (64, 128, 1024),
                 fc_dims: Tuple[int, ...] = (512, 256),
                 dropout: float = 0.3):
        super(TinyPointNet, self).__init__()
        self.use_tnet = use_tnet
        if use_tnet:
            self.tnet = TNet(3, 3)
        self.backbone = PointNetBackbone(channel_dims)
        prev = channel_dims[-1]
        fc_layers = []
        for d in fc_dims:
            fc_layers += [nn.Linear(prev, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = d
        fc_layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_tnet:
            trans = self.tnet(x)
            x = torch.bmm(x, trans)
        feat = self.backbone(x)
        return self.classifier(feat)
