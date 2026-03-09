"""
Preprocessing utilities for point cloud data augmentation and normalization.

This module provides functions for:
- Random rotation around z-axis
- Random translation in x-y plane
- Normalization to unit sphere
- Point cloud sampling and padding
"""

import numpy as np
import torch
from typing import Tuple, Optional


def normalize_to_unit_sphere(points: np.ndarray, normalize_center: bool = True, normalize_scale: bool = True) -> np.ndarray:
    """
    Normalize point cloud to fit within a unit sphere centered at origin.
    
    This function supports three modes:
    1. No normalization (normalize_center=False, normalize_scale=False): returns original points
    2. Center only (normalize_center=True, normalize_scale=False): centers at origin
    3. Center + scale (normalize_center=True, normalize_scale=True): centers and scales to unit sphere
    
    Args:
        points: numpy array of shape (N, 3) where N is number of points
        normalize_center: whether to center the point cloud at origin
        normalize_scale: whether to scale to unit sphere (requires normalize_center=True)
        
    Returns:
        normalized_points: numpy array of shape (N, 3)
        
    Example:
        >>> points = np.array([[1, 2, 3], [4, 5, 6]])
        >>> no_norm = normalize_to_unit_sphere(points, normalize_center=False, normalize_scale=False)  # Original
        >>> centered = normalize_to_unit_sphere(points, normalize_center=True, normalize_scale=False)  # Center only
        >>> full_norm = normalize_to_unit_sphere(points, normalize_center=True, normalize_scale=True)  # Center + scale
    """
    # If no normalization requested, return original points
    if not normalize_center:
        return points.copy()
    
    # Step 1: Center the point cloud by subtracting the centroid (mean of all points)
    centroid = np.mean(points, axis=0)  # Shape: (3,) - average x, y, z
    points_centered = points - centroid  # Broadcasting: (N, 3) - (3,) = (N, 3)
    
    # If only centering requested, return centered points
    if not normalize_scale:
        return points_centered
    
    # Step 2: Find the maximum distance from origin to any point
    # np.linalg.norm computes Euclidean distance for each point
    distances = np.linalg.norm(points_centered, axis=1)  # Shape: (N,)
    max_distance = np.max(distances)  # Scalar: farthest point distance
    
    # Step 3: Scale all points so max distance becomes 1.0
    # Add small epsilon to avoid division by zero
    if max_distance > 0:
        points_normalized = points_centered / max_distance
    else:
        points_normalized = points_centered
    
    return points_normalized


def random_rotation_z(points: np.ndarray, angle_range: float = 360.0) -> np.ndarray:
    """
    Apply random rotation around z-axis to simulate different orientations.
    
    Rotation matrix around z-axis:
    | cos(θ)  -sin(θ)  0 |
    | sin(θ)   cos(θ)  0 |
    |   0        0     1 |
    
    Args:
        points: numpy array of shape (N, 3)
        angle_range: maximum rotation angle in degrees (default 360 for full rotation)
        
    Returns:
        rotated_points: numpy array of shape (N, 3)
        
    Example:
        >>> points = np.array([[1, 0, 0], [0, 1, 0]])
        >>> rotated = random_rotation_z(points, angle_range=90)
    """
    # Generate random angle in radians
    angle_deg = np.random.uniform(0, angle_range)  # Random angle in [0, angle_range]
    angle_rad = np.deg2rad(angle_deg)  # Convert to radians for trigonometric functions
    
    # Compute cos and sin for rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Construct 3x3 rotation matrix for z-axis rotation
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],  # First row rotates x,y coordinates
        [sin_theta,  cos_theta, 0],  # Second row rotates x,y coordinates
        [0,          0,         1]   # Third row keeps z unchanged
    ], dtype=np.float32)
    
    # Apply rotation: (N, 3) @ (3, 3).T = (N, 3)
    # Each point is multiplied by rotation matrix
    rotated_points = points @ rotation_matrix.T
    
    return rotated_points


def random_translation(points: np.ndarray, translation_range: float = 3.0) -> np.ndarray:
    """
    Apply random translation in x-y plane to simulate different positions.
    
    This simulates people standing at different locations in the room.
    Translation in z is typically not applied as radar is at fixed height.
    
    Args:
        points: numpy array of shape (N, 3)
        translation_range: maximum translation distance in meters (default 3.0)
        
    Returns:
        translated_points: numpy array of shape (N, 3)
        
    Example:
        >>> points = np.array([[0, 0, 1], [1, 1, 1]])
        >>> translated = random_translation(points, translation_range=2.0)
    """
    # Generate random translation vector in x-y plane
    # z translation is 0 to maintain height consistency
    tx = np.random.uniform(-translation_range, translation_range)  # x offset
    ty = np.random.uniform(-translation_range, translation_range)  # y offset
    tz = 0.0  # No vertical translation
    
    translation_vector = np.array([tx, ty, tz], dtype=np.float32)  # Shape: (3,)
    
    # Add translation to all points via broadcasting
    translated_points = points + translation_vector  # (N, 3) + (3,) = (N, 3)
    
    return translated_points


def apply_augmentation(points: np.ndarray, 
                      rotation: bool = True,
                      translation: bool = True,
                      normalize: bool = True,
                      rotation_range: float = 360.0,
                      translation_range: float = 3.0,
                      normalize_center: bool = True,
                      normalize_scale: bool = True) -> np.ndarray:
    """
    Apply full augmentation pipeline to point cloud.
    
    Standard pipeline order:
    1. Normalize to unit sphere (if enabled)
    2. Random rotation (if enabled)
    3. Random translation (if enabled)
    
    Args:
        points: numpy array of shape (N, 3)
        rotation: whether to apply random rotation (default True)
        translation: whether to apply random translation (default True)
        normalize: whether to normalize (default True)
        rotation_range: max rotation angle in degrees
        translation_range: max translation distance in meters
        normalize_center: whether to center the point cloud
        normalize_scale: whether to scale to unit sphere
        
    Returns:
        augmented_points: numpy array of shape (N, 3)
        
    Example:
        >>> points = np.random.rand(100, 3)
        >>> augmented = apply_augmentation(points, rotation=True, translation=True)
    """
    augmented = points.copy()  # Copy to avoid modifying original
    
    # Step 1: Normalize first to have consistent scale
    if normalize:
        augmented = normalize_to_unit_sphere(augmented, normalize_center=normalize_center, normalize_scale=normalize_scale)
    
    # Step 2: Apply rotation for orientation invariance
    if rotation:
        augmented = random_rotation_z(augmented, angle_range=rotation_range)
    
    # Step 3: Apply translation for position invariance
    if translation:
        augmented = random_translation(augmented, translation_range=translation_range)
    
    # Step 4: Re-normalize after augmentation to maintain consistent distribution
    # This ensures train and val data have same distribution
    if normalize and (rotation or translation):
        augmented = normalize_to_unit_sphere(augmented, normalize_center=normalize_center, normalize_scale=normalize_scale)
    
    return augmented


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Sample fixed number of points using Farthest Point Sampling (FPS).
    
    FPS algorithm:
    1. Start with a random point
    2. Iteratively select the point that is farthest from all selected points
    3. Repeat until desired number of points is reached
    
    This preserves the overall shape better than random sampling.
    
    Args:
        points: numpy array of shape (N, 3) where N >= num_samples
        num_samples: target number of points to sample
        
    Returns:
        sampled_points: numpy array of shape (num_samples, 3)
        
    Example:
        >>> points = np.random.rand(500, 3)
        >>> sampled = farthest_point_sampling(points, 200)
        >>> sampled.shape
        (200, 3)
    """
    N = points.shape[0]  # Total number of points
    
    # If already have exact number, return as is
    if N == num_samples:
        return points
    
    # Initialize: select first point randomly
    selected_indices = [np.random.randint(0, N)]  # List of selected point indices
    
    # Track minimum distance from each point to selected set
    distances = np.full(N, np.inf, dtype=np.float32)  # Shape: (N,), all initialized to infinity
    
    # Iteratively select num_samples-1 more points
    for _ in range(num_samples - 1):
        last_selected = selected_indices[-1]  # Most recently selected point index
        
        # Compute distance from all points to the newly selected point
        # Shape: (N, 3) - (3,) = (N, 3), then norm over axis=1 gives (N,)
        new_distances = np.linalg.norm(points - points[last_selected], axis=1)
        
        # Update minimum distances: for each point, keep the smaller distance
        distances = np.minimum(distances, new_distances)
        
        # Select the point with maximum minimum distance (farthest from selected set)
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
    
    # Return the selected points
    sampled_points = points[selected_indices]
    
    return sampled_points


def sample_or_pad_points(points: np.ndarray, num_points: int, use_fps: bool = True) -> np.ndarray:
    """
    Ensure point cloud has exactly num_points by sampling or padding.
    
    Cases:
    - If N > num_points: sample down using FPS or random sampling
    - If N < num_points: pad with zeros
    - If N == num_points: return as is
    
    Args:
        points: numpy array of shape (N, 3)
        num_points: target number of points (e.g., 200)
        use_fps: if True, use farthest point sampling; else random sampling
        
    Returns:
        processed_points: numpy array of shape (num_points, 3)
        
    Example:
        >>> points = np.random.rand(150, 3)
        >>> processed = sample_or_pad_points(points, 200)
        >>> processed.shape
        (200, 3)
    """
    N = points.shape[0]  # Current number of points
    
    if N == num_points:
        # Already correct size
        return points
    
    elif N > num_points:
        # Too many points: need to downsample
        if use_fps:
            # Use Farthest Point Sampling for better shape preservation
            return farthest_point_sampling(points, num_points)
        else:
            # Use random sampling (faster but less shape-preserving)
            indices = np.random.choice(N, num_points, replace=False)
            return points[indices]
    
    else:
        # Too few points: need to pad with zeros
        # Create array of zeros with target shape
        padded = np.zeros((num_points, 3), dtype=np.float32)
        
        # Copy existing points to the beginning
        padded[:N] = points
        
        # Remaining (num_points - N) points are zeros
        return padded


def convert_to_tensor(points: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor and move to specified device.
    
    Args:
        points: numpy array of shape (N, 3) or (B, N, 3)
        device: 'cpu' or 'cuda' for GPU
        
    Returns:
        tensor: PyTorch tensor on specified device
        
    Example:
        >>> points = np.random.rand(200, 3)
        >>> tensor = convert_to_tensor(points, device='cuda')
        >>> tensor.shape
        torch.Size([200, 3])
    """
    # Convert numpy to torch tensor with float32 precision
    tensor = torch.from_numpy(points).float()
    
    # Move to specified device (CPU or GPU)
    tensor = tensor.to(device)
    
    return tensor


def batch_augment(points_batch: np.ndarray,
                 rotation: bool = True,
                 translation: bool = True,
                 normalize: bool = True) -> np.ndarray:
    """
    Apply augmentation to a batch of point clouds.
    
    Args:
        points_batch: numpy array of shape (B, N, 3) where B is batch size
        rotation: whether to apply rotation
        translation: whether to apply translation
        normalize: whether to normalize
        
    Returns:
        augmented_batch: numpy array of shape (B, N, 3)
        
    Example:
        >>> batch = np.random.rand(32, 200, 3)  # 32 samples, 200 points each
        >>> augmented = batch_augment(batch)
        >>> augmented.shape
        (32, 200, 3)
    """
    B = points_batch.shape[0]  # Batch size
    augmented_batch = np.zeros_like(points_batch)  # Initialize output array
    
    # Apply augmentation to each sample in batch independently
    for i in range(B):
        augmented_batch[i] = apply_augmentation(
            points_batch[i],
            rotation=rotation,
            translation=translation,
            normalize=normalize
        )
    
    return augmented_batch


def mesh_to_point_cloud(mesh_vertices: np.ndarray, num_samples: int = 200) -> np.ndarray:
    """
    Sample point cloud from mesh vertices.
    
    For FAUST dataset, meshes typically have thousands of vertices.
    We sample a smaller number for computational efficiency.
    
    Args:
        mesh_vertices: numpy array of shape (V, 3) where V is number of vertices
        num_samples: number of points to sample from mesh
        
    Returns:
        point_cloud: numpy array of shape (num_samples, 3)
        
    Example:
        >>> vertices = np.random.rand(6890, 3)  # FAUST mesh has ~6890 vertices
        >>> pc = mesh_to_point_cloud(vertices, num_samples=200)
        >>> pc.shape
        (200, 3)
    """
    # Use farthest point sampling to get representative points
    point_cloud = farthest_point_sampling(mesh_vertices, num_samples)
    
    return point_cloud


# Summary of preprocessing pipeline:
# 
# 1. mesh_to_point_cloud: Convert mesh → point cloud
# 2. sample_or_pad_points: Ensure fixed size (200 points)
# 3. apply_augmentation: 
#    - normalize_to_unit_sphere: Scale to [-1, 1]
#    - random_rotation_z: Rotate 0-360° for orientation invariance
#    - random_translation: Shift x,y for position invariance
# 4. convert_to_tensor: Convert to PyTorch tensor for model input
