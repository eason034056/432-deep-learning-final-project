"""
Dataset loader for FAUST human mesh dataset.

This module handles:
- Loading FAUST mesh files (.ply or .obj format)
- Converting meshes to point clouds
- Train/Val/Test splitting with stratification
- Data augmentation during training
- PyTorch Dataset and DataLoader creation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
import trimesh
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing import (
    mesh_to_point_cloud,
    sample_or_pad_points,
    apply_augmentation,
    convert_to_tensor,
    normalize_to_unit_sphere
)


class FAUSTPointCloudDataset(Dataset):
    """
    PyTorch Dataset for FAUST point cloud data.
    
    FAUST dataset structure:
    - 10 subjects (subjects 0-9)
    - Each subject has 10 different poses
    - Total: 100 mesh files
    - Task: 10-class classification (identify which subject)
    
    Attributes:
        data: numpy array of point clouds, shape (N_samples, num_points, 3)
        labels: numpy array of subject IDs, shape (N_samples,)
        augment: whether to apply augmentation (True for train, False for val/test)
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray,
                 augment: bool = False,
                 rotation_range: float = 360.0,
                 translation_range: float = 3.0,
                 normalize_center: bool = True,
                 normalize_scale: bool = True):
        """
        Initialize dataset with preprocessed point clouds.
        
        Args:
            data: point cloud data, shape (N, num_points, 3)
            labels: subject labels, shape (N,)
            augment: if True, apply random augmentation on-the-fly
            rotation_range: max rotation angle in degrees
            translation_range: max translation distance in meters
            normalize_center: whether to center the point cloud
            normalize_scale: whether to scale to unit sphere
        """
        self.data = data  # Shape: (N_samples, 200, 3)
        self.labels = labels  # Shape: (N_samples,)
        self.augment = augment  # Boolean flag for augmentation
        self.rotation_range = rotation_range  # For random rotation
        self.translation_range = translation_range  # For random translation
        self.normalize_center = normalize_center  # Whether to center the point cloud
        self.normalize_scale = normalize_scale  # Whether to scale to unit sphere
        
    def __len__(self) -> int:
        """
        Return total number of samples in dataset.
        
        Required by PyTorch Dataset interface.
        
        Returns:
            length: number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample by index.
        
        This method is called by DataLoader during training/evaluation.
        
        Args:
            idx: index of sample to retrieve (0 to len-1)
            
        Returns:
            point_cloud: torch tensor of shape (num_points, 3)
            label: torch tensor of shape (1,) containing subject ID
        """
        # Get the point cloud and label for this index
        point_cloud = self.data[idx].copy()  # Shape: (200, 3)
        label = self.labels[idx]  # Scalar: subject ID (0-9)
        
        # Apply augmentation if in training mode
        if self.augment:
            point_cloud = apply_augmentation(
                point_cloud,
                rotation=True,
                translation=True,
                normalize=True,
                rotation_range=self.rotation_range,
                translation_range=self.translation_range,
                normalize_center=self.normalize_center,
                normalize_scale=self.normalize_scale
            )
        
        # Convert to PyTorch tensors
        point_cloud = torch.from_numpy(point_cloud).float()  # (200, 3)
        label = torch.tensor(label, dtype=torch.long)  # Scalar
        
        return point_cloud, label


def load_mesh_file(file_path: str) -> np.ndarray:
    """
    Load a mesh file and return its vertices.
    
    Supports .ply, .obj, .off formats via trimesh library.
    
    Args:
        file_path: path to mesh file
        
    Returns:
        vertices: numpy array of shape (V, 3) where V is number of vertices
        
    Example:
        >>> vertices = load_mesh_file("data/raw/tr_reg_000.ply")
        >>> vertices.shape
        (6890, 3)
    """
    try:
        # Load mesh using trimesh (handles multiple formats)
        mesh = trimesh.load(file_path)
        
        # Extract vertices as numpy array
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        return vertices
    
    except Exception as e:
        print(f"Error loading mesh file {file_path}: {e}")
        raise


def parse_faust_filename(filename: str) -> Tuple[int, int]:
    """
    Parse FAUST filename to extract subject ID and pose ID.
    
    FAUST naming convention:
    - Format: tr_reg_XXX.ply or tr_reg_XXX.obj
    - XXX is a 3-digit number from 000 to 099
    - Subject ID = XXX // 10 (0-9)
    - Pose ID = XXX % 10 (0-9)
    
    Args:
        filename: e.g., "tr_reg_000.ply" or "tr_reg_049.obj"
        
    Returns:
        subject_id: which person (0-9)
        pose_id: which pose (0-9)
        
    Example:
        >>> parse_faust_filename("tr_reg_037.ply")
        (3, 7)  # Subject 3, Pose 7
    """
    # Extract the numeric part from filename
    # e.g., "tr_reg_037.ply" -> "037"
    basename = os.path.basename(filename)  # Get filename without path
    name_without_ext = os.path.splitext(basename)[0]  # Remove extension
    
    # Extract the last 3 digits
    numeric_str = name_without_ext.split('_')[-1]  # "037"
    file_idx = int(numeric_str)  # Convert to integer: 37
    
    # FAUST convention: 10 subjects × 10 poses = 100 files
    subject_id = file_idx // 10  # Integer division: 37 // 10 = 3
    pose_id = file_idx % 10  # Modulo: 37 % 10 = 7
    
    return subject_id, pose_id


def load_faust_dataset(data_dir: str,
                       num_points: int = 200,
                       samples_per_mesh: int = 100,
                       use_fps: bool = True,
                       normalize_center: bool = True,
                       normalize_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all FAUST meshes and convert to point cloud dataset.
    
    Process:
    1. Scan data_dir for all mesh files
    2. For each mesh, generate multiple point cloud samples (default 100 per mesh, configurable)
    3. Each sample has num_points (200) uniformly sampled points
    4. Total samples: 100 meshes × samples_per_mesh = N_total samples
    
    Args:
        data_dir: directory containing FAUST mesh files
        num_points: number of points per point cloud sample
        samples_per_mesh: number of point cloud samples to generate per mesh
        use_fps: whether to use farthest point sampling (True) or random (False)
        
    Returns:
        data: numpy array of shape (N_total, num_points, 3)
        labels: numpy array of shape (N_total,) with subject IDs
        filenames: list of original mesh filenames
        
    Example:
        >>> data, labels, files = load_faust_dataset("data/raw", num_points=200, samples_per_mesh=500)
        >>> data.shape
        (50000, 200, 3)  # 100 meshes × 500 samples each
        >>> labels.shape
        (10000,)
    """
    data_path = Path(data_dir)
    
    # Find all mesh files (supports .ply, .obj, .off)
    mesh_files = []
    for ext in ['*.ply', '*.obj', '*.off']:
        mesh_files.extend(list(data_path.glob(ext)))
    
    # Sort files for reproducibility
    mesh_files = sorted(mesh_files)
    
    if len(mesh_files) == 0:
        raise FileNotFoundError(f"No mesh files found in {data_dir}")
    
    print(f"Found {len(mesh_files)} mesh files in {data_dir}")
    
    # Initialize lists to collect all samples
    all_point_clouds = []  # List of (num_points, 3) arrays
    all_labels = []  # List of subject IDs
    all_filenames = []  # List of filenames
    
    # Process each mesh file
    for mesh_file in mesh_files:
        filename = str(mesh_file.name)
        
        # Parse filename to get subject ID
        try:
            subject_id, pose_id = parse_faust_filename(filename)
        except Exception as e:
            print(f"Warning: Could not parse filename {filename}, skipping. Error: {e}")
            continue
        
        # Load mesh vertices
        try:
            vertices = load_mesh_file(str(mesh_file))
        except Exception as e:
            print(f"Warning: Could not load {filename}, skipping. Error: {e}")
            continue
        
        # Generate multiple samples from this mesh
        for _ in range(samples_per_mesh):
            # Sample point cloud from mesh
            point_cloud = mesh_to_point_cloud(vertices, num_samples=num_points)
            
            # Ensure correct size via sampling or padding
            point_cloud = sample_or_pad_points(point_cloud, num_points, use_fps=use_fps)

            # Normalize for consistency across splits
            point_cloud = normalize_to_unit_sphere(point_cloud, normalize_center=normalize_center, normalize_scale=normalize_scale)
            
            # Add to collection
            all_point_clouds.append(point_cloud)
            all_labels.append(subject_id)
            all_filenames.append(filename)
    
    # Convert lists to numpy arrays
    data = np.array(all_point_clouds, dtype=np.float32)  # Shape: (N, num_points, 3)
    labels = np.array(all_labels, dtype=np.int64)  # Shape: (N,)
    
    print(f"Generated {len(data)} point cloud samples")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique subjects: {np.unique(labels)}")
    
    return data, labels, all_filenames


def stratified_split_grouped(data: np.ndarray,
                             labels: np.ndarray,
                             filenames: List[str],
                             samples_per_mesh: int,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.2,
                             random_seed: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Split dataset by mesh groups to prevent data leakage.
    
    ✅ This fixes the data leakage issue by ensuring all samples from
    the same mesh stay in the same split (train/val/test).
    
    Problem with old approach:
    - Each mesh generates multiple samples (e.g., 100 samples per mesh)
    - These samples are very similar (just different random sampling)
    - Old split scattered these samples across train/val/test
    - Model memorizes training meshes → high train acc, low val acc
    
    New approach:
    - Group samples by their source mesh
    - Split meshes (not individual samples) into train/val/test
    - All samples from same mesh go to same split
    
    Args:
        data: point cloud data, shape (N, num_points, 3)
        labels: subject labels, shape (N,)
        filenames: list of source mesh filenames for each sample
        samples_per_mesh: number of samples generated per mesh
        train_ratio: proportion of meshes for training
        val_ratio: proportion of meshes for validation
        test_ratio: proportion of meshes for testing
        random_seed: random seed for reproducibility
        
    Returns:
        X_train, y_train: training data and labels
        X_val, y_val: validation data and labels
        X_test, y_test: test data and labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Group samples by mesh filename
    unique_meshes = []
    mesh_to_indices = {}
    mesh_to_subject = {}
    
    for idx, filename in enumerate(filenames):
        if filename not in mesh_to_indices:
            unique_meshes.append(filename)
            mesh_to_indices[filename] = []
            # Get subject ID from label
            mesh_to_subject[filename] = labels[idx]
        mesh_to_indices[filename].append(idx)
    
    print(f"\nGrouped split info:")
    print(f"Total samples: {len(data)}")
    print(f"Unique meshes: {len(unique_meshes)}")
    print(f"Samples per mesh: {samples_per_mesh}")
    
    # Convert to arrays for sklearn
    unique_meshes = np.array(unique_meshes)
    mesh_subjects = np.array([mesh_to_subject[m] for m in unique_meshes])
    
    # Split meshes (not samples) with stratification by subject
    meshes_temp, meshes_test, subjects_temp, subjects_test = train_test_split(
        unique_meshes, mesh_subjects,
        test_size=test_ratio,
        stratify=mesh_subjects,
        random_state=random_seed
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    meshes_train, meshes_val, subjects_train, subjects_val = train_test_split(
        meshes_temp, subjects_temp,
        test_size=val_ratio_adjusted,
        stratify=subjects_temp,
        random_state=random_seed
    )
    
    # Collect sample indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for mesh in meshes_train:
        train_indices.extend(mesh_to_indices[mesh])
    for mesh in meshes_val:
        val_indices.extend(mesh_to_indices[mesh])
    for mesh in meshes_test:
        test_indices.extend(mesh_to_indices[mesh])
    
    # Create splits
    X_train = data[train_indices]
    y_train = labels[train_indices]
    X_val = data[val_indices]
    y_val = labels[val_indices]
    X_test = data[test_indices]
    y_test = labels[test_indices]
    
    print(f"\nMesh split:")
    print(f"Train meshes: {len(meshes_train)} → {len(X_train)} samples")
    print(f"Val meshes: {len(meshes_val)} → {len(X_val)} samples")
    print(f"Test meshes: {len(meshes_test)} → {len(X_test)} samples")
    
    print(f"\nSubject distribution:")
    print(f"Train: {np.bincount(y_train)}")
    print(f"Val: {np.bincount(y_val)}")
    print(f"Test: {np.bincount(y_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      batch_size: int = 64,
                      num_workers: int = 4,
                      augment_train: bool = True,
                      device: str = 'cpu',
                      rotation_range: float = 15.0,
                      translation_range: float = 0.1,
                      normalize_center: bool = True,
                      normalize_scale: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    DataLoader handles:
    - Batching: group samples into batches
    - Shuffling: randomize order (for training)
    - Parallel loading: use multiple workers for speed
    
    Args:
        X_train, y_train: training data and labels
        X_val, y_val: validation data and labels
        X_test, y_test: test data and labels
        batch_size: number of samples per batch
        num_workers: number of parallel data loading workers
        augment_train: whether to apply augmentation to training data
        device: device type ('cuda', 'mps', or 'cpu') - affects pin_memory setting
        rotation_range: max rotation angle in degrees for augmentation
        translation_range: max translation distance in meters for augmentation
        normalize_scale: whether to scale point clouds to unit sphere
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
        ... )
        >>> for batch_data, batch_labels in train_loader:
        ...     # batch_data shape: (64, 200, 3)
        ...     # batch_labels shape: (64,)
        ...     pass
    """
    # Create Dataset objects
    train_dataset = FAUSTPointCloudDataset(
        X_train, y_train,
        augment=augment_train,  # Apply augmentation only to training
        rotation_range=rotation_range,
        translation_range=translation_range,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    
    val_dataset = FAUSTPointCloudDataset(
        X_val, y_val,
        augment=False,  # No augmentation for validation
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    
    test_dataset = FAUSTPointCloudDataset(
        X_test, y_test,
        augment=False,  # No augmentation for testing
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    
    # pin_memory speeds up CPU->GPU transfer, but only works with CUDA
    # MPS (Mac M1/M2 GPU) and CPU don't support it
    use_pin_memory = (device == 'cuda')
    
    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only use with CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader


def save_processed_dataset(data: np.ndarray,
                          labels: np.ndarray,
                          save_path: str,
                          filenames: Optional[List[str]] = None,
                          normalized: bool = True,
                          samples_per_mesh: Optional[int] = None) -> None:
    """
    Save preprocessed dataset to disk for faster loading later.
    
    Args:
        data: point cloud data, shape (N, num_points, 3)
        labels: subject labels, shape (N,)
        save_path: path to save .npz file
        filenames: optional list of source mesh filenames for each sample
        
    Example:
        >>> save_processed_dataset(data, labels, "data/processed/faust_pc.npz", filenames)
    """
    save_kwargs = {
        'data': data,
        'labels': labels,
        'normalized': np.array(int(normalized), dtype=np.int32)
    }

    if samples_per_mesh is not None:
        save_kwargs['samples_per_mesh'] = np.array(samples_per_mesh, dtype=np.int32)

    if filenames is not None:
        save_kwargs['filenames'] = np.array(filenames, dtype=object)
    
    np.savez_compressed(save_path, **save_kwargs)
    print(f"Saved processed dataset to {save_path}")


def load_processed_dataset(load_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
    """
    Load preprocessed dataset from disk.
    
    Args:
        load_path: path to .npz file
        
    Returns:
        data: point cloud data, shape (N, num_points, 3)
        labels: subject labels, shape (N,)
        filenames: list of source mesh filenames (None if not available)
        
    Example:
        >>> data, labels, filenames = load_processed_dataset("data/processed/faust_pc.npz")
    """
    loaded = np.load(load_path, allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    filenames = loaded['filenames'].tolist() if 'filenames' in loaded else None
    metadata: Dict[str, Any] = {
        'normalized': bool(int(loaded['normalized'])) if 'normalized' in loaded else True,
        'samples_per_mesh': int(loaded['samples_per_mesh']) if 'samples_per_mesh' in loaded else None
    }
    print(f"Loaded processed dataset from {load_path}")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    if filenames:
        print(f"Filenames: {len(filenames)} entries")
    return data, labels, filenames, metadata


# Complete pipeline example:
#
# 1. Load FAUST dataset:
#    data, labels, files = load_faust_dataset("data/raw")
#
# 2. Split into train/val/test (grouped by mesh to avoid data leakage):
#    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
#        data, labels, files, samples_per_mesh
#    )
#
# 3. Create DataLoaders:
#    train_loader, val_loader, test_loader = create_dataloaders(
#        X_train, y_train, X_val, y_val, X_test, y_test
#    )
#
# 4. Use in training:
#    for epoch in range(num_epochs):
#        for batch_data, batch_labels in train_loader:
#            # Training step
#            pass

