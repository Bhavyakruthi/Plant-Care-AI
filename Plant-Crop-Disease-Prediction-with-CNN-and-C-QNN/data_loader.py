"""
Data Loading and Preprocessing for Quantum-Classical Hybrid Models
Handles both classical-size and quantum-size image preprocessing with careful attention
to quantum embedding requirements (power-of-2 dimensions, normalization)
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import json


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PlantVillageDataset(Dataset):
    """Plant Village Dataset with support for classical and quantum preprocessing"""
    
    def __init__(
        self,
        root_dir: str,
        transform_classical: Optional[transforms.Compose] = None,
        transform_quantum: Optional[transforms.Compose] = None,
        mode: str = "classical"  # "classical", "quantum", or "both"
    ):
        """
        Args:
            root_dir: Path to PlantVillage directory
            transform_classical: Transforms for classical models (224x224)
            transform_quantum: Transforms for quantum models (8x8 or 16x16)
            mode: Which preprocessing to apply
        """
        self.root_dir = Path(root_dir)
        self.transform_classical = transform_classical
        self.transform_quantum = transform_quantum
        self.mode = mode
        
        # Load all images and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Scan directory and load image paths with labels"""
        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Get all images in this class folder
            image_files = list(class_folder.glob("*.jpg")) + \
                         list(class_folder.glob("*.JPG")) + \
                         list(class_folder.glob("*.png"))
            
            for img_path in image_files:
                self.samples.append((str(img_path), idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.mode == "classical":
            if self.transform_classical:
                image = self.transform_classical(image)
            return image, label
            
        elif self.mode == "quantum":
            if self.transform_quantum:
                image = self.transform_quantum(image)
            return image, label
            
        elif self.mode == "both":
            # Return both versions
            img_classical = self.transform_classical(image) if self.transform_classical else image
            img_quantum = self.transform_quantum(image) if self.transform_quantum else image
            return img_classical, img_quantum, label
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_class_distribution(self):
        """Get number of samples per class"""
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[label] += 1
        return dict(class_counts)


def get_transforms(config, is_train: bool = True, augmentation: bool = True):
    """
    Create transformation pipelines for classical and quantum preprocessing
    
    Args:
        config: DataConfig object
        is_train: Whether this is training data
        augmentation: Whether to apply augmentation (should be False for quantum)
    """
    # Classical transforms (224x224 for ResNet/CNNs)
    if is_train and augmentation:
        transform_classical = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(config.classical_image_size),
            transforms.RandomHorizontalFlip(p=config.augmentation_params['horizontal_flip']),
            transforms.RandomVerticalFlip(p=config.augmentation_params.get('vertical_flip', 0.3)),
            transforms.RandomRotation(config.augmentation_params['rotation_degrees']),
            transforms.ColorJitter(**config.augmentation_params['color_jitter']),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])
    else:
        transform_classical = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.classical_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])
    
    # Quantum transforms (8x8 or 16x16, grayscale, NO augmentation)
    # Critical: Images must be power-of-2 dimensions for amplitude embedding
    transform_quantum = transforms.Compose([
        transforms.Resize(config.quantum_image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Normalize to [0, 1] then convert to proper format for quantum encoding
        # For amplitude encoding: will need L2 normalization later
        # For angle encoding: will scale to [-π, π] later
    ])
    
    return transform_classical, transform_quantum


def create_stratified_splits(
    dataset: PlantVillageDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/test splits maintaining class distribution
    
    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    set_seed(seed)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    # Split each class separately
    for class_idx, indices in class_indices.items():
        n_samples = len(indices)
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())
    
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def save_splits(train_idx, val_idx, test_idx, save_path: str):
    """Save split indices to file for reproducibility"""
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    with open(save_path, 'w') as f:
        json.dump(splits, f)
    print(f"Saved splits to {save_path}")


def load_splits(load_path: str) -> Tuple[List[int], List[int], List[int]]:
    """Load split indices from file"""
    with open(load_path, 'r') as f:
        splits = json.load(f)
    return splits['train'], splits['val'], splits['test']


def get_dataloaders(
    config,
    mode: str = "classical",
    data_fraction: float = 1.0
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create dataloaders for train/val/test splits
    
    Args:
        config: ExperimentConfig object
        mode: "classical", "quantum", or "both"
        data_fraction: Fraction of training data to use (for ablation studies)
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    data_config = config.data
    
    # Determine if we should use augmentation
    # Critical: NO augmentation for quantum-only models based on paper findings
    use_augmentation = data_config.use_augmentation and mode != "quantum"
    
    # Get transforms
    transform_classical_train, transform_quantum_train = get_transforms(
        data_config, is_train=True, augmentation=use_augmentation
    )
    transform_classical_val, transform_quantum_val = get_transforms(
        data_config, is_train=False, augmentation=False
    )
    
    # Create datasets
    if mode == "classical":
        train_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_train,
            mode="classical"
        )
        val_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_val,
            mode="classical"
        )
        test_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_val,
            mode="classical"
        )
        batch_size = data_config.batch_size_classical
        
    elif mode == "quantum":
        train_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_quantum=transform_quantum_train,
            mode="quantum"
        )
        val_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_quantum=transform_quantum_val,
            mode="quantum"
        )
        test_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_quantum=transform_quantum_val,
            mode="quantum"
        )
        batch_size = data_config.batch_size_quantum
        
    elif mode == "both":
        train_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_train,
            transform_quantum=transform_quantum_train,
            mode="both"
        )
        val_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_val,
            transform_quantum=transform_quantum_val,
            mode="both"
        )
        test_dataset = PlantVillageDataset(
            data_config.dataset_path,
            transform_classical=transform_classical_val,
            transform_quantum=transform_quantum_val,
            mode="both"
        )
        batch_size = data_config.batch_size_quantum  # Use smaller batch for quantum
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Get num classes
    num_classes = len(train_dataset.class_to_idx)
    
    # Create or load splits
    split_file = os.path.join(data_config.dataset_path, "../splits.json")
    if os.path.exists(split_file):
        print(f"Loading existing splits from {split_file}")
        train_idx, val_idx, test_idx = load_splits(split_file)
    else:
        print("Creating new stratified splits...")
        train_idx, val_idx, test_idx = create_stratified_splits(
            train_dataset,
            data_config.train_ratio,
            data_config.val_ratio,
            data_config.test_ratio,
            data_config.seed
        )
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        save_splits(train_idx, val_idx, test_idx, split_file)
    
    # Apply data fraction for ablation studies
    if data_fraction < 1.0:
        n_train_samples = int(len(train_idx) * data_fraction)
        # Stratified sampling
        class_indices = defaultdict(list)
        for idx in train_idx:
            _, label = train_dataset.samples[idx]
            class_indices[label].append(idx)
        
        sampled_train_idx = []
        for class_idx, indices in class_indices.items():
            n_class_samples = int(len(indices) * data_fraction)
            sampled_train_idx.extend(indices[:n_class_samples])
        
        train_idx = sampled_train_idx
        print(f"Using {data_fraction*100}% of training data: {len(train_idx)} samples")
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # For batch norm stability
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, num_classes


# Quantum-specific preprocessing utilities
def normalize_for_amplitude_encoding(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor for amplitude encoding
    Amplitude encoding requires L2 norm = 1
    
    Args:
        tensor: Input tensor of shape (batch, channels, height, width) or (batch, features)
    """
    if tensor.dim() == 4:
        # Flatten spatial dimensions
        tensor = tensor.view(tensor.size(0), -1)
    
    # L2 normalization
    norms = torch.norm(tensor, p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)  # Avoid division by zero
    normalized = tensor / norms
    
    return normalized


def normalize_for_angle_encoding(tensor: torch.Tensor, range_min: float = -np.pi, range_max: float = np.pi) -> torch.Tensor:
    """
    Normalize tensor for angle encoding
    Angle encoding typically uses angles in [-π, π]
    
    Args:
        tensor: Input tensor
        range_min, range_max: Target range for angles
    """
    if tensor.dim() == 4:
        tensor = tensor.view(tensor.size(0), -1)
    
    # Normalize to [0, 1]
    tensor_min = tensor.min(dim=1, keepdim=True)[0]
    tensor_max = tensor.max(dim=1, keepdim=True)[0]
    normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)
    
    # Scale to [range_min, range_max]
    scaled = normalized * (range_max - range_min) + range_min
    
    return scaled


if __name__ == "__main__":
    # Test data loading
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    config.data.dataset_path = "./PlantVillage"
    
    print("\n=== Testing Classical Dataloader ===")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(config, mode="classical")
    print(f"Number of classes: {num_classes}")
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        break
    
    print("\n=== Testing Quantum Dataloader ===")
    train_loader_q, val_loader_q, test_loader_q, _ = get_dataloaders(config, mode="quantum")
    
    for images, labels in train_loader_q:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        # Test amplitude encoding normalization
        normalized = normalize_for_amplitude_encoding(images)
        print(f"After amplitude normalization: {normalized.shape}, L2 norms: {torch.norm(normalized, p=2, dim=1)[:5]}")
        break
