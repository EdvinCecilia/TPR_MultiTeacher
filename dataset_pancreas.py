"""Pancreas dataset loader using MONAI."""

import json
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    EnsureTyped,
    Orientationd,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    SpatialPadd,
)


class PancreasDataset(Dataset):
    """Dataset for Pancreas 3D CT segmentation."""

    def __init__(
        self,
        data_dir: str,
        json_file: str = "dataset.json",
        data_list: Optional[list] = None,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        max_samples: Optional[int] = None,
        roi_size: tuple = (96, 96, 96),
        space_x: float = 1.5,
        space_y: float = 1.5,
        space_z: float = 1.5,
        a_min: float = -175.0,
        a_max: float = 250.0,
        pos: int = 1,
        neg: int = 1,
        num_samples: int = 1,
        flip_prob: float = 0.1,
        rotate_prob: float = 0.1,
    ):
        """
        Args:
            data_dir: Directory containing dataset
            json_file: JSON file with dataset split info
            data_list: Optional pre-computed data list (if provided, will use this instead of loading from JSON)
            transform: Optional transform pipeline
            is_training: Whether this is training data
            max_samples: Maximum number of samples to use (None for all samples)
            flip_prob: Probability for random flip augmentation
            rotate_prob: Probability for random rotation augmentation
        """
        self.data_dir = Path(data_dir)
        self.is_training = is_training
        self.roi_size = roi_size
        self.space_x = space_x
        self.space_y = space_y
        self.space_z = space_z
        self.a_min = a_min
        self.a_max = a_max
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

        # Use provided data_list or load from JSON
        if data_list is not None:
            self.data_list = data_list
        else:
            # Load dataset split from JSON
            json_path = self.data_dir / json_file
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Get split (training or validation)
            split_key = "training" if is_training else "validation"
            if split_key not in dataset_info:
                raise ValueError(f"JSON file does not contain '{split_key}' field")
            
            data_list_from_json = dataset_info[split_key]
            
            # Build full paths
            self.data_list = []
            for item in data_list_from_json:
                image_path = self.data_dir / item["image"].replace("./", "")
                label_path = self.data_dir / item["label"].replace("./", "")
                
                if image_path.exists() and label_path.exists():
                    self.data_list.append({
                        "image": str(image_path),
                        "label": str(label_path),
                    })
            
            # Limit dataset size if specified
            if max_samples is not None and max_samples > 0:
                self.data_list = self.data_list[:max_samples]
                print(f"Limited to {len(self.data_list)} {split_key} samples (max_samples={max_samples})")
            else:
                print(f"Found {len(self.data_list)} {split_key} samples")

        # Default transform if not provided
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _get_preprocessing_transform(self):
        """Preprocessing: load, orient, spacing, HU clip, crop foreground."""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(self.space_x, self.space_y, self.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.a_min,
                a_max=self.a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ])
    
    def _get_augmentation_transform(self):
        """Augmentation: train = rand crop + flip/rotate90/shift; val = type only."""
        crop_size = self.roi_size
        if self.is_training:
            return Compose([
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=crop_size,
                    pos=self.pos,
                    neg=self.neg,
                    num_samples=self.num_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True,
                ),
                # Pad to fixed size for batching
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=crop_size,
                    mode="constant",
                ),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.flip_prob),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=self.flip_prob),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=self.flip_prob),
                RandRotate90d(keys=["image", "label"], prob=self.rotate_prob, max_k=3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                EnsureTyped(keys=["image", "label"]),
            ])
        else:
            return Compose([
                EnsureTyped(keys=["image", "label"]),
            ])
    
    def _get_default_transform(self):
        """Get default transform pipeline."""
        preprocessing = self._get_preprocessing_transform()
        augmentation = self._get_augmentation_transform()
        return Compose([preprocessing, augmentation])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index].copy()
        if self.transform is not None:
            data = self.transform(data)
        return data


def get_pancreas_dataloaders(
    data_dir: str,
    json_file: str = "dataset.json",
    batch_size: int = 2,
    num_workers: int = 8,
    pin_memory: bool = True,
    cache_rate: float = 1.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    roi_size: tuple = (96, 96, 96),
    space_x: float = 1.5,
    space_y: float = 1.5,
    space_z: float = 2.0,
    a_min: float = -175.0,
    a_max: float = 250.0,
    pos: int = 1,
    neg: int = 1,
    num_samples: int = 1,
    flip_prob: float = 0.1,
    rotate_prob: float = 0.1,
    train_ratio: float = 0.8,
    seed: int = 42,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Create train and validation dataloaders.
    If dataset.json has both "training" and "validation", use them; else split "training" by train_ratio.

    Args:
        data_dir: Directory containing dataset
        json_file: JSON file with dataset split info
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        cache_rate: Fraction of data to cache in memory
        max_train_samples: Maximum number of training samples (None for all)
        max_val_samples: Maximum number of validation samples (None for all)
        train_ratio: Ratio of training data (used when only "training" field exists)
        seed: Random seed for train/val split
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    import random
    
    data_dir_path = Path(data_dir)
    json_path = data_dir_path / json_file
    
    # Load dataset info
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Check if we have separate training and validation splits
    has_separate_val = "validation" in dataset_info
    
    if has_separate_val:
        # Mode 1: Use existing training and validation splits
        print("Using training/validation split from JSON")
        train_data_list = dataset_info["training"]
        val_data_list = dataset_info["validation"]
    else:
        # Mode 2: Split training data into train/val
        print(f"Splitting training data by {train_ratio:.0%} for train/val")
        all_training_data = dataset_info["training"]
        
        # Shuffle and split
        random.seed(seed)
        shuffled = all_training_data.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_data_list = shuffled[:split_idx]
        val_data_list = shuffled[split_idx:]
        
        print(f"Train: {len(train_data_list)} samples, Val: {len(val_data_list)} samples")
    
    # Build full paths for training set
    train_list = []
    for item in train_data_list:
        image_path = data_dir_path / item["image"].replace("./", "")
        label_path = data_dir_path / item["label"].replace("./", "")
        
        if image_path.exists() and label_path.exists():
            train_list.append({
                "image": str(image_path),
                "label": str(label_path),
            })
    
    # Build full paths for validation set
    val_list = []
    for item in val_data_list:
        image_path = data_dir_path / item["image"].replace("./", "")
        label_path = data_dir_path / item["label"].replace("./", "")
        
        if image_path.exists() and label_path.exists():
            val_list.append({
                "image": str(image_path),
                "label": str(label_path),
            })
    
    # Limit dataset size if specified
    if max_train_samples is not None and max_train_samples > 0:
        train_list = train_list[:max_train_samples]
    if max_val_samples is not None and max_val_samples > 0:
        val_list = val_list[:max_val_samples]
    
    print(f"Final train: {len(train_list)}, val: {len(val_list)} samples")
    
    # Create datasets with custom data lists
    train_dataset = PancreasDataset(
        data_dir=data_dir,
        json_file=json_file,
        data_list=train_list,
        is_training=True,
        roi_size=roi_size,
        space_x=space_x,
        space_y=space_y,
        space_z=space_z,
        a_min=a_min,
        a_max=a_max,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    val_dataset = PancreasDataset(
        data_dir=data_dir,
        json_file=json_file,
        data_list=val_list,
        is_training=False,
        roi_size=roi_size,
        space_x=space_x,
        space_y=space_y,
        space_z=space_z,
        a_min=a_min,
        a_max=a_max,
        pos=pos,
        neg=neg,
        num_samples=1,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    
    # Get preprocessing and augmentation transforms
    train_preprocessing = train_dataset._get_preprocessing_transform()
    train_augmentation = train_dataset._get_augmentation_transform()
    
    val_preprocessing = val_dataset._get_preprocessing_transform()
    val_augmentation = val_dataset._get_augmentation_transform()
    
    # Create cached datasets
    train_cached = CacheDataset(
        data=train_dataset.data_list,
        transform=train_preprocessing,
        cache_rate=cache_rate,
        num_workers=min(num_workers, 4),
    )
    val_cached = CacheDataset(
        data=val_dataset.data_list,
        transform=val_preprocessing,
        cache_rate=cache_rate,
        num_workers=min(num_workers, 4),
    )
    
    # Wrap with augmentation transform
    class AugmentedDataset(Dataset):
        def __init__(self, base_dataset, augmentation_transform, num_samples=1):
            super().__init__(data=[], transform=None)
            self.base_dataset = base_dataset
            self.augmentation_transform = augmentation_transform
            self.num_samples = num_samples
        
        def __len__(self):
            return len(self.base_dataset) * self.num_samples

        def __getitem__(self, index):
            base_idx = index // self.num_samples
            sample_idx = index % self.num_samples
            
            data = self.base_dataset[base_idx]
            augmented = self.augmentation_transform(data)
            
            if self.num_samples > 1 and isinstance(augmented, list):
                return augmented[sample_idx]
            else:
                return augmented
    
    train_dataset = AugmentedDataset(train_cached, train_augmentation, num_samples=num_samples)
    val_dataset = AugmentedDataset(val_cached, val_augmentation, num_samples=1)
    
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "/path/to/your/dataset"

    train_loader, val_loader = get_pancreas_dataloaders(
        data_dir=data_dir,
        batch_size=2,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Unique labels: {torch.unique(batch['label'])}")
        break

