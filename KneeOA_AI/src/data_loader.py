"""
Data Loading, Preprocessing, and Augmentation Module
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

# ==================== IMAGE PREPROCESSING ====================
class ImagePreprocessor:
    """
    Handles X-ray image preprocessing including:
    - Resizing
    - Normalization
    - Histogram equalization for improved contrast
    - ROI extraction (optional)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize_mean: List[float] = None,
                 normalize_std: List[float] = None,
                 apply_clahe: bool = True):
        """
        Args:
            target_size: Target image dimensions
            normalize_mean: Normalization mean values
            normalize_std: Normalization std values
            apply_clahe: Apply Contrast Limited Adaptive Histogram Equalization
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.apply_clahe = apply_clahe
        
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from disk"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        if self.apply_clahe and image.dtype == np.uint8:
            return self.clahe.apply(image)
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving aspect ratio"""
        h, w = image.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = size[1]
            new_h = int(size[1] / aspect)
        else:
            new_h = size[0]
            new_w = int(size[0] * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target size
        top = (size[0] - new_h) // 2
        bottom = size[0] - new_h - top
        left = (size[1] - new_w) // 2
        right = size[1] - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=0)
        return padded
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to [0, 1]"""
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to X-ray image
            
        Returns:
            Preprocessed image (H, W, 3) in range [0, 1]
        """
        # Load image
        image = self.load_image(image_path)
        
        # Enhance contrast
        image = self.apply_clahe_enhancement(image)
        
        # Resize
        image = self.resize_image(image, self.target_size)
        
        # Normalize intensity
        image = self.normalize_intensity(image)
        
        # Convert grayscale to RGB (3 channels)
        image = np.stack([image] * 3, axis=-1)
        
        return image  # (H, W, 3)

# ==================== DATA AUGMENTATION ====================
class XRayAugmentation:
    """
    Medical imaging-specific augmentation techniques
    """
    
    def __init__(self, augmentation_enabled: bool = True,
                 prob: float = 0.5,
                 rotation_degrees: int = 15,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Args:
            augmentation_enabled: Whether to apply augmentation
            prob: Probability of applying each augmentation
            rotation_degrees: Range of rotation angles
            brightness_range: Min/max brightness multiplier
            contrast_range: Min/max contrast multiplier
        """
        self.augmentation_enabled = augmentation_enabled
        self.prob = prob
        
        if augmentation_enabled:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=rotation_degrees, p=prob),
                transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range, p=prob),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=prob * 0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), p=prob * 0.5),
            ])
        else:
            self.transform = transforms.Identity()
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to image tensor"""
        if self.augmentation_enabled:
            return self.transform(image)
        return image

# ==================== CUSTOM DATASET ====================
class KneeXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for Knee X-Ray Images
    Handles multiple expert annotations with consensus learning
    """
    
    def __init__(self, image_paths: List[str],
                 labels: List[int],
                 expert_annotations: Optional[List[Dict]] = None,
                 preprocessor: Optional[ImagePreprocessor] = None,
                 augmentation: Optional[XRayAugmentation] = None,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of class labels (KL grades)
            expert_annotations: Optional consensus labels from multiple experts
            preprocessor: Image preprocessing pipeline
            augmentation: Data augmentation pipeline
            transform: Additional torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.expert_annotations = expert_annotations or []
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.transform = transform or transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Preprocessed and augmented image tensor
            label: Class label (KL grade)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Preprocess image
        image = self.preprocessor.preprocess(image_path) if self.preprocessor else self._load_image(image_path)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        image = image.float()
        
        # Apply augmentation
        if self.augmentation:
            image = self.augmentation(image)
        
        # Apply additional transforms (normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Fallback image loading without preprocessing"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = np.stack([image] * 3, axis=-1)
        return image / 255.0

# ==================== DATA LOADER ====================
class KneeXRayDataLoader:
    """
    Manages data loading, splitting, and processing for training
    """
    
    def __init__(self, dataset_roots: Dict[str, Path],
                 data_config,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        Args:
            dataset_roots: Dictionary mapping dataset names to their root paths
            data_config: Configuration object for data
            batch_size: Batch size for DataLoader
            num_workers: Number of worker threads
        """
        self.dataset_roots = dataset_roots
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.preprocessor = ImagePreprocessor(
            target_size=data_config.image_size,
            normalize_mean=data_config.normalize_mean,
            normalize_std=data_config.normalize_std
        )
        
        self.augmentation_train = XRayAugmentation(
            augmentation_enabled=data_config.augmentation_enabled,
            prob=data_config.augmentation_prob,
            rotation_degrees=data_config.rotation_degrees,
            brightness_range=data_config.brightness_range,
            contrast_range=data_config.contrast_range
        )
        
        self.augmentation_val = XRayAugmentation(augmentation_enabled=False)
    
    def collect_images_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Scan dataset directories and collect image paths and labels
        
        Returns:
            image_paths: List of image file paths
            labels: List of KL grades (0-4)
        """
        image_paths = []
        labels = []
        
        for dataset_name, dataset_root in self.dataset_roots.items():
            dataset_path = Path(dataset_root)
            if not dataset_path.exists():
                logger.warning(f"Dataset not found: {dataset_path}")
                continue
            
            for class_idx, class_name in enumerate(self.data_config.class_names):
                class_dir = dataset_path / str(class_idx) + class_name
                
                if not class_dir.exists():
                    # Try alternative naming: just class number
                    class_dir = dataset_path / str(class_idx)
                
                if class_dir.exists():
                    for image_file in class_dir.glob("*"):
                        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
                            image_paths.append(str(image_file))
                            labels.append(class_idx)
        
        if not image_paths:
            logger.warning("No images found! Check dataset directory structure.")
        
        return image_paths, labels
    
    def split_dataset(self, image_paths: List[str], labels: List[int]) -> Dict:
        """
        Split data into train, validation, and test sets
        
        Returns:
            Dictionary with train/val/test indices and datasets
        """
        indices = np.arange(len(image_paths))
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.data_config.test_ratio,
            random_state=self.data_config.random_seed,
            stratify=labels
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.data_config.val_ratio / (1 - self.data_config.test_ratio),
            random_state=self.data_config.random_seed,
            stratify=[labels[i] for i in train_val_idx]
        )
        
        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Collect images and labels
        image_paths, labels = self.collect_images_and_labels()
        
        if not image_paths:
            raise ValueError("No images found in dataset!")
        
        logger.info(f"Found {len(image_paths)} images across {len(set(labels))} classes")
        
        # Split dataset
        splits = self.split_dataset(image_paths, labels)
        
        # Create datasets
        train_dataset = KneeXRayDataset(
            image_paths=[image_paths[i] for i in splits['train_idx']],
            labels=[labels[i] for i in splits['train_idx']],
            preprocessor=self.preprocessor,
            augmentation=self.augmentation_train
        )
        
        val_dataset = KneeXRayDataset(
            image_paths=[image_paths[i] for i in splits['val_idx']],
            labels=[labels[i] for i in splits['val_idx']],
            preprocessor=self.preprocessor,
            augmentation=self.augmentation_val
        )
        
        test_dataset = KneeXRayDataset(
            image_paths=[image_paths[i] for i in splits['test_idx']],
            labels=[labels[i] for i in splits['test_idx']],
            preprocessor=self.preprocessor,
            augmentation=self.augmentation_val
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader

# ==================== UTILITY FUNCTIONS ====================
def get_class_weights(image_paths: List[str], labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset (inverse frequency weighting)
    """
    from collections import Counter
    
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)

if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    # This would need actual data
    print("Data loading module loaded successfully")
