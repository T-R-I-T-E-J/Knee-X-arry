"""
Robust Data Loading and Preprocessing for Knee X-Ray analysis.
Supports dual MedicalExpert datasets and Windows-safe multiprocessing.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

from src.preprocessing import AutoCutter

class ImagePreprocessor:
    """Handles medical image loading, enhancement, and resizing."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize_mean: List[float] = None,
                 normalize_std: List[float] = None,
                 apply_clahe: bool = True,
                 use_autocutter: bool = True):
        self.target_size = target_size
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.apply_clahe = apply_clahe
        self.autocutter = AutoCutter() if use_autocutter else None
        self._clahe_clip = 2.0
        self._clahe_grid = (8, 8)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """Complete pipeline: Load -> AutoCrop -> CLAHE -> Resize -> Normalize."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load: {image_path}")

        # 1. NEW: Auto-Cutter (Focus on anatomical center)
        if self.autocutter:
            # Convert to RGB for AutoCutter (it expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_pil = Image.fromarray(image_rgb)
            cropped_pil = self.autocutter.crop(image_pil)
            image = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2GRAY)
            
        # 2. CLAHE (Contrast Enhancement)
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self._clahe_clip, tileGridSize=self._clahe_grid)
            image = clahe.apply(image)
            
        # 3. Resize with padding
        h, w = image.shape[:2]
        aspect = w / h
        new_w, new_h = (self.target_size[1], int(self.target_size[1]/aspect)) if aspect > 1 else (int(self.target_size[0]*aspect), self.target_size[0])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        top = (self.target_size[0] - new_h) // 2
        bottom = self.target_size[0] - new_h - top
        left = (self.target_size[1] - new_w) // 2
        right = self.target_size[1] - new_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # 4. Normalize
        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=-1) # To RGB
        return image

class XRayAugmentation:
    """Medical-specific augmentations."""
    
    def __init__(self, enabled: bool = True, prob: float = 0.5):
        self.enabled = enabled
        if enabled:
            self.transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(15)], p=prob),
                transforms.RandomApply([transforms.ColorJitter(0.1, 0.1)], p=prob),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=prob*0.5),
            ])
        else:
            self.transform = nn.Identity()
            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

class KneeXRayDataset(Dataset):
    """Dataset for KL grading classification."""
    
    def __init__(self, paths: List[str], labels: List[int], preprocessor: ImagePreprocessor, augmenter: Optional[XRayAugmentation] = None):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        image = self.preprocessor.preprocess(self.paths[idx])
        label = self.labels[idx]
        image = self.to_tensor(image)
        if self.augmenter:
            image = self.augmenter(image)
        return image, label

class KneeXRayDataLoader:
    """Orchestrates multi-dataset loading and splitting."""
    
    def __init__(self, roots: Dict[str, Path], data_config):
        self.roots = roots
        self.config = data_config
        self.preprocessor = ImagePreprocessor(target_size=data_config.image_size)
        self.augmenter = XRayAugmentation(enabled=data_config.augmentation_enabled, prob=data_config.augmentation_prob)

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        paths, labels = self._collect_data()
        
        # Split
        idx = np.arange(len(paths))
        tr_val_idx, te_idx = train_test_split(idx, test_size=self.config.test_ratio, random_state=self.config.random_seed, stratify=labels)
        tr_idx, val_idx = train_test_split(tr_val_idx, test_size=self.config.val_ratio / (1-self.config.test_ratio), random_state=self.config.random_seed, stratify=[labels[i] for i in tr_val_idx])
        
        # Create loader helper
        def _get_loader(indices, augment=False):
            ds = KneeXRayDataset(
                paths=[paths[i] for i in indices],
                labels=[labels[i] for i in indices],
                preprocessor=self.preprocessor,
                augmenter=self.augmenter if augment else None
            )
            return DataLoader(
                ds, batch_size=self.config.batch_size, 
                num_workers=self.config.num_workers, 
                pin_memory=self.config.pin_memory, 
                shuffle=augment
            )
            
        return _get_loader(tr_idx, True), _get_loader(val_idx), _get_loader(te_idx)

    def _collect_data(self) -> Tuple[List[str], List[int]]:
        paths, labels = [], []
        folder_map = {0: ["0Normal", "0"], 1: ["1Doubtful", "1"], 2: ["2Mild", "2"], 3: ["3Moderate", "3"], 4: ["4Severe", "4"]}
        
        for name, root in self.roots.items():
            dataset_path = Path(root)
            if not dataset_path.exists(): continue
            
            for grade, folders in folder_map.items():
                for folder in folders:
                    dir_path = dataset_path / folder
                    if dir_path.exists():
                        files = [str(f) for f in dir_path.glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
                        paths.extend(files)
                        labels.extend([grade] * len(files))
                        logger.info(f"Loaded {len(files)} {name} images for grade {grade}")
                        break
        return paths, labels
