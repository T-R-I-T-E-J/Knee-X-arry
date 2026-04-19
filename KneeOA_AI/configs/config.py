"""
Configuration for Knee OA Detection System
All hyperparameters and paths in one place.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories
for directory in [MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    # Dataset paths (both datasets are merged for training)
    dataset_paths = {
        "MedicalExpert-I": PROJECT_ROOT.parent / "MedicalExpert-I",
        "MedicalExpert-II": PROJECT_ROOT.parent / "MedicalExpert-II"
    }
    
    # KL Grading labels
    class_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
    num_classes = 5
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean = [0.485, 0.456, 0.406]  # ImageNet
    normalize_std = [0.229, 0.224, 0.225]
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Augmentation
    augmentation_enabled: bool = True
    augmentation_prob: float = 0.5
    rotation_degrees: int = 15
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # DataLoader
    batch_size: int = 32
    num_workers: int = 0   # 0 for Windows compatibility
    pin_memory: bool = False
    shuffle_train: bool = True

# ==================== MODEL CONFIGURATION ====================
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "efficientnet_b0"  # Options: efficientnet_b0, resnet50, resnet101
    pretrained: bool = True
    freeze_backbone: bool = False
    
    num_classes: int = 5
    include_clinical_params: bool = True
    num_clinical_params: int = 4
    clinical_param_names = ["jsw", "osteophyte_score", "sclerosis_score", "contour_score"]
    dropout_rate: float = 0.3
    
    feature_dim: int = 1280  # EfficientNet-B0
    
    # Multi-task loss weights
    classification_weight: float = 1.0
    clinical_params_weight: float = 0.3

# ==================== TRAINING CONFIGURATION ====================
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    optimizer: str = "adam"
    
    # Loss
    use_focal_loss: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"
    
    # Checkpointing
    checkpoint_dir: Path = MODELS_DIR
    save_best_only: bool = True
    save_frequency: int = 5
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = True

# ==================== EVALUATION CONFIGURATION ====================
@dataclass
class EvalConfig:
    """Evaluation and explainability configuration"""
    metrics = ["accuracy", "f1_weighted", "precision", "recall", "confusion_matrix"]
    use_grad_cam: bool = True
    grad_cam_target_layer: str = "features"
    num_visualization_samples: int = 5
    confidence_threshold: float = 0.7
    generate_classification_report: bool = True
    generate_confusion_matrix: bool = True

# ==================== HARDWARE CONFIGURATION ====================
@dataclass
class HardwareConfig:
    """Hardware and optimization settings"""
    use_gpu: bool = True
    gpu_id: int = 0
    device: str = "cuda"
    use_gradient_checkpointing: bool = False
    num_threads: int = 8
    benchmark_cudnn: bool = True

# ==================== LOGGING CONFIGURATION ====================
@dataclass
class LoggingConfig:
    """Logging settings"""
    log_level: str = "INFO"
    log_file: Path = OUTPUTS_DIR / "training.log"
    tensorboard_enabled: bool = True
    tensorboard_dir: Path = OUTPUTS_DIR / "tensorboard_logs"

# ==================== GLOBAL INSTANCES ====================
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvalConfig()
hardware_config = HardwareConfig()
logging_config = LoggingConfig()

if __name__ == "__main__":
    print("Configuration Loaded Successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Dir:   {MODELS_DIR}")
    print(f"Outputs Dir:  {OUTPUTS_DIR}")
    print(f"Datasets:     {list(data_config.dataset_paths.keys())}")
    print(f"Clinical:     {model_config.clinical_param_names}")
