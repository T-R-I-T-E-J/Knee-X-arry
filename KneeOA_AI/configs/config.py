"""
Configuration file for Knee OA Analysis System
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    # Dataset structure
    dataset_paths = {
        "MedicalExpert-I": "/path/to/MedicalExpert-I",
        "MedicalExpert-II": "/path/to/MedicalExpert-II"
    }
    
    # Class labels (KL Grading System)
    class_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
    num_classes = 5
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)  # For EfficientNet-B0
    normalize_mean = [0.485, 0.456, 0.406]  # ImageNet standards
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
    
    # Batch processing
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True

# ==================== MODEL CONFIGURATION ====================
@dataclass
class ModelConfig:
    """Deep Learning Model Configuration"""
    # Architecture choices
    backbone: str = "efficientnet_b0"  # Options: efficientnet_b0, resnet50, resnet101
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Model components
    num_classes: int = 5  # KL grades
    include_t_score_head: bool = True  # Multi-task learning
    dropout_rate: float = 0.3
    
    # Feature extraction
    feature_dim: int = 1280  # EfficientNet-B0 output dimension
    
    # Task weights for multi-task learning
    classification_weight: float = 1.0
    regression_weight: float = 0.5

# ==================== TRAINING CONFIGURATION ====================
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Training schedule
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # Options: cosine, step, exponential
    
    # Optimizer
    optimizer: str = "adam"  # Options: adam, sgd, adamw
    
    # Loss functions
    use_focal_loss: bool = False  # For class imbalance
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Regularization
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"  # Options: val_loss, val_accuracy
    
    # Checkpointing
    checkpoint_dir: Path = MODELS_DIR
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs
    
    # Device
    device: str = "cuda"  # Options: cuda, cpu
    mixed_precision: bool = True  # Automatic Mixed Precision (AMP)

# ==================== EVALUATION CONFIGURATION ====================
@dataclass
class EvalConfig:
    """Evaluation and metrics configuration"""
    # Metrics
    metrics = ["accuracy", "f1_weighted", "precision", "recall", "confusion_matrix"]
    
    # Explainability
    use_grad_cam: bool = True
    grad_cam_target_layer: str = "features"  # Layer to visualize
    num_visualization_samples: int = 5  # Per class
    
    # Thresholds
    confidence_threshold: float = 0.7
    
    # Output
    generate_classification_report: bool = True
    generate_confusion_matrix: bool = True

# ==================== HARDWARE & OPTIMIZATION ====================
@dataclass
class HardwareConfig:
    """Hardware and optimization settings"""
    # GPU
    use_gpu: bool = True
    gpu_id: int = 0
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 4
    
    # Performance
    num_threads: int = 8
    benchmark_cudnn: bool = True

# ==================== DEPLOYMENT CONFIGURATION ====================
@dataclass
class DeploymentConfig:
    """Deployment and inference configuration"""
    # UI Framework
    framework: str = "gradio"  # Options: gradio, flask
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 7860  # Gradio default
    
    # Inference
    batch_inference_enabled: bool = True
    max_batch_size: int = 8
    
    # Preprocessing
    preprocessing_pipeline: str = "default"

# ==================== LOGGING & MONITORING ====================
@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = "INFO"
    log_file: Path = OUTPUTS_DIR / "training.log"
    tensorboard_enabled: bool = True
    tensorboard_dir: Path = OUTPUTS_DIR / "tensorboard_logs"
    wandb_enabled: bool = False  # Weights & Biases integration

# ==================== ENSEMBLE CONFIGURATION ====================
@dataclass
class EnsembleConfig:
    """Configuration for handling multiple expert annotations"""
    # Consensus methods: "majority_vote", "weighted_vote", "average_confidence"
    consensus_method: str = "majority_vote"
    confidence_weighted: bool = True
    disagreement_threshold: float = 0.5

# ==================== INSTANTIATE GLOBAL CONFIGS ====================
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvalConfig()
hardware_config = HardwareConfig()
deployment_config = DeploymentConfig()
logging_config = LoggingConfig()
ensemble_config = EnsembleConfig()

# ==================== HELPER FUNCTIONS ====================
def get_class_weights(dataset_dir: Path, data_config: DataConfig) -> dict:
    """
    Calculate class weights for imbalanced dataset
    """
    from collections import Counter
    import torch
    
    class_counts = Counter()
    for class_name in data_config.class_names:
        class_path = dataset_dir / class_name
        if class_path.exists():
            class_counts[class_name] = len(list(class_path.glob("*")))
    
    total = sum(class_counts.values())
    weights = {}
    for class_name, count in class_counts.items():
        weights[class_name] = total / (len(class_counts) * count) if count > 0 else 1.0
    
    return weights

if __name__ == "__main__":
    print("Configuration Loaded Successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
