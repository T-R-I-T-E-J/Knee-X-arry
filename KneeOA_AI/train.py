"""
Main Training Script - Orchestrates the entire training pipeline
"""

import sys
import os
import logging
import argparse
from pathlib import Path
import json
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs.config import (
    data_config, model_config, training_config, eval_config,
    hardware_config, logging_config, DATA_DIR, MODELS_DIR, OUTPUTS_DIR
)
from src.data_loader import KneeXRayDataLoader, get_class_weights
from src.model import create_model, MultiTaskLoss
from src.trainer import setup_training, save_training_history
from src.evaluation import evaluate_model

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Configure logging"""
    logging_config.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, logging_config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_config.log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== MAIN TRAINING PIPELINE ====================
def main(args):
    """Main training pipeline"""
    
    logger.info("="*80)
    logger.info("Knee OA Detection System - Training Pipeline")
    logger.info("="*80)
    
    # ==================== CONFIGURATION ====================
    logger.info("\n1. Loading Configuration...")
    
    # Update dataset paths if provided
    if args.dataset_path:
        data_config.dataset_paths = {
            "data": Path(args.dataset_path)
        }
    
    logger.info(f"Dataset paths: {data_config.dataset_paths}")
    logger.info(f"Model backbone: {model_config.backbone}")
    logger.info(f"Number of epochs: {training_config.num_epochs}")
    logger.info(f"Device: {hardware_config.device}")
    
    # ==================== DATA LOADING ====================
    logger.info("\n2. Loading and Processing Data...")
    
    data_loader = KneeXRayDataLoader(
        dataset_roots={name: Path(path) for name, path in data_config.dataset_paths.items()},
        data_config=data_config,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers
    )
    
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    logger.info(f"Data loading complete")
    
    # ==================== MODEL CREATION ====================
    logger.info("\n3. Creating Model...")
    
    model = create_model(model_config)
    model = model.to(hardware_config.device)
    
    # ==================== LOSS FUNCTION ====================
    logger.info("\n4. Setting up Loss Functions...")
    
    criterion = MultiTaskLoss(
        classification_weight=model_config.classification_weight,
        regression_weight=model_config.regression_weight,
        use_focal_loss=training_config.use_focal_loss,
        focal_alpha=training_config.focal_loss_alpha,
        focal_gamma=training_config.focal_loss_gamma
    )
    
    # ==================== TRAINING SETUP ====================
    logger.info("\n5. Setting up Training...")
    
    trainer = setup_training(model, train_loader, val_loader, training_config)
    
    # ==================== TRAINING ====================
    logger.info("\n6. Starting Training...")
    
    history = trainer.train()
    
    # Save training history
    history_path = OUTPUTS_DIR / "training_history.json"
    save_training_history(history, history_path)
    
    # ==================== EVALUATION ====================
    logger.info("\n7. Evaluating Model...")
    
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=hardware_config.device,
        class_names=data_config.class_names,
        use_gradcam=eval_config.use_grad_cam,
        output_dir=OUTPUTS_DIR / "visualizations"
    )
    
    logger.info("\nMetrics:")
    for metric_name, metric_value in eval_results['metrics'].items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # ==================== SAVE RESULTS ====================
    logger.info("\n8. Saving Results...")
    
    # Save model
    model_path = MODELS_DIR / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = OUTPUTS_DIR / "evaluation_metrics.json"
    metrics_to_save = {k: v for k, v in eval_results['metrics'].items() if isinstance(v, (int, float))}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save configuration
    config_path = OUTPUTS_DIR / "config.json"
    config_dict = {
        'data_config': {
            'class_names': data_config.class_names,
            'num_classes': data_config.num_classes,
            'image_size': data_config.image_size,
            'batch_size': data_config.batch_size,
        },
        'model_config': {
            'backbone': model_config.backbone,
            'num_classes': model_config.num_classes,
            'include_t_score_head': model_config.include_t_score_head,
        },
        'training_config': {
            'num_epochs': training_config.num_epochs,
            'learning_rate': training_config.learning_rate,
            'optimizer': training_config.optimizer,
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best model saved to: {model_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Evaluation metrics saved to: {metrics_path}")
    logger.info(f"Visualizations saved to: {OUTPUTS_DIR / 'visualizations'}")

# ==================== COMMAND LINE INTERFACE ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Knee OA Detection Model"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset root directory"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b1", "resnet50", "resnet101"],
        help="Backbone architecture"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--disable-grad-cam",
        action="store_true",
        help="Disable Grad-CAM visualization"
    )
    
    args = parser.parse_args()
    
    # Update configuration from command line arguments
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.batch_size:
        data_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.backbone:
        model_config.backbone = args.backbone
    if args.device:
        hardware_config.device = args.device
    if args.disable_grad_cam:
        eval_config.use_grad_cam = False
    
    main(args)
