"""
Knee OA Detection System - Main Training Script.
Orchestrates data loading, multi-task model creation, and training.
"""

import sys
import argparse
import logging
import torch
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs.config import data_config, model_config, training_config, hardware_config, OUTPUTS_DIR
from src.data_loader import KneeXRayDataLoader
from src.model import create_model, MultiTaskLoss
from src.trainer import setup_training, save_training_history
from src.evaluation import evaluate_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main(args):
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("Knee OA Detection AI — Training v2.0")
    logger.info("="*60)

    # 1. Load Data
    loader = KneeXRayDataLoader(
        roots={name: Path(p) for name, p in data_config.dataset_paths.items()},
        data_config=data_config
    )
    train_dl, val_dl, test_dl = loader.create_dataloaders()
    
    # 2. Build Model
    model = create_model(model_config)
    
    # 3. Loss & Optimizer
    criterion = MultiTaskLoss(
        classification_weight=model_config.classification_weight,
        clinical_params_weight=model_config.clinical_params_weight
    )
    
    trainer = setup_training(model, train_dl, val_dl, training_config, criterion, args.device)
    
    # 4. Train
    history = trainer.train()
    save_training_history(history, OUTPUTS_DIR / "training_history.json")
    
    # 5. Evaluate
    results = evaluate_model(model, test_dl, args.device, data_config.class_names, OUTPUTS_DIR)
    
    logger.info("\nFinal Accuracy: {:.2%}".format(results['metrics']['accuracy']))
    logger.info("Evaluation Complete. Best model saved in models/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    training_config.num_epochs = args.epochs
    data_config.batch_size = args.batch_size
    
    main(args)
