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
    
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("="*60)
    logger.info("Classification Accuracy: {:.2%}".format(results['metrics']['accuracy']))
    logger.info("Weighted F1 Score:       {:.4f}".format(results['metrics']['f1_weighted']))
    
    if 'clinical_stats' in results:
        def get_mark(val, inverse=False):
            # JSW is inverse (High is good, Low is bad)
            if inverse:
                if val > 0.7: return "Healthy"
                if val > 0.5: return "Mild"
                if val > 0.3: return "Moderate"
                return "Critical"
            else:
                if val < 0.3: return "Minimal"
                if val < 0.5: return "Noticeable"
                if val < 0.7: return "Significant"
                return "High/Severe"

        logger.info("\nClinical Marker Spearman Correlations (vs KL Grade):")
        for key, val in results['metrics'].items():
            if key.startswith('spearman_'):
                param_name = key.replace('spearman_', '').capitalize()
                logger.info(f"    - {param_name:12}: {val:.4f}")
        
        logger.info("\nMean Marker Scores per KL Grade (with Medical Marks):")
        param_names = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
        header = "    Grade | " + " | ".join([f"{p:15}" for p in param_names])
        logger.info(header)
        logger.info("    " + "-" * len(header))
        for g in range(5):
            row = f"    {g:5} | "
            val_strs = []
            for p in param_names:
                v = results['clinical_stats'][p][g]
                mark = get_mark(v, inverse=(p == "JSW"))
                val_strs.append(f"{v:.2f} ({mark:^8})")
            row += " | ".join(val_strs)
            logger.info(row)

    logger.info("\nEvaluation Complete. Best model saved in models/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    training_config.num_epochs = args.epochs
    data_config.batch_size = args.batch_size
    
    main(args)
