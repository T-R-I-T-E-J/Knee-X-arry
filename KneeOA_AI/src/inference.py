"""
Inference and Prediction Module
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# ==================== INFERENCE CONFIGURATION ====================
@dataclass
class InferenceConfig:
    """Configuration for inference"""
    model_path: Path = None
    device: str = "cuda"
    confidence_threshold: float = 0.5

# ==================== INFERENCE ENGINE ====================
class InferenceEngine:
    """
    Single and batch inference engine for X-ray predictions
    """
    
    def __init__(self, model, model_path: Path, device: str = "cuda",
                 class_names: List[str] = None, preprocessor = None):
        """
        Args:
            model: Trained PyTorch model
            model_path: Path to model checkpoint
            device: Device to use
            class_names: List of class names
            preprocessor: Image preprocessor
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names or ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
        self.preprocessor = preprocessor
        
        # Load model weights
        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {model_path}")
    
    @torch.no_grad()
    def predict_single(self, image_path: str, return_features: bool = False) -> Dict:
        """
        Single image prediction
        
        Args:
            image_path: Path to X-ray image
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        from src.data_loader import ImagePreprocessor
        
        # Preprocess image
        if self.preprocessor is None:
            preprocessor = ImagePreprocessor()
        else:
            preprocessor = self.preprocessor
        
        image = preprocessor.preprocess(image_path)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
                t_score = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs
                t_score = None
        
        # Process outputs
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        result = {
            'predicted_class': int(predicted_class),
            'predicted_label': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'image_path': str(image_path)
        }
        
        # Add T-score if available
        if t_score is not None:
            result['t_score'] = float(t_score[0].cpu().numpy()[0])
        
        # Add features for Grad-CAM
        if return_features:
            features = self.model.get_features(image_tensor)
            result['features'] = features.cpu().numpy()
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Batch prediction
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {image_path}: {e}")
                results.append({'error': str(e), 'image_path': str(image_path)})
        
        return results

# ==================== PREDICTION FORMATTER ====================
class PredictionFormatter:
    """Format predictions for display and logging"""
    
    @staticmethod
    def format_prediction(prediction: Dict) -> str:
        """Format prediction as readable text"""
        text = f"Prediction: {prediction['predicted_label']}\n"
        text += f"Confidence: {prediction['confidence']:.2%}\n"
        text += f"\nClass Probabilities:\n"
        
        for class_name, prob in prediction['probabilities'].items():
            text += f"  {class_name}: {prob:.2%}\n"
        
        if 't_score' in prediction:
            text += f"\nT-Score: {prediction['t_score']:.2f}\n"
        
        return text
    
    @staticmethod
    def format_batch_predictions(predictions: List[Dict]) -> str:
        """Format batch predictions"""
        text = f"Batch Predictions ({len(predictions)} images)\n"
        text += "="*50 + "\n"
        
        for i, pred in enumerate(predictions):
            text += f"\n{i+1}. {Path(pred['image_path']).name}\n"
            text += f"   {pred['predicted_label']} ({pred['confidence']:.2%})\n"
        
        return text

# ==================== MODEL ADAPTER ====================
def load_model_for_inference(model_path: Path, device: str = "cuda"):
    """
    Load trained model for inference
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use
        
    Returns:
        Loaded model
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from src.model import KneeOADetectionModel
    
    model = KneeOADetectionModel(
        backbone="efficientnet_b0",
        num_classes=5,
        include_regression=True
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

# ==================== REPORT GENERATION ====================
class PredictionReportGenerator:
    """Generate reports from predictions"""
    
    @staticmethod
    def generate_json_report(predictions: List[Dict], output_path: Path):
        """Generate JSON report"""
        report = {
            'total_predictions': len(predictions),
            'timestamp': str(np.datetime64('now')),
            'predictions': []
        }
        
        for pred in predictions:
            if 'error' not in pred:
                report['predictions'].append({
                    'image': pred['image_path'],
                    'prediction': pred['predicted_label'],
                    'confidence': pred['confidence'],
                    'probabilities': pred['probabilities']
                })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
    
    @staticmethod
    def generate_csv_report(predictions: List[Dict], output_path: Path):
        """Generate CSV report"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['image', 'predicted_class', 'confidence', 't_score']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pred in predictions:
                if 'error' not in pred:
                    writer.writerow({
                        'image': Path(pred['image_path']).name,
                        'predicted_class': pred['predicted_label'],
                        'confidence': f"{pred['confidence']:.4f}",
                        't_score': pred.get('t_score', 'N/A')
                    })
        
        logger.info(f"CSV report saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Inference module loaded successfully")
