"""
Evaluation Metrics and Explainability (Grad-CAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, classification_report)
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==================== EVALUATION METRICS ====================
class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.labels.extend(targets.cpu().numpy().tolist())
        self.probabilities.extend(probs.detach().cpu().numpy())
    
    def calculate(self) -> Dict[str, float]:
        """Calculate all metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == i).mean()
                metrics[f'accuracy_{class_name}'] = class_acc
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.labels, self.predictions, labels=range(self.num_classes))
    
    def get_classification_report(self) -> str:
        """Get classification report"""
        return classification_report(self.labels, self.predictions, 
                                    target_names=self.class_names,
                                    zero_division=0)

# ==================== GRAD-CAM VISUALIZATION ====================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for explainability
    Visualizes which regions of the X-ray are important for classification
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Args:
            model: PyTorch model
            target_layer_name: Name of the layer to compute gradients for
        """
        self.model = model
        self.target_layer_name = target_layer_name
        
        # Register hooks
        self.activation = None
        self.gradient = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activation = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        # Find and register hooks for all layers
        for name, module in self.model.named_modules():
            if name == self.target_layer_name or name.endswith('.' + self.target_layer_name):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
    
    def compute_gradcam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute Grad-CAM for an image
        
        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class index
            
        Returns:
            CAM (H, W) - attention map
        """
        self.model.eval()
        
        # Forward pass
        image_input = image.requires_grad_(True)
        with torch.enable_grad():
            outputs = self.model(image_input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take classification output
            
            score = outputs[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        if self.gradient is not None and self.activation is not None:
            # Compute weights
            weights = self.gradient[0].mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
            
            # Weighted sum
            cam = (weights * self.activation[0]).sum(dim=0)  # (H, W)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.cpu().detach().numpy()
            cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam_normalized
        
        return None

class ExplainabilityVisualizer:
    """Visualize model predictions with explanations"""
    
    def __init__(self, class_names: List[str], output_dir: Path = None):
        """
        Args:
            class_names: List of class names
            output_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_gradcam(self,
                         image: np.ndarray,
                         gradcam: np.ndarray,
                         true_label: int,
                         pred_label: int,
                         confidence: float,
                         save_name: str = None) -> np.ndarray:
        """
        Visualize Grad-CAM overlay on image
        
        Args:
            image: Original X-ray image (H, W, 3) in [0, 1]
            gradcam: Grad-CAM attention map (H, W)
            true_label: True class index
            pred_label: Predicted class index
            confidence: Confidence score
            save_name: Name to save figure
            
        Returns:
            Visualization image
        """
        # Normalize image to [0, 255]
        image_uint8 = (image * 255).astype(np.uint8)
        if image_uint8.shape[2] == 3:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        
        # Resize Grad-CAM to match image
        h, w = image_uint8.shape[:2]
        gradcam_resized = cv2.resize(gradcam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((gradcam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend
        overlay = cv2.addWeighted(image_uint8, 0.6, heatmap, 0.4, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_uint8, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        axes[1].imshow(gradcam_resized, cmap='hot')
        axes[1].set_title('Grad-CAM Attention')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add prediction info
        true_class = self.class_names[true_label]
        pred_class = self.class_names[pred_label]
        title = f"True: {true_class} | Pred: {pred_class} (Conf: {confidence:.2f})"
        fig.suptitle(title)
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.close()
        
        return overlay
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                             class_names: List[str],
                             save_path: Path = None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_metrics(metrics_dict: Dict[str, List[float]], save_path: Path = None):
        """Plot training history metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ax.plot(values, marker='o')
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        
        plt.close()

# ==================== COMPREHENSIVE EVALUATION ====================
def evaluate_model(model: nn.Module,
                  test_loader,
                  device: str,
                  class_names: List[str],
                  use_gradcam: bool = True,
                  output_dir: Path = None) -> Dict:
    """
    Comprehensive evaluation including metrics and visualizations
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
        use_gradcam: Generate Grad-CAM visualizations
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary with metrics and results
    """
    model.eval()
    metrics_calc = MetricsCalculator(len(class_names), class_names)
    visualizer = ExplainabilityVisualizer(class_names, output_dir)
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            metrics_calc.update(outputs, labels)
            
            # Collect batch results
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(images)):
                all_results.append({
                    'image': images[i].cpu().numpy(),
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'confidence': probs[i, preds[i]].item(),
                    'probabilities': probs[i].cpu().numpy()
                })
    
    # Calculate metrics
    metrics = metrics_calc.calculate()
    cm = metrics_calc.get_confusion_matrix()
    class_report = metrics_calc.get_classification_report()
    
    logger.info("\n" + class_report)
    
    # Save visualizations
    if output_dir:
        visualizer.plot_confusion_matrix(cm, class_names, 
                                        Path(output_dir) / "confusion_matrix.png")
    
    # Grad-CAM visualization for sample images
    if use_gradcam:
        try:
            gradcam = GradCAM(model, "features")
            
            for idx in range(min(5, len(all_results))):  # Visualize first 5 samples
                result = all_results[idx]
                image = torch.from_numpy(result['image']).unsqueeze(0).to(device)
                cam = gradcam.compute_gradcam(image, result['pred_label'])
                
                if cam is not None:
                    # Normalize image for visualization
                    image_vis = result['image'].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                    visualizer.visualize_gradcam(
                        image_vis,
                        cam,
                        result['true_label'],
                        result['pred_label'],
                        result['confidence'],
                        save_name=f"gradcam_sample_{idx}.png"
                    )
        except Exception as e:
            logger.warning(f"Failed to generate Grad-CAM visualizations: {e}")
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'detailed_results': all_results
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module loaded successfully")
