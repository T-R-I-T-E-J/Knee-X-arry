"""
Performance Evaluation and Interpretability (Grad-CAM).
Calculates classification metrics and analyzes clinical parameters via Spearman correlation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.stats import spearmanr
from pathlib import Path
from typing import Dict, List, Optional
import logging
import cv2

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names, output_dir=None) -> Dict:
    """Evaluates the model across classification and clinical tasks."""
    model.eval()
    all_preds, all_labels = [], []
    all_params = []
    
    logger.info("Evaluating model on test set...")
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            logits, params = outputs
            all_params.append(params.cpu().numpy())
        else:
            logits = outputs
            
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Classification Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    results = {
        "metrics": {"accuracy": accuracy, "f1_weighted": f1},
        "predictions": all_preds,
        "labels": all_labels
    }
    
    # Clinical Parameter Correlation
    if all_params:
        params_concat = np.concatenate(all_params, axis=0)
        param_names = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
        results["clinical_stats"] = {}
        
        for i, name in enumerate(param_names):
            p_vals = params_concat[:, i]
            corr, _ = spearmanr(all_labels, p_vals)
            results["metrics"][f"spearman_{name.lower()}"] = corr
            
            # Mean analysis per grade
            means = [np.mean(p_vals[np.array(all_labels) == g]) for g in range(5)]
            results["clinical_stats"][name] = means
            
    # Visualize Confusion Matrix
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir / "confusion_matrix.png")
        
    return results

def _plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

class GradCAM:
    """Computes attention maps for model interpretability."""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = None
        
        # Automatic target layer discovery
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            # Fallback to feature_extractor's last layer
            self.target_layer = list(model.feature_extractor.backbone.children())[-2]

        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output): self.activations = output
    def _save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if target_class is None: target_class = torch.argmax(logits, dim=1)
        
        self.model.zero_grad()
        loss = logits[:, target_class].sum()
        loss.backward()
        
        # Pool grads
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
