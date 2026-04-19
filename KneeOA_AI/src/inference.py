"""
Inference Engine for Knee OA Prediction.
Provides prediction with clinical parameter interpretation and severity descriptions.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Optional
from src.data_loader import ImagePreprocessor

class InferenceEngine:
    """Wraps a trained model for easy inference."""
    
    def __init__(self, model, device: str = "cpu"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = ImagePreprocessor()
        self.class_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """Predicts KL grade and clinical parameters for a single image."""
        # Preprocess
        img_np = self.preprocessor.preprocess(image_path)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        
        # Inference
        outputs = self.model(img_tensor)
        if isinstance(outputs, tuple):
            logits, params = outputs
            params = params.squeeze().cpu().numpy()
        else:
            logits, params = outputs, None
            
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = np.argmax(probs)
        
        result = {
            "grade": pred_idx,
            "label": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": probs.tolist(),
        }
        
        if params is not None:
            param_names = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
            result["clinical_params"] = {name: float(val) for name, val in zip(param_names, params)}
            result["descriptions"] = self._get_descriptions(result["clinical_params"])
            
        return result

    def _get_descriptions(self, params: Dict[str, float]) -> Dict[str, str]:
        """Maps normalized parameter values to human-readable clinical descriptions."""
        desc = {}
        
        # JSW (Inverse relationship)
        jsw = params["JSW"]
        if jsw > 0.8: desc["JSW"] = "Healthy joint space"
        elif jsw > 0.5: desc["JSW"] = "Mild narrowing"
        elif jsw > 0.3: desc["JSW"] = "Moderate narrowing"
        else: desc["JSW"] = "Severe joint space loss"
        
        # Osteophytes
        ost = params["Osteophytes"]
        if ost < 0.2: desc["Osteophytes"] = "No significant osteophytes"
        elif ost < 0.5: desc["Osteophytes"] = "Definite small osteophytes"
        else: desc["Osteophytes"] = "Large, prominent osteophytes"
        
        # Sclerosis
        scl = params["Sclerosis"]
        if scl < 0.3: desc["Sclerosis"] = "Normal subchondral bone density"
        elif scl < 0.6: desc["Sclerosis"] = "Mild subchondral sclerosis"
        else: desc["Sclerosis"] = "Marked sclerosis (bone hardening)"
        
        # Contour
        con = params["Contour"]
        if con < 0.4: desc["Contour"] = "Smooth bone contour"
        else: desc["Contour"] = "Significant bone deformity observed"
        
        return desc
