"""
Deep Learning Model Architecture for Knee Osteoarthritis Detection.
Features:
- EfficientNet and ResNet backbones
- Multi-task learning: KL Grading (Classification) + Clinical Parameter Extraction
- Ordinal Consistency Loss for label-free clinical feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List, Dict
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for high-performance feature extraction."""
    
    def __init__(self, model_name: str = "efficientnet_b0", weights: str = "DEFAULT", freeze: bool = False):
        super().__init__()
        
        # Mapping for easier model selection
        model_funcs = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
        }
        
        if model_name not in model_funcs:
            raise ValueError(f"Unsupported EfficientNet: {model_name}")
            
        # Use the modern weights API
        self.backbone = model_funcs[model_name](weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Convert classifier to identity to keep pooling output
        self.backbone.classifier = nn.Identity()
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        logger.info(f"Initialized {model_name} (weights={weights}, frozen={freeze})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ResNetBackbone(nn.Module):
    """ResNet backbone for robust feature extraction."""
    
    def __init__(self, model_name: str = "resnet50", weights: str = "DEFAULT", freeze: bool = False):
        super().__init__()
        
        model_funcs = {
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }
        
        if model_name not in model_funcs:
            raise ValueError(f"Unsupported ResNet: {model_name}")
            
        self.backbone = model_funcs[model_name](weights=weights)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        logger.info(f"Initialized {model_name} (weights={weights}, frozen={freeze})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ClassificationHead(nn.Module):
    """Dense head for KL Grading classification."""
    
    def __init__(self, feature_dim: int, num_classes: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class ClinicalParameterHead(nn.Module):
    """
    Predicts 4 normalized radiographic markers [0, 1].
    Learned via Ordinal Consistency with KL grades.
    """
    
    def __init__(self, feature_dim: int, num_params: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_params),
            nn.Sigmoid() 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class KneeOADetectionModel(nn.Module):
    """Multi-task model combining classification and clinical parameter extraction."""
    
    def __init__(self,
                 backbone: str = "efficientnet_b0",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 num_classes: int = 5,
                 include_clinical_params: bool = True,
                 num_clinical_params: int = 4,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        weights = "DEFAULT" if pretrained else None
        
        # Select backbone
        if "efficientnet" in backbone.lower():
            self.feature_extractor = EfficientNetBackbone(backbone, weights, freeze_backbone)
        elif "resnet" in backbone.lower():
            self.feature_extractor = ResNetBackbone(backbone, weights, freeze_backbone)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        self.feature_dim = self.feature_extractor.feature_dim
        
        # Heads
        self.classifier = ClassificationHead(self.feature_dim, num_classes, dropout_rate)
        
        self.include_clinical_params = include_clinical_params
        if include_clinical_params:
            self.clinical_head = ClinicalParameterHead(self.feature_dim, num_clinical_params, dropout_rate)
        else:
            self.clinical_head = None
            
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        if self.include_clinical_params and self.clinical_head:
            params = self.clinical_head(features)
            return logits, params
            
        return logits

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance in KL grades."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class OrdinalConsistencyLoss(nn.Module):
    """Enforces biological rules: JSW decreases with severity, others increase."""
    
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin
        self.directions = [-1, 1, 1, 1]  # JSW, Osteo, Sclerosis, Contour
    
    def forward(self, params: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
        batch_size = params.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=params.device, requires_grad=True)
            
        total_loss = torch.tensor(0.0, device=params.device, requires_grad=True)
        pairs = 0
        
        # Efficient vector-based comparison would be better, but loop is clearer for 4 params
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if grades[i] == grades[j]: continue
                
                # Determine relative severity
                high_idx, low_idx = (i, j) if grades[i] > grades[j] else (j, i)
                
                for p, direct in enumerate(self.directions):
                    high_val = params[high_idx, p]
                    low_val = params[low_idx, p]
                    
                    if direct == -1: # Should decrease (high severity = low value)
                        violation = high_val - low_val + self.margin
                    else: # Should increase (high severity = high value)
                        violation = low_val - high_val + self.margin
                        
                    total_loss = total_loss + F.relu(violation)
                pairs += 1
                
        return total_loss / max(pairs, 1)

class MultiTaskLoss(nn.Module):
    """Combines Classification and Ordinal Consistency losses."""
    
    def __init__(self, 
                 classification_weight: float = 1.0, 
                 clinical_params_weight: float = 0.3,
                 use_focal: bool = False):
        super().__init__()
        self.w_class = classification_weight
        self.w_params = clinical_params_weight
        self.class_fn = FocalLoss() if use_focal else nn.CrossEntropyLoss()
        self.param_fn = OrdinalConsistencyLoss()
        
    def forward(self, logits, targets, params=None):
        loss = self.class_fn(logits, targets) * self.w_class
        if params is not None:
            loss += self.param_fn(params, targets) * self.w_params
        return loss

def create_model(config) -> KneeOADetectionModel:
    """Helper to instantiate model from config."""
    model = KneeOADetectionModel(
        backbone=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        num_classes=config.num_classes,
        include_clinical_params=config.include_clinical_params,
        num_clinical_params=config.num_clinical_params,
        dropout_rate=config.dropout_rate
    )
    return model
