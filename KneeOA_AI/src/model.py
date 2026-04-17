"""
Deep Learning Model Architecture
Includes EfficientNet, ResNet with multi-task learning support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

# ==================== BASE FEATURE EXTRACTORS ====================
class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction"""
    
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True, freeze: bool = False):
        """
        Args:
            model_name: EfficientNet model name (b0-b7)
            pretrained: Load ImageNet pretrained weights
            freeze: Freeze backbone parameters
        """
        super().__init__()
        
        # Load pretrained model
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown EfficientNet model: {model_name}")
        
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Remove classification head
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        logger.info(f"Loaded {model_name} (pretrained={pretrained}, frozen={freeze})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        return self.backbone(x)

class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction"""
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True, freeze: bool = False):
        """
        Args:
            model_name: ResNet model name (resnet50, resnet101, resnet152)
            pretrained: Load ImageNet pretrained weights
            freeze: Freeze backbone parameters
        """
        super().__init__()
        
        # Load pretrained model
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Remove classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        logger.info(f"Loaded {model_name} (pretrained={pretrained}, frozen={freeze})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        return self.backbone(x)

# ==================== MULTI-TASK LEARNING HEADS ====================
class ClassificationHead(nn.Module):
    """Classification head for KL grading"""
    
    def __init__(self, feature_dim: int, num_classes: int, dropout_rate: float = 0.3):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of KL grades (5)
            dropout_rate: Dropout rate
        """
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
        """
        Args:
            x: Features (batch_size, feature_dim)
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.classifier(x)

class RegressionHead(nn.Module):
    """Regression head for T-score prediction (optional multi-task learning)"""
    
    def __init__(self, feature_dim: int, num_outputs: int = 1, dropout_rate: float = 0.3):
        """
        Args:
            feature_dim: Input feature dimension
            num_outputs: Number of regression outputs (1 for T-score)
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (batch_size, feature_dim)
        Returns:
            T-score predictions (batch_size, num_outputs)
        """
        return self.regressor(x)

# ==================== MAIN MODEL ====================
class KneeOADetectionModel(nn.Module):
    """
    Multi-task learning model for:
    1. Osteoarthritis severity classification (KL grading)
    2. Optional T-score regression for bone health
    """
    
    def __init__(self,
                 backbone: str = "efficientnet_b0",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 num_classes: int = 5,
                 include_regression: bool = True,
                 dropout_rate: float = 0.3):
        """
        Args:
            backbone: Backbone architecture name
            pretrained: Load pretrained weights
            freeze_backbone: Freeze backbone parameters
            num_classes: Number of KL grades
            include_regression: Include T-score regression head
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.include_regression = include_regression
        
        # Feature extractor (backbone)
        if "efficientnet" in backbone.lower():
            self.backbone = EfficientNetBackbone(backbone, pretrained, freeze_backbone)
        elif "resnet" in backbone.lower():
            self.backbone = ResNetBackbone(backbone, pretrained, freeze_backbone)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        feature_dim = self.backbone.feature_dim
        
        # Classification head (KL grading)
        self.classification_head = ClassificationHead(feature_dim, num_classes, dropout_rate)
        
        # Optional regression head (T-score)
        if include_regression:
            self.regression_head = RegressionHead(feature_dim, num_outputs=1, dropout_rate=dropout_rate)
        else:
            self.regression_head = None
        
        logger.info(f"Model created: {backbone} -> Classification({num_classes}) + Regression({include_regression})")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, H, W)
        
        Returns:
            If training: (classification_logits, regression_output) if regression enabled
            If inference: classification_logits
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification branch
        class_logits = self.classification_head(features)
        
        # Regression branch (optional)
        if self.include_regression and self.regression_head is not None:
            t_scores = self.regression_head(features)
            return class_logits, t_scores
        
        return class_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features (useful for Grad-CAM)"""
        return self.backbone(x)

# ==================== LOSS FUNCTIONS ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor in range (0,1)
            gamma: Focus parameter for modulating loss
            reduction: 'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss
        """
        ce = F.cross_entropy(inputs, targets, reduction='none')
        probs = torch.exp(-ce)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self,
                 classification_weight: float = 1.0,
                 regression_weight: float = 0.5,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            use_focal_loss: Use Focal Loss for classifications
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        if use_focal_loss:
            self.classification_loss = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.regression_loss = nn.MSELoss()
    
    def forward(self,
                class_logits: torch.Tensor,
                targets: torch.Tensor,
                t_scores: Optional[torch.Tensor] = None,
                t_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            class_logits: Classification logits (batch_size, num_classes)
            targets: Classification targets (batch_size,)
            t_scores: Regression predictions (batch_size, 1)
            t_targets: Regression targets (batch_size, 1)
        
        Returns:
            Combined loss
        """
        class_loss = self.classification_loss(class_logits, targets)
        
        if t_scores is not None and t_targets is not None:
            reg_loss = self.regression_loss(t_scores, t_targets)
            total_loss = (self.classification_weight * class_loss + 
                         self.regression_weight * reg_loss)
            return total_loss
        
        return class_loss

# ==================== UTILITY FUNCTIONS ====================
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def create_model(model_config) -> KneeOADetectionModel:
    """Factory function to create model from config"""
    model = KneeOADetectionModel(
        backbone=model_config.backbone,
        pretrained=model_config.pretrained,
        freeze_backbone=model_config.freeze_backbone,
        num_classes=model_config.num_classes,
        include_regression=model_config.include_t_score_head,
        dropout_rate=model_config.dropout_rate
    )
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    model = KneeOADetectionModel(
        backbone="efficientnet_b0",
        num_classes=5,
        include_regression=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    print(f"Classification output shape: {outputs[0].shape}")
    print(f"Regression output shape: {outputs[1].shape}")
