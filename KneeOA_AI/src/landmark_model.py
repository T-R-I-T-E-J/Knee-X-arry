import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class KneeLandmarkDetector(nn.Module):
    """
    A Deep Learning Keypoint Regression Model for Knee Joint detection.
    Pre-trained on ImageNet, fine-tuned to predict 6 continuous coordinate values:
    [x_medial, y_medial, x_center, y_center, x_lateral, y_lateral]
    
    Coordinates should ideally be normalized between [0, 1] relative to image dimensions.
    """
    def __init__(self, pretrained=True):
        super(KneeLandmarkDetector, self).__init__()
        
        # Load pre-trained ResNet50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Freeze early layers to prevent overfitting on small clinical datasets
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        # Replace the final classification layer with a Coordinate Regression head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            # 6 output neurons: 3 points * 2 coordinates (x, y)
            nn.Linear(128, 6),
            # Sigmoid guarantees outputs are purely bounded [0, 1] (normalized screen coordinates)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input: Tensor[Batch, Channels, Height, Width]
        Output: Tensor[Batch, 6] -> Array of normalized [0, 1] coordinates
        """
        return self.backbone(x)

if __name__ == "__main__":
    # Quick sanity validation
    model = KneeLandmarkDetector()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Test passed! Model output shape:", output.shape)
    print("Sample generated coordinate prediction (normalized):", output.detach().numpy())
