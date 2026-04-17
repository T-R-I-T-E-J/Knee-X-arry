# Knee Osteoarthritis (OA) AI Detection System

A comprehensive deep learning system for automated analysis of knee X-ray images to detect osteoarthritis severity and assess bone health.

## 🎯 Features

- **Osteoarthritis Classification**: Classify knee OA severity using the Kellgren-Lawrence (KL) grading system (0-4)
- **Bone Health Assessment**: Predict bone density T-scores for osteoporosis risk detection
- **Automatic Feature Learning**: Deep neural network automatically learns radiographic features (JSW, osteophytes, sclerosis)
- **Explainability**: Grad-CAM visualization shows which X-ray regions influence predictions
- **Multi-Expert Handling**: Support for consensus learning from multiple radiologist annotations
- **Production-Ready Deployment**: Web UI via Gradio for easy clinical use

## 📊 Dataset Structure

```
MedicalExpert-I/
├── 0Normal/          (KL Grade 0)
├── 1Doubtful/        (KL Grade 1)
├── 2Mild/            (KL Grade 2)
├── 3Moderate/        (KL Grade 3)
└── 4Severe/          (KL Grade 4)

MedicalExpert-II/
├── 0Normal/
├── 1Doubtful/
├── 2Mild/
├── 3Moderate/
└── 4Severe/
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.0+ |
| **Backbone** | EfficientNet-B0 / ResNet50 |
| **Image Processing** | OpenCV, PIL, Albumentations |
| **Visualization** | Matplotlib, Grad-CAM |
| **Web UI** | Gradio, Flask |
| **ML Metrics** | scikit-learn |
| **Logging** | TensorBoard |
| **Python** | 3.9+ |

## 💻 Hardware Requirements

### Minimum (CPU Training)
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 50 GB

### Recommended (GPU Training)
- GPU: NVIDIA RTX 3060 Ti 8GB+ (or equivalent)
- RAM: 32 GB
- Storage: 100+ GB (SSD)

### Optimal (Production)
- GPU: NVIDIA RTX 4090 24GB+
- RAM: 64 GB
- Storage: 500+ GB (NVMe SSD)

## 📁 Project Structure

```
KneeOA_AI/
├── configs/
│   └── config.py              # Configuration (data, model, training)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── model.py               # Model architecture (EfficientNet, ResNet)
│   ├── trainer.py             # Training loop & callbacks
│   ├── evaluation.py          # Metrics & Grad-CAM
│   └── inference.py           # Prediction engine
├── ui/
│   └── app.py                 # Gradio web interface
├── data/
│   ├── raw/                   # Original X-ray images
│   └── processed/             # Preprocessed images
├── models/
│   └── best_model.pt          # Trained model weights
├── outputs/
│   ├── training_history.json  # Training metrics
│   ├── evaluation_metrics.json # Test metrics
│   └── visualizations/        # Grad-CAM outputs
├── notebooks/                 # Jupyter notebooks for exploration
├── train.py                   # Main training script
├── inference.py               # Standalone inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (if applicable)
cd KneeOA_AI

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your X-ray images into the directory structure above:

```bash
mkdir -p data/raw/MedicalExpert-I/{0Normal,1Doubtful,2Mild,3Moderate,4Severe}
mkdir -p data/raw/MedicalExpert-II/{0Normal,1Doubtful,2Mild,3Moderate,4Severe}

# Copy X-ray images to respective class folders
cp /path/to/normal_xrays/* data/raw/MedicalExpert-I/0Normal/
cp /path/to/doubtful_xrays/* data/raw/MedicalExpert-I/1Doubtful/
# ... continue for other classes
```

### 3. Configure Training

Edit `configs/config.py`:

```python
# Update dataset paths
data_config.dataset_paths = {
    "MedicalExpert-I": "path/to/MedicalExpert-I",
    "MedicalExpert-II": "path/to/MedicalExpert-II"
}

# Set model parameters
model_config.backbone = "efficientnet_b0"  # or "resnet50"
model_config.include_t_score_head = True   # Enable bone health prediction

# Configure training
training_config.num_epochs = 100
training_config.learning_rate = 1e-3
training_config.batch_size = 32
```

### 4. Train Model

```bash
python train.py \
    --dataset-path data/raw \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --backbone efficientnet_b0 \
    --device cuda
```

**Output Files:**
- `models/best_model.pt` - Trained model weights
- `outputs/training_history.json` - Training metrics
- `outputs/evaluation_metrics.json` - Test set performance
- `outputs/visualizations/` - Grad-CAM visualizations

### 5. Deploy Web UI

```bash
# Start Gradio web interface
python ui/app.py

# Access at http://localhost:7860
```

Then:
1. Load your trained model: `models/best_model.pt`
2. Upload knee X-ray images
3. View predictions with attention maps

## 🔧 Advanced Usage

### Training with Custom Configuration

```python
# train_custom.py
from configs.config import model_config, training_config
from src.data_loader import KneeXRayDataLoader
from src.model import create_model
from src.trainer import setup_training

# Customize config
model_config.backbone = "resnet50"
training_config.num_epochs = 200
training_config.early_stopping_patience = 20

# Your training code here
```

### Inference on New Images

```python
from src.inference import InferenceEngine
from src.model import KneeOADetectionModel
import torch

# Load model
model = KneeOADetectionModel()
state_dict = torch.load("models/best_model.pt")
model.load_state_dict(state_dict)

# Create inference engine
engine = InferenceEngine(model, device="cuda")

# Make predictions
result = engine.predict_single("path/to/xray.jpg")
print(f"OA Grade: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"T-Score: {result.get('t_score', 'N/A')}")

# Batch prediction
results = engine.predict_batch([
    "xray1.jpg",
    "xray2.jpg",
    "xray3.jpg"
])
```

### Multi-Task Learning (Classification + Regression)

```python
# Automatically handles multi-task loss
model = KneeOADetectionModel(
    num_classes=5,
    include_regression=True  # Enable T-score prediction
)

# Training automatically handles both tasks
# Adjust weights in training_config:
model_config.classification_weight = 1.0
model_config.regression_weight = 0.5
```

### Explainability with Grad-CAM

```python
from src.evaluation import GradCAM, ExplainabilityVisualizer
import torch

# Create Grad-CAM
gradcam = GradCAM(model, target_layer_name="features")

# Compute attention map
image = torch.randn(1, 3, 224, 224)
attention_map = gradcam.compute_gradcam(image, target_class=2)

# Visualize
visualizer = ExplainabilityVisualizer(class_names)
overlay = visualizer.visualize_gradcam(
    image=original_image,
    gradcam=attention_map,
    true_label=2,
    pred_label=2,
    confidence=0.95
)
```

## 📈 Training Results Interpretation

### Confusion Matrix
Shows classification accuracy per OA grade. Diagonal values indicate correct predictions.

### Training History
- **train_loss**: Should decrease steadily
- **val_loss**: Monitor for overfitting (increasing while train decreases)
- **accuracy**: Should increase over epochs
- **F1-Score**: Weighted average accounting for class imbalance

### Grad-CAM Visualization
Highlights X-ray regions important for prediction:
- Red areas: High attention (important for classification)
- Blue areas: Low attention (less important)

## 🎓 Kellgren-Lawrence Grading System

| Grade | Name | Characteristics |
|-------|------|-----------------|
| 0 | None | No OA signs |
| 1 | Doubtful | Doubtful narrowing, possible osteophytes |
| 2 | Mild | Definite osteophytes, minimal JSW narrowing |
| 3 | Moderate | Moderate osteophytes, substantial JSW narrowing |
| 4 | Severe | Large osteophytes, severe JSW narrowing, sclerosis |

## 📊 Radiographic Features Learned

The deep learning model automatically identifies:

1. **Joint Space Width (JSW)**: Gap between femoral and tibial cartilage
2. **Osteophytes**: Bone spurs at joint margins
3. **Subchondral Sclerosis**: Increased bone density below cartilage
4. **Bone Contour Changes**: Deformities and irregularities
5. **Soft Tissue Changes**: Effusions and synovial thickening

## 🔍 Model Architecture

```
Input (224×224 RGB)
        ↓
[Backbone - EfficientNet-B0]
    ↓          ↓
[Features: 1280-dim]
    ↓          ↓
[Classification Head]  [Regression Head]
    ↓                      ↓
[5 Classes]  →  [T-Score]
(KL 0-4)       (-4 to +3)
```

## ⚙️ Configuration Parameters

### Data Configuration
- **image_size**: (224, 224) - Input resolution
- **batch_size**: 32 - Samples per batch
- **augmentation_enabled**: True - Apply data augmentation

### Model Configuration
- **backbone**: "efficientnet_b0" - Feature extractor
- **pretrained**: True - Use ImageNet weights
- **num_classes**: 5 - KL grades (0-4)
- **include_t_score_head**: True - Multi-task learning

### Training Configuration
- **num_epochs**: 100 - Training duration
- **learning_rate**: 1e-3 - Optimizer step size
- **optimizer**: "adam" - Optimization algorithm
- **early_stopping_patience**: 15 - Epochs before stopping

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch_size: `data_config.batch_size = 16`
- Enable gradient checkpointing: `hardware_config.use_gradient_checkpointing = True`
- Use smaller backbone: `model_config.backbone = "efficientnet_b0"`

### Poor Model Performance
- Check data quality and labeling
- Augment dataset with rotations/brightness changes
- Increase training epochs
- Reduce learning rate
- Use class weights for imbalanced data

### Grad-CAM Not Generating
- Ensure model is in eval mode
- Check layer name exists in model
- Verify input has gradients enabled

## 📚 References

- Kellgren, J.H. and Lawrence, J.S. (1957). "Radiological assessment of OA"
- Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling"
- He, K., et al. (2016). "Deep Residual Learning" (ResNet)
- Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from CNNs"

## 📄 License

Research and Educational Use Only

## 🤝 Contributing

Contributions welcome! Please:
1. Test thoroughly on your hardware
2. Update documentation
3. Follow code style guidelines
4. Submit pull requests with clear descriptions

## ⚠️ Medical Disclaimer

**This system is for research and educational purposes only.**

- Not intended for clinical diagnosis
- Should not replace professional medical evaluation
- Always consult qualified radiologists and physicians
- Results may contain errors; expert review required

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review code comments
3. Examine training logs in `outputs/training.log`

## 🔗 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gradio Guide](https://gradio.app/guides/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Keras/TF Implementation](https://github.com/qubvel/classification_models)

---

**Last Updated:** April 2026  
**Version:** 1.0.0
