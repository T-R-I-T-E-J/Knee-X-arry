# Knee Osteoarthritis (OA) AI Detection System

A deep learning system for automated analysis of knee X-ray images. Classifies osteoarthritis severity using the Kellgren-Lawrence grading system and extracts 4 clinical parameters — all learned automatically by the CNN.

## Features

- **OA Classification**: KL grading (0–4) with confidence scores
- **Clinical Parameter Extraction**: JSW, Osteophytes, Sclerosis, Bone Contour — learned by the CNN via ordinal consistency (no manual labels)
- **Dual Dataset Support**: Merges MedicalExpert-I and MedicalExpert-II for training
- **Explainability**: Grad-CAM attention maps highlighting diagnostic regions
- **Web UI**: Gradio interface for uploading X-rays and viewing results

## Dataset Structure

```
Knee-X-arry/
├── MedicalExpert-I/
│   ├── 0Normal/          (KL Grade 0 — 514 images)
│   ├── 1Doubtful/        (KL Grade 1 — 477 images)
│   ├── 2Mild/            (KL Grade 2 — 232 images)
│   ├── 3Moderate/        (KL Grade 3 — 221 images)
│   └── 4Severe/          (KL Grade 4 — 206 images)
│
├── MedicalExpert-II/
│   ├── 0Normal/          (503 images)
│   ├── 1Doubtful/        (488 images)
│   ├── 2Mild/            (232 images)
│   ├── 3Moderate/        (221 images)
│   └── 4Severe/          (206 images)
│
└── KneeOA_AI/            ← This project
```

**Total: 3,300 images** merged → 70% train / 15% val / 15% test

## Project Structure

```
KneeOA_AI/
├── configs/
│   ├── __init__.py
│   └── config.py              # All configuration (data, model, training, hardware)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Preprocessing, augmentation, dataset loading
│   ├── model.py               # Architecture + ClinicalParameterHead + OrdinalConsistencyLoss
│   ├── trainer.py             # Training loop, checkpointing, early stopping
│   ├── evaluation.py          # Metrics, Grad-CAM, clinical parameter analysis
│   └── inference.py           # Single/batch prediction with severity descriptions
├── ui/
│   └── app.py                 # Gradio web interface
├── models/                    # Saved model weights (created during training)
├── outputs/                   # Logs, metrics, visualizations (created during training)
├── train.py                   # Main training script (CLI entry point)
├── requirements.txt           # Python dependencies
└── README.md
```

## Model Architecture

```
Input X-Ray (224×224×3)
        │
  EfficientNet-B0 Backbone (pretrained ImageNet)
        │
  Feature Vector (1280-d)
        │
   ┌────┴────┐
   │         │
ClassHead  ClinicalHead
 (5 cls)   (4 params → sigmoid)
   │         │
 KL Grade   ├── JSW           [1.0=wide → 0.0=gone]       ↓ with severity
 0–4        ├── Osteophytes   [0.0=none → 1.0=severe]     ↑ with severity
            ├── Sclerosis     [0.0=normal → 1.0=hardened]  ↑ with severity
            └── Contour       [0.0=smooth → 1.0=deformed]  ↑ with severity

Loss = CrossEntropy(grade) + 0.3 × OrdinalConsistency(params, grades)
```

**How clinical parameters are learned** — The CNN has no ground-truth measurements. Instead, ordinal consistency loss enforces that JSW decreases and the other scores increase as the KL grade goes up. The model discovers the visual patterns from the X-rays automatically.

## Quick Start

### 1. Install Dependencies

```bash
cd KneeOA_AI
pip install -r requirements.txt
```

> **GPU users**: Install PyTorch with CUDA for faster training:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Train the Model

```bash
# CPU training
python train.py --epochs 50 --device cpu

# GPU training (recommended)
python train.py --epochs 50 --device cuda

# Custom settings
python train.py --epochs 100 --batch-size 16 --backbone resnet50 --learning-rate 0.0005
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--backbone` | efficientnet_b0 | Architecture (efficientnet_b0, resnet50, resnet101) |
| `--device` | cuda | Device (cuda, cpu) |
| `--disable-grad-cam` | — | Skip Grad-CAM generation |

**Training outputs:**
- `models/best_model.pt` — Saved model weights
- `outputs/training_history.json` — Loss/accuracy per epoch
- `outputs/evaluation_metrics.json` — Test performance + clinical parameter correlations
- `outputs/visualizations/` — Confusion matrix, Grad-CAM samples

### 3. Run the Web UI

```bash
python ui/app.py
# Open http://localhost:7860
```

1. Enter model path → Load Model
2. Upload knee X-ray → Analyze
3. View: KL grade, confidence, 4 clinical parameter bars, Grad-CAM attention map

## Inference (Python API)

```python
from src.inference import InferenceEngine
from src.model import KneeOADetectionModel
import torch

model = KneeOADetectionModel(
    backbone="efficientnet_b0",
    num_classes=5,
    include_clinical_params=True,
    num_clinical_params=4
)
state_dict = torch.load("models/best_model.pt", map_location="cpu")
model.load_state_dict(state_dict)

engine = InferenceEngine(model, device="cpu")
result = engine.predict_single("path/to/xray.jpg")

print(f"KL Grade: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.1%}")
for name, value in result['clinical_params'].items():
    desc = result['clinical_descriptions'][name]
    print(f"  {name}: {value:.3f} — {desc}")
```

## KL Grading System

| Grade | Name | Key Features |
|-------|------|-------------|
| 0 | Normal | No OA signs |
| 1 | Doubtful | Possible osteophyte, normal joint space |
| 2 | Mild | Definite osteophytes, minimal JSW narrowing |
| 3 | Moderate | Moderate osteophytes + JSW narrowing, possible sclerosis |
| 4 | Severe | Large osteophytes, severe JSW narrowing, sclerosis, deformity |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | PyTorch |
| Backbone | EfficientNet-B0 / ResNet |
| Image Processing | OpenCV (CLAHE), PIL |
| Explainability | Grad-CAM |
| Web UI | Gradio |
| Metrics | scikit-learn, scipy |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` to 8 or 16 |
| CUDA not available | Use `--device cpu` or install CUDA PyTorch |
| Slow training on CPU | ~5-10 min/epoch for 3300 images; use GPU for 10-20× speedup |
| Windows pickle error | Already fixed — `num_workers=0` in config |

## Disclaimer

**This system is for research and educational purposes only.** Not intended for clinical diagnosis. Always consult qualified medical professionals.

---

**Version:** 2.0.0 — Clinical Parameter Extraction  
**Last Updated:** April 2026
