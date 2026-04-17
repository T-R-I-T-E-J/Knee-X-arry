# System Architecture

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNEE OA DETECTION SYSTEM                         │
└─────────────────────────────────────────────────────────────────────┘

                           INPUT LAYER
                                │
                    ┌───────────┴───────────┐
                    │                       │
            TRAINING PIPELINE       INFERENCE PIPELINE
                    │                       │
        ┌───────────────────┐    ┌─────────────────────┐
        │  DATA LOADING     │    │   MODEL LOADING     │
        │  & PREPROCESSING  │    │   (weights)         │
        └─────────┬─────────┘    └────────┬────────────┘
                  │                       │
        ┌─────────▼─────────┐    ┌───────▼────────────┐
        │  AUGMENTATION     │    │  INFERENCE ENGINE  │
        │  • Rotation       │    │  • Single image    │
        │  • Brightness     │    │  • Batch processing│
        │  • Contrast       │    └───────┬────────────┘
        └─────────┬─────────┘            │
                  │                      │
        ┌─────────▼─────────────────────▼┐
        │   DEEP LEARNING MODEL           │
        │  ┌─────────────────────────┐   │
        │  │   BACKBONE              │   │
        │  │   (EfficientNet-B0)     │   │
        │  │   - ImageNet pretrained │   │
        │  │   - 1280 features       │   │
        │  └──────────┬──────────────┘  │
        │             │                  │
        │  ┌──────────▼──────────┬──────▼─────────┐
        │  │                     │                │
        │  │  CLASSIFICATION     │  REGRESSION    │
        │  │  HEAD               │  HEAD          │
        │  │                     │                │
        │  │  • 512 neurons      │  • 512 neurons │
        │  │  • BatchNorm        │  • BatchNorm   │
        │  │  • Dropout (0.3)    │  • Dropout     │
        │  │  • 256 neurons      │  • 256 neurons │
        │  │  • 5 outputs (KL)   │  • 1 output    │
        │  └──────┬──────────────┴────────┬──────┘
        │         │                       │
        │  [KL Grade Logits]      [T-Score Pred]
        └─────────┬───────────────────────┘
                  │
        ┌─────────▼──────────┬──────────┐
        │  LOSS FUNCTIONS    │          │
        │                    │          │
        │ CrossEntropy       │ MSELoss  │
        │ (Classification)   │(T-Score) │
        └─────────┬──────────┴─────┬────┘
                  │                │
        ┌─────────▼────────────────▼──┐
        │  MULTI-TASK LOSS            │
        │  = 1.0 × CLF_Loss           │
        │    + 0.5 × REG_Loss         │
        └─────────┬────────────────────┘
                  │
        ┌─────────▼────────────┐
        │  BACKPROPAGATION     │
        │  GRADIENT DESCENT    │
        └─────────┬────────────┘
                  │
                  ▼
        [UPDATED WEIGHTS]

              EVALUATION LAYER
                    │
        ┌───────────┼───────────┬───────────┐
        │           │           │           │
    METRICS      G-CAM      CLASS        CONFUSION
  CALCULATION  GENERATION   REPORT       MATRIX
        │           │           │           │
        └───────────┼───────────┼───────────┘
                    │
            ┌───────▼────────┐
            │   OUTPUT       │
            │   • Metrics    │
            │   • Visuals    │
            │   • Reports    │
            └────────────────┘

                OUTPUT LAYER
                    │
        ┌───────────┼───────────┐
        │           │           │
    WEB UI      CSV/JSON     INFERENCE
    (Gradio)    REPORTS      SERVICE
```

## Data Flow Diagram

```
TRAINING MODE:
────────────

Raw X-rays        Metadata
    │                │
    └────┬───────────┘
         │
    [Data Loader]
         │
    ┌────┼────┐
    │   Train│Val│Test    (70/15/15)
    └────┼────┘
         │
    [Preprocessing]
    • Resize (224×224)
    • CLAHE enhancement
    • Normalization
         │
    ┌────┴────┐
    │ Augment  │ (Train only)
    │ • Random rotation
    │ • Brightness/Contrast
    └────┬────┘
         │
    [Batch Creation]
         │
    ┌────▼──────────────┐
    │ PyTorch DataLoader│
    │ batch_size=32     │
    └────┬──────────────┘
         │
    [Training Loop]
    ├─ Forward pass
    ├─ Loss calculation
    ├─ Backward pass
    ├─ Gradient update
    └─ Validation
         │
    [Checkpoint]
    └──→ models/best_model.pt


INFERENCE MODE:
─────────────

Single/Batch X-rays
         │
    [Input Validation]
         │
    [Preprocessing]
    (Same as training)
         │
    [Model Forward]
         │
    ┌────┴────────┬───────────┐
    │             │           │
Classification  Regression  Features
  Logits        Output    (for Grad-CAM)
    │             │           │
    └────┬────────┴───────────┘
         │
    [Post-processing]
    • Softmax → Probabilities
    • Argmax → Class
         │
    ┌────▼──────────────────┐
    │  Prediction Result    │
    │ • KL Grade (0-4)      │
    │ • Confidence (%)      │
    │ • T-Score             │
    │ • All Probabilities   │
    └────┬──────────────────┘
         │
    [Optional: Grad-CAM]
         │
    [Output Formatting]
         └──→ JSON/UI
```

## Component Architecture

```
┌─────────────────────────────────────┐
│      configs/config.py              │
│  ┌──────────────────────────────┐   │
│  │ Data Configuration           │   │
│  │ • Paths, class names         │   │
│  │ • Image size, augmentation   │   │
│  │ • Train/val/test split       │   │
│  ├──────────────────────────────┤   │
│  │ Model Configuration          │   │
│  │ • Backbone (EffNet/ResNet)   │   │
│  │ • Multi-task learning        │   │
│  ├──────────────────────────────┤   │
│  │ Training Configuration       │   │
│  │ • Epochs, LR schedule        │   │
│  │ • Optimizer, regularization  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      src/data_loader.py             │
│  ┌──────────────────────────────┐   │
│  │ ImagePreprocessor            │   │
│  │ • Load X-ray files           │   │
│  │ • CLAHE enhancement          │   │
│  │ • Resize & normalize         │   │
│  ├──────────────────────────────┤   │
│  │ XRayAugmentation             │   │
│  │ • Random transforms          │   │
│  │ • Medical-specific           │   │
│  ├──────────────────────────────┤   │
│  │ KneeXRayDataset              │   │
│  │ • PyTorch Dataset class      │   │
│  ├──────────────────────────────┤   │
│  │ KneeXRayDataLoader           │   │
│  │ • Data collection            │   │
│  │ • Train/val/test split       │   │
│  │ • DataLoader creation        │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      src/model.py                   │
│  ┌──────────────────────────────┐   │
│  │ EfficientNetBackbone         │   │
│  │ • Feature extraction         │   │
│  ├──────────────────────────────┤   │
│  │ ClassificationHead           │   │
│  │ • KL grade prediction        │   │
│  ├──────────────────────────────┤   │
│  │ RegressionHead               │   │
│  │ • T-score prediction         │   │
│  ├──────────────────────────────┤   │
│  │ KneeOADetectionModel         │   │
│  │ • Main model class           │   │
│  │ • Multi-task interface       │   │
│  ├──────────────────────────────┤   │
│  │ Loss Functions               │   │
│  │ • FocalLoss (class weight)   │   │
│  │ • MultiTaskLoss              │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      src/trainer.py                 │
│  ┌──────────────────────────────┐   │
│  │ Callbacks                    │   │
│  │ • EarlyStopping              │   │
│  │ • Checkpoint                 │   │
│  │ • TensorBoard                │   │
│  ├──────────────────────────────┤   │
│  │ Trainer                      │   │
│  │ • Training loop              │   │
│  │ • Validation loop            │   │
│  │ • Metrics tracking           │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      src/evaluation.py              │
│  ┌──────────────────────────────┐   │
│  │ MetricsCalculator            │   │
│  │ • Accuracy, F1, Precision    │   │
│  │ • Confusion matrix           │   │
│  ├──────────────────────────────┤   │
│  │ GradCAM                      │   │
│  │ • Attention visualization    │   │
│  ├──────────────────────────────┤   │
│  │ ExplainabilityVisualizer     │   │
│  │ • Plot generation            │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      src/inference.py               │
│  ┌──────────────────────────────┐   │
│  │ InferenceEngine              │   │
│  │ • Single prediction          │   │
│  │ • Batch prediction           │   │
│  ├──────────────────────────────┤   │
│  │ PredictionFormatter          │   │
│  │ • Format output              │   │
│  ├──────────────────────────────┤   │
│  │ PredictionReportGenerator    │   │
│  │ • JSON/CSV reports           │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      ui/app.py                      │
│  ┌──────────────────────────────┐   │
│  │ Gradio Interface             │   │
│  │ • Model loading              │   │
│  │ • Image upload               │   │
│  │ • Prediction display         │   │
│  │ • Grad-CAM visualization     │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Model Architecture Details

```
┌─────────────────────────────────────────┐
│     INPUT: X-ray Image (224×224)        │
└────────────────┬────────────────────────┘
                 │
        ┌────────▼────────┐
        │  EfficientNet-B0 │
        │  (1M parameters) │
        │                  │
        │ Blocks 1-9       │
        │ • Conv blocks    │
        │ • MBConv layers  │
        │ • Squeeze-Excite │
        └────────┬─────────┘
                 │
        ┌────────▼──────────┐
        │ Feature Vector    │
        │ (1280 features)   │
        └────────┬──────────┘
                 │
        ┌────────┴──────────────┬─────────────────┐
        │                       │                 │
   ┌────▼──────┐          ┌────▼──────┐      ┌──▼─────────┐
   │  CLASSIF   │          │  REGRESS  │      │ Optional:  │
   │   HEAD     │          │   HEAD    │      │ Grad-CAM   │
   │            │          │           │      │            │
   │ Linear(1280,512)      │Linear(1280,512)  │ Hook       │
   │ BatchNorm1d(512)      │BatchNorm1d(512)  │ features   │
   │ ReLU                  │ReLU               │            │
   │                       │                  │            │
   │ Dropout(0.3)          │Dropout(0.3)      │            │
   │ Linear(512,256)       │Linear(512,256)   │            │
   │ BatchNorm1d(256)      │BatchNorm1d(256)  │            │
   │ ReLU                  │ReLU               │            │
   │                       │                  │            │
   │ Dropout(0.3)          │Dropout(0.3)      │            │
   │ Linear(256,5) ────┐   │Linear(256,1) ──┐ │            │
   └────────────────────┼──┘           │    │    │            │
                        │              │   │    │            │
              ┌─────────▼───┐  ┌──────▼─┐ │    │            │
              │  Logits     │  │ T-Score│ │    │            │
              │  (5 classes)│  │        │ │    │            │
              └─────────────┘  └────────┘ │    │            │
                    │               │     │    │            │
        ┌───────────▼──────────────▼──┐  │    │            │
        │   MULTI-TASK LOSS          │  │    │            │
        │                            │  │    │            │
        │ Loss = 1.0 × CrossEntropy  │  │    │            │
        │       + 0.5 × MSELoss      │  │    │            │
        └────────────────────────────┘  │    │            │
                                        │    │            │
        ┌──────────────────────────────▼┴────▼┐            │
        │ OUTPUTS                             │            │
        │ • KL Grade (0-4)                    │            │
        │ • Class Probabilities (5×)          │            │
        │ • Confidence Score (0-1)            │            │
        │ • T-Score (optional) (-4 to +3)     │            │
        │ • Attention Map (optional)          │            │
        └─────────────────────────────────────┘            │
                                                            │
                    ┌───────────────────────────────────────┘
                    │
        ┌───────────▼──────────────┐
        │   GRAD-CAM ATTENTION     │
        │   (Explainability)       │
        │                          │
        │ • Compute gradients      │
        │ • Weight activations     │
        │ • Generate heatmap       │
        │ • Overlay on image       │
        └──────────────────────────┘
```

## Training Pipeline States

```
START
  │
  ├─→ Initialize Model (EfficientNet)
  │   └─→ Load ImageNet weights
  │
  ├─→ Setup Optimizer (Adam)
  │   └─→ Learning rate scheduler
  │
  ├─→ Setup Loss Functions
  │   ├─→ CrossEntropy (classification)
  │   ├─→ MSELoss (regression)
  │   └─→ MultiTaskLoss (combined)
  │
  ├─→ Register Callbacks
  │   ├─→ EarlyStopping
  │   ├─→ ModelCheckpoint
  │   └─→ TensorBoard
  │
  └─→ TRAINING LOOP:
      ├─ FOR each epoch:
      │  ├─ FOR each batch:
      │  │  ├─ Forward pass
      │  │  ├─ Loss computation
      │  │  ├─ Backward pass
      │  │  └─ Gradient update
      │  │
      │  ├─ Validation phase
      │  ├─ Metrics calculation
      │  ├─ Callback triggers
      │  │
      │  └─ Check early stopping?
      │     └─→ YES: Break
      │
      ├─ Save best model checkpoint
      ├─ Save training history
      │
      └─→ EVALUATION
          ├─ Test set evaluation
          ├─ Metrics computation
          ├─ Grad-CAM visualization
          ├─ Confusion matrix
          ├─ Classification report
          └─ Save results

END
```

## Deployment Architecture

```
┌──────────────────────────────────────────┐
│        Docker Container (Optional)       │
│ ┌──────────────────────────────────────┐ │
│ │  Gradio Web Application              │ │
│ │  ┌────────────────────────────────┐  │ │
│ │  │  Frontend (HTML/CSS/JS)        │  │ │
│ │  │  • Image upload UI             │  │ │
│ │  │  • Model selection             │  │ │
│ │  │  • Result display              │  │ │
│ │  └────────┬───────────────────────┘  │ │
│ │           │                          │ │
│ │  ┌────────▼───────────────────────┐  │ │
│ │  │  Backend (Python/Gradio)       │  │ │
│ │  │  • Request handling            │  │ │
│ │  │  • Input validation            │  │ │
│ │  └────────┬───────────────────────┘  │ │
│ │           │                          │ │
│ │  ┌────────▼───────────────────────┐  │ │
│ │  │  Inference Engine              │  │ │
│ │  │  • Model loading               │  │ │
│ │  │  • Preprocessing               │  │ │
│ │  │  • Prediction                  │  │ │
│ │  │  • Post-processing             │  │ │
│ │  └────────┬───────────────────────┘  │ │
│ │           │                          │ │
│ │  ┌────────▼───────────────────────┐  │ │
│ │  │  Trained Model (GPU Memory)    │  │ │
│ │  │  • EfficientNet backbone       │  │ │
│ │  │  • Classification head         │  │ │
│ │  │  • Regression head (optional)  │  │ │
│ │  └────────────────────────────────┘  │ │
│ │                                      │ │
│ └──────────────────────────────────────┘ │
│           ↑           ▼                  │
│   Port: 7860 (Mapped to host:7860)      │
│           (Optional)                     │
└──────────────────────────────────────────┘

HOST MACHINE:
• Browser: http://localhost:7860
• GPU support (NVIDIA CUDA)
• Memory allocation
```

---

**Diagram Updated:** April 2026
