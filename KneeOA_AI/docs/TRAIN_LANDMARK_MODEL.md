# Knee Joint Landmark Model Training Guide

This repository now contains a specialized Deep Learning **Keypoint Regression model** (`src/landmark_model.py`) designed to specifically predict the X/Y sub-pixel coordinates of the Tibial Plateau and Knee Joint Center, thereby overriding issues natively tied to OpenCV geometry mapping like doctors' pen traces, extreme zooms, or thick clinical text overlays.

If you are cloning this repository, you must supply spatial coordinate data to train the model, because the existing dataset currently only categorizes severity tags (KL Grading).

## 1. Preparing the Annotation Dataset
You must tell the AI where the correct physical anchor points are. 
1. Place roughly `200+` raw, uncropped AP knee X-Rays into the `data/images` folder.
2. Create a JSON dictionary file named `data/landmark_labels.json`.
3. The JSON file must strictly map the image filename to an array of **6 floating point variables** normalized to a `[0, 1]` proportional scale representing the `[X, Y]` pairs for the Medial, Center, and Lateral knee anchors.

**Example `data/landmark_labels.json`:**
```json
{
    "patient1_XRay.jpg": [0.352, 0.450, 0.500, 0.452, 0.655, 0.448],
    "patient2_XRay.jpg": [0.310, 0.500, 0.505, 0.502, 0.690, 0.510]
}
```
> [!NOTE]
> *Normalization Math*: If your image is 1000px wide, and the center joint is located at 500px, your `x_center` should be written as `0.5`. Do this for all 6 coordinate variables.

## 2. Launching Training
Because this is utilizing a `ResNet50` mathematical backbone, we heavily penalize wrong pixel coordinate distancing using MSE (Mean Squared Error Loss).

To initiate the tracking optimization, just run:
```bash
python train_landmarks.py
```

### Hyperparameters
- **Epochs:** `50` (Recommended) - Due to the heavily frozen backbone (`requires_grad = False` on base layers), 50 epochs should securely convergence the new regression head without catastrophic overfitting on a small clinical sample set.
- **Batch Size:** `16` (Decrease to `8` if training causes CUDA OutOfMemory Exceptions).
- **Learning Rate:** `1e-4` (Standard Adam optimizer threshold for finely tuning normalized decimals).

## 3. Integration
Training will spit out `models/best_landmark_model.pt`.
You can then link this PyTorch binary object directly right into `src/tibial_plateau_analyzer.py` via `model.load_state_dict()` to officially replace the OpenCV module tracker with real-time neural tracking logic!
