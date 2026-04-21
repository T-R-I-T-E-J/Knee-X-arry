"""
Inference script for Knee OA Detection System.
Loads the best model and predicts KL Grade + Clinical Markers for a single image.
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs.config import model_config, data_config
from src.model import create_model

def get_mark(val, inverse=False):
    if inverse:
        if val > 0.7: return "Healthy"
        if val > 0.5: return "Mild"
        if val > 0.3: return "Moderate"
        return "Critical"
    else:
        if val < 0.3: return "Minimal"
        if val < 0.5: return "Noticeable"
        if val < 0.7: return "Significant"
        return "High/Severe"

def predict(image_path, model_path, device="cpu"):
    # 1. Load Model
    model = create_model(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Preprocess Image
    from src.preprocessing import AutoCutter
    cutter = AutoCutter()
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cutter.crop(x)),
        transforms.Resize(data_config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(data_config.normalize_mean, data_config.normalize_std)
    ])
    
    orig_img = cv2.imread(image_path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(orig_img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 3. Inference & Grad-CAM
    from src.evaluation import GradCAM
    # We target the last layer of the feature extractor
    cam_gen = GradCAM(model, "features") 
    
    with torch.set_grad_enabled(True): # Needed for Grad-CAM
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            logits, params = outputs
            params = params.squeeze().detach().cpu().numpy()
        else:
            logits = outputs
            params = None
            
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        pred_grade = np.argmax(probs)
        confidence = probs[pred_grade]
        
        # Generate Heatmap
        heatmap = cam_gen.generate(input_tensor, target_class=pred_grade)

    # 4. Draw Gap Lines (Image Processing)
    # Resize original for visualization
    vis_img = cv2.resize(orig_img, (448, 448))
    gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    
    # Enhance edges for "Sharpness" using Sobel for directional bone margins
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Threshold to find horizontal bone surfaces (the gap boundaries)
    _, edges = cv2.threshold(abs_grad_y, 40, 255, cv2.THRESH_BINARY)
    
    # Create Overlay
    heatmap_resized = cv2.resize(heatmap, (448, 448))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # 4. Auto-Focus (ROI Detection)
    # Find the densest cluster of horizontal gradients to locate the joint center
    row_sums = np.sum(edges, axis=1)
    joint_y = np.argmax(row_sums)
    joint_y = np.clip(joint_y, 112, 336) # Keep within reasonable center bounds
    
    # Define ROI (Zoom window)
    y1, y2 = joint_y - 120, joint_y + 120
    x1, x2 = 80, 448 - 80
    
    # Crop and Resize for "Perfect" view
    roi_img = vis_img[y1:y2, x1:x2]
    roi_heatmap = heatmap_color[y1:y2, x1:x2]
    roi_edges = edges[y1:y2, x1:x2]
    
    # Blend image and heatmap in the zoom view
    overlay_roi = cv2.addWeighted(roi_img, 0.7, roi_heatmap, 0.3, 0)
    
    # Draw Clean Clinical Curves (Numerical Smoothing)
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by length and pick the best ones for Top (Femur) and Bottom (Tibia)
    top_curves = []
    bottom_curves = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 60: # Only look at long stable lines
            points = cnt.reshape(-1, 2)
            if np.mean(points[:, 1]) < 110: # Top half of ROI
                top_curves.append(points)
            else: # Bottom half
                bottom_curves.append(points)

    def draw_smooth_curve(target_img, points, color, label):
        if len(points) < 5: return
        # Sort points by x to fit a function y = f(x)
        points = points[points[:, 0].argsort()]
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Fit a smooth polynomial (Degree 2 for knee curvature)
        poly = np.polyfit(x_coords, y_coords, 2)
        fit_fn = np.poly1d(poly)
        
        # Draw the smooth line
        draw_x = np.linspace(x_coords.min(), x_coords.max(), 100).astype(int)
        draw_y = fit_fn(draw_x).astype(int)
        
        # Clip to image boundaries
        valid = (draw_y >= 0) & (draw_y < target_img.shape[0])
        pts = np.column_stack((draw_x[valid], draw_y[valid])).reshape((-1, 1, 2))
        cv2.polylines(target_img, [pts], False, color, 2, cv2.LINE_AA)
        
        # Professional Labeling at the start of the line
        cv2.putText(target_img, label, (draw_x[0], draw_y[0]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw the best top and bottom curves
    if top_curves:
        # Combine all top fragments and fit one smooth line
        all_top = np.vstack(top_curves)
        draw_smooth_curve(overlay_roi, all_top, (0, 255, 0), "Femur Margin")
        
    if bottom_curves:
        all_bottom = np.vstack(bottom_curves)
        draw_smooth_curve(overlay_roi, all_bottom, (0, 255, 255), "Tibial Plateau")
            
    # 5. Save Report & Visual
    report = {
        "image": Path(image_path).name,
        "grade": int(pred_grade),
        "label": data_config.class_names[pred_grade],
        "confidence": float(confidence),
        "markers": {}
    }
    
    if params is not None:
        marker_names = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
        for i, name in enumerate(marker_names):
            score = float(params[i])
            mark = get_mark(score, inverse=(i==0))
            report["markers"][name] = {"score": score, "mark": mark}
    
    import json
    with open("results.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Save the High-Precision Zoomed image
    cv2.imwrite("diagnostic_overlay.png", overlay_roi)
    
    print(f"Analysis complete! Zoomed Diagnostic overlay saved to 'diagnostic_overlay.png'")
    print(f"Summary: Grade {pred_grade} ({report['label']})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
    elif not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}. Did you train first?")
    else:
        predict(args.image, args.model)
