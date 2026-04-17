"""
Gradio Web UI for Knee OA Detection System
Provides an interactive interface for X-ray analysis
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.inference import InferenceEngine, PredictionFormatter
from src.data_loader import ImagePreprocessor
from src.model import KneeOADetectionModel
from src.evaluation import GradCAM, ExplainabilityVisualizer

# ==================== GLOBAL STATE ====================
class AppState:
    """Global application state"""
    def __init__(self):
        self.model = None
        self.inference_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
        self.model_loaded = False

app_state = AppState()

# ==================== MODEL INITIALIZATION ====================
def initialize_model(model_path: str) -> str:
    """
    Initialize the model for inference
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Status message
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            return "❌ Model file not found"
        
        # Create model
        app_state.model = KneeOADetectionModel(
            backbone="efficientnet_b0",
            num_classes=5,
            include_regression=True
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=app_state.device)
        app_state.model.load_state_dict(state_dict)
        app_state.model = app_state.model.to(app_state.device)
        app_state.model.eval()
        
        # Create inference engine
        app_state.inference_engine = InferenceEngine(
            model=app_state.model,
            model_path=None,  # Already loaded
            device=app_state.device,
            class_names=app_state.class_names,
            preprocessor=ImagePreprocessor()
        )
        
        app_state.model_loaded = True
        
        return f"✅ Model loaded successfully on {app_state.device.upper()}"
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app_state.model_loaded = False
        return f"❌ Error loading model: {str(e)}"

# ==================== PREDICTION FUNCTIONS ====================
def predict_knee_xray(image_input) -> Tuple[str, str, str, Optional[np.ndarray]]:
    """
    Predict osteoarthritis grade and bone health from X-ray image
    
    Args:
        image_input: Input image (PIL Image or numpy array)
        
    Returns:
        Prediction text, probability chart description, Grad-CAM visualization, detailed metrics
    """
    if not app_state.model_loaded:
        return "❌ Model not loaded. Please initialize the model first.", "", "", None
    
    try:
        # Convert image to numpy if needed
        if hasattr(image_input, 'convert'):  # PIL Image
            image_array = np.array(image_input.convert('RGB'))
        else:
            image_array = image_input
        
        # Save temporarily
        temp_path = Path("/tmp/temp_xray.png")
        if image_array.dtype == np.uint8 and image_array.max() > 1:
            # Already in [0, 255] range
            from PIL import Image
            Image.fromarray(image_array.astype(np.uint8)).save(temp_path)
        else:
            # Normalize to [0, 255]
            from PIL import Image
            normalized = ((image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(normalized).save(temp_path)
        
        # Make prediction
        prediction = app_state.inference_engine.predict_single(str(temp_path), return_features=True)
        
        # Format output text
        pred_text = f"🏥 **Osteoarthritis Assessment**\n\n"
        pred_text += f"**Kellgren-Lawrence Grade:** {prediction['predicted_label']}\n"
        pred_text += f"**Confidence:** {prediction['confidence']:.1%}\n"
        
        if 't_score' in prediction:
            t_score = prediction['t_score']
            pred_text += f"\n**Bone Density T-Score:** {t_score:.2f}\n"
            
            if t_score > -1:
                pred_text += "**Bone Health Status:** Normal\n"
            elif t_score > -2.5:
                pred_text += "**Bone Health Status:** Osteopenia (Low Bone Density)\n"
            else:
                pred_text += "**Bone Health Status:** Osteoporosis\n"
        
        # Create probability chart
        probs_text = "**Classification Probabilities:**\n\n"
        for class_name, prob in sorted(prediction['probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            probs_text += f"{class_name:12} │ {bar} │ {prob:.1%}\n"
        
        # Generate Grad-CAM
        try:
            gradcam = GradCAM(app_state.model, "features")
            cam_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float().to(app_state.device)
            cam = gradcam.compute_gradcam(cam_tensor, prediction['predicted_class'])
            
            if cam is not None:
                # Normalize image for visualization
                if image_array.dtype != np.uint8:
                    vis_image = ((image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    vis_image = image_array
                
                # Create overlay
                import cv2
                h, w = vis_image.shape[:2]
                cam_resized = cv2.resize(cam, (w, h))
                heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                overlay = cv2.addWeighted(vis_image, 0.6, heatmap, 0.4, 0)
                cam_output = overlay
            else:
                cam_output = None
        except Exception as e:
            logger.warning(f"Could not generate Grad-CAM: {e}")
            cam_output = None
        
        # Detailed metrics
        metrics_text = f"📊 **Detailed Metrics**\n\n"
        for class_name in app_state.class_names:
            prob = prediction['probabilities'].get(class_name, 0)
            metrics_text += f"• {class_name}: {prob:.4f}\n"
        
        return pred_text, probs_text, metrics_text, cam_output
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Error during prediction: {str(e)}", "", "", None

# ==================== INFO FUNCTIONS ====================
def get_kl_grading_info() -> str:
    """Get Kellgren-Lawrence grading system information"""
    return """
## Kellgren-Lawrence (KL) Grading System

The Kellgren-Lawrence grading system is the standard for assessing osteoarthritis severity in knee X-rays:

**Grade 0 - Normal:** No signs of OA
- Normal joint space
- No osteophytes

**Grade 1 - Doubtful:** Possible OA changes
- Possible osteophyte
- Normal joint space

**Grade 2 - Mild OA:** Definite OA
- Definite osteophytes
- Minimal joint space narrowing
- No subchondral sclerosis

**Grade 3 - Moderate OA:** Moderate OA
- Moderate osteophytes
- Moderate joint space narrowing
- Possible subchondral sclerosis

**Grade 4 - Severe OA:** Severe OA
- Large osteophytes
- Severe joint space narrowing
- Marked subchondral sclerosis
- Bone deformity

### Key Radiographic Features:
- **Joint Space Width (JSW):** Distance between bone surfaces
- **Osteophytes:** Bone spurs at joint edges
- **Subchondral Sclerosis:** Increased bone density below cartilage
- **Bone Contour Changes:** Deformities in bone shape
"""

def get_bone_health_info() -> str:
    """Get bone health assessment information"""
    return """
## Bone Health Assessment (T-Score)

The T-Score measures bone density compared to healthy young adults:

**T-Score > -1.0:** Normal bone density
- No intervention needed
- Continue regular exercise and adequate calcium/vitamin D

**T-Score -1.0 to -2.5:** Osteopenia (Low bone density)
- Increased fracture risk
- Consider bone density medications
- Lifestyle modifications recommended

**T-Score < -2.5:** Osteoporosis
- Significantly increased fracture risk
- Medical intervention recommended
- Regular monitoring required

### Risk Factors:
- Age (especially post-menopausal women)
- Family history
- Low body mass index
- Inadequate calcium/vitamin D
- Sedentary lifestyle
- Certain medications (corticosteroids)
"""

# ==================== GRADIO INTERFACE ====================
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Knee OA AI Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Knee OA Detection System")
        gr.Markdown("Automated Analysis of Knee X-rays for Osteoarthritis Severity and Bone Health")
        
        with gr.Tabs():
            # ==================== PREDICTION TAB ====================
            with gr.Tab("🔍 Prediction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Setup")
                        model_path_input = gr.Textbox(
                            label="Model Path",
                            placeholder="Enter path to model checkpoint",
                            value=""
                        )
                        init_button = gr.Button("Load Model", variant="primary")
                        init_status = gr.Textbox(label="Status", interactive=False)
                        
                        init_button.click(
                            fn=initialize_model,
                            inputs=[model_path_input],
                            outputs=[init_status]
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Upload X-ray Image")
                        image_input = gr.Image(
                            label="Knee X-ray Image",
                            type="pil",
                            shape=(512, 512)
                        )
                        predict_button = gr.Button("Analyze X-ray", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        prediction_output = gr.Markdown(label="Prediction")
                    with gr.Column():
                        probabilities_output = gr.Markdown(label="Confidence Scores")
                    with gr.Column():
                        metrics_output = gr.Markdown(label="Metrics")
                
                with gr.Row():
                    gradcam_output = gr.Image(label="Attention Map (Grad-CAM)")
                
                predict_button.click(
                    fn=predict_knee_xray,
                    inputs=[image_input],
                    outputs=[prediction_output, probabilities_output, metrics_output, gradcam_output]
                )
            
            # ==================== INFO TABS ====================
            with gr.Tab("📚 KL Grading System"):
                kl_info = get_kl_grading_info()
                gr.Markdown(kl_info)
            
            with gr.Tab("🦴 Bone Health"):
                bone_info = get_bone_health_info()
                gr.Markdown(bone_info)
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
## About This System

This AI system automatically analyzes knee X-ray images to:
1. **Classify osteoarthritis severity** using the Kellgren-Lawrence grading system
2. **Assess bone health** by predicting bone density T-scores
3. **Identify key features** such as joint space width and osteophytes

### Technology
- **Deep Learning Framework:** PyTorch
- **Backbone Architecture:** EfficientNet-B0
- **Explainability:** Grad-CAM visualization
- **Training Data:** Multiple expert annotations

### Limitations
- This system is for **research and educational purposes**
- Should not be used for clinical diagnosis without expert review
- Always consult qualified medical professionals
- Prediction accuracy depends on image quality

### Disclaimer
This tool is provided as-is for research purposes. Medical professionals should review all results.
                """)
    
    return demo

# ==================== MAIN ====================
if __name__ == "__main__":
    logger.info("Starting Knee OA Detection Web UI...")
    logger.info(f"Using device: {app_state.device}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
