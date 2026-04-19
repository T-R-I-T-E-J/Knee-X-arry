"""
Gradio Web Interface for Knee OA AI analysis.
Provides a premium desktop-like experience for clinical diagnostics.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.inference import InferenceEngine
from configs.config import model_config, data_config
from src.evaluation import GradCAM

# Global state for inference
engine = None
gradcam = None

def load_ai_model(model_path):
    """Loads a trained model and initializes inference engines."""
    global engine, gradcam
    try:
        model = create_model(model_config)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        engine = InferenceEngine(model, device="cpu")
        gradcam = GradCAM(model, target_layer_name="features")
        return f"✅ Model loaded successfully from {Path(model_path).name}"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def analyze_knee(image):
    """Main diagnostic function."""
    if engine is None:
        return "Please load a model first.", None, None
        
    # Temporary save for inference
    temp_path = "temp_xray.png"
    Image.fromarray(image).save(temp_path)
    
    # 1. Prediction
    result = engine.predict(temp_path)
    
    # 2. Explainability
    img_np = engine.preprocessor.preprocess(temp_path)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    heatmap = gradcam.generate(img_tensor, target_class=result['grade'])
    
    # Superimpose heatmap
    heatmap_colored = gr.ImageMask(heatmap, label="Attention Area") # Abstracting for display
    
    # Build text report
    report = f"### Diagnostic Report\n\n"
    report += f"**Predicted Severity: {result['label']}** (Confidence: {result['confidence']:.1%})\n\n"
    report += f"**Clinical Markers:**\n"
    for p, val in result['clinical_params'].items():
        desc = result['descriptions'][p]
        report += f"- {p}: {val:.3f} — *{desc}*\n"
        
    return report, image, heatmap

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="Knee OA AI") as demo:
    gr.Markdown("# 🏥 Knee Osteoarthritis AI Diagnostic Suite")
    
    with gr.Row():
        with gr.Column(scale=2):
            model_input = gr.Textbox(label="Model File Path", placeholder="e.g., models/best_model.pt")
            load_btn = gr.Button("🚀 Initialize Diagnostic Engine", variant="primary")
            load_status = gr.Markdown("*Model not loaded*")
            
        with gr.Column(scale=3):
            gr.Markdown("### Instructions\n1. Load your trained `.pt` model file.\n2. Upload a knee X-ray image (Grayscale/DICOM converted).\n3. Click 'Analyze' to generate the clinical report.")

    with gr.Tabs():
        with gr.TabItem("Single Patient Analysis"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Patient X-ray")
                    analyze_btn = gr.Button("🔍 Run AI Analysis", variant="primary")
                    
                with gr.Column():
                    report_md = gr.Markdown("### Clinical Outcome\n*Results will appear here after analysis.*")
                    
            with gr.Row():
                 gradcam_plot = gr.Image(label="Grad-CAM Attention Map (Diagnostic Focus)")

        with gr.TabItem("About Automated KL-Grading"):
            gr.Markdown("""
            ### The Kellgren-Lawrence Grading System
            Our AI classifies severity into 5 grades (0-4):
            - **Grade 0**: Healthy joint.
            - **Grade 1**: Doubtful joint space narrowing.
            - **Grade 2**: Definite osteophytes; possible narrowing.
            - **Grade 3**: Multiple osteophytes; definite narrowing; possible sclerosis.
            - **Grade 4**: Large osteophytes; severe narrowing; definite sclerosis; bone deformity.
            """)

    # Events
    load_btn.click(load_ai_model, inputs=[model_input], outputs=[load_status])
    analyze_btn.click(analyze_knee, inputs=[input_img], outputs=[report_md, input_img, gradcam_plot])

if __name__ == "__main__":
    demo.launch()
