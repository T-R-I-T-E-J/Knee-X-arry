"""
Example Usage Script - Demonstrates the complete workflow
Run this to understand how to use the Knee OA Detection System
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from configs.config import (
    data_config, model_config, training_config, 
    hardware_config, eval_config
)
from src.data_loader import KneeXRayDataLoader, ImagePreprocessor
from src.model import create_model, KneeOADetectionModel
from src.inference import InferenceEngine, PredictionFormatter
from src.evaluation import GradCAM, ExplainabilityVisualizer, MetricsCalculator

# ==================== EXAMPLE 1: DATA LOADING ====================
def example_data_loading():
    """Example 1: Load and inspect dataset"""
    print("\n" + "="*80)
    print("EXAMPLE 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    # Setup data loader
    data_loader = KneeXRayDataLoader(
        dataset_roots={
            "data": Path("data/raw")
        },
        data_config=data_config,
        batch_size=32,
        num_workers=4
    )
    
    # Collect images
    image_paths, labels = data_loader.collect_images_and_labels()
    print(f"\n✓ Found {len(image_paths)} images")
    print(f"✓ Classes: {data_config.class_names}")
    
    # Count per class
    from collections import Counter
    class_counts = Counter(labels)
    print(f"\nClass distribution:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = data_config.class_names[class_idx]
        print(f"  {class_name}: {count} images")
    
    # Create data loaders
    print("\n✓ Creating DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # Inspect a batch
    print(f"\nBatch inspection:")
    images, labels = next(iter(train_loader))
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Sample labels: {labels[:5].numpy()}")
    print(f"  Sample label names: {[data_config.class_names[l] for l in labels[:5]]}")

# ==================== EXAMPLE 2: MODEL CREATION ====================
def example_model_creation():
    """Example 2: Create and inspect model"""
    print("\n" + "="*80)
    print("EXAMPLE 2: MODEL CREATION AND INSPECTION")
    print("="*80)
    
    # Create model
    print("\n✓ Creating model...")
    model = create_model(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Backbone: {model_config.backbone}")
    print(f"  Pretrained: {model_config.pretrained}")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    model = model.to(hardware_config.device)
    
    with torch.no_grad():
        outputs = model(x.to(hardware_config.device))
    
    if isinstance(outputs, tuple):
        class_logits, t_scores = outputs
        print(f"  Classification output: {class_logits.shape}")
        print(f"  Regression output: {t_scores.shape}")
    else:
        print(f"  Output shape: {outputs.shape}")
    
    # List model layers
    print(f"\nModel Architecture:")
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: {param.shape}")
    print("  ...")

# ==================== EXAMPLE 3: SINGLE IMAGE INFERENCE ====================
def example_single_inference():
    """Example 3: Single image prediction"""
    print("\n" + "="*80)
    print("EXAMPLE 3: SINGLE IMAGE INFERENCE")
    print("="*80)
    
    # Create model
    print("\n✓ Loading model...")
    model = KneeOADetectionModel(
        backbone="efficientnet_b0",
        num_classes=5,
        include_regression=True
    )
    
    # Create dummy image for demonstration
    print("✓ Creating sample image...")
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    temp_image_path = "temp_xray_sample.jpg"
    Image.fromarray(dummy_image).save(temp_image_path)
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        model_path=None,
        device=hardware_config.device,
        class_names=data_config.class_names,
        preprocessor=ImagePreprocessor()
    )
    
    # Make prediction
    print(f"✓ Making prediction...")
    result = engine.predict_single(temp_image_path)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {result['predicted_label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\n  Class Probabilities:")
    for class_name, prob in sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"    {class_name}: {prob:.4f}")
    
    if 't_score' in result:
        print(f"\n  T-Score: {result['t_score']:.2f}")
        if result['t_score'] > -1:
            status = "Normal"
        elif result['t_score'] > -2.5:
            status = "Osteopenia"
        else:
            status = "Osteoporosis"
        print(f"  Bone Health: {status}")
    
    # Format for display
    formatted = PredictionFormatter.format_prediction(result)
    print(f"\nFormatted Output:\n{formatted}")
    
    # Cleanup
    Path(temp_image_path).unlink()

# ==================== EXAMPLE 4: BATCH INFERENCE ====================
def example_batch_inference():
    """Example 4: Batch predictions"""
    print("\n" + "="*80)
    print("EXAMPLE 4: BATCH INFERENCE")
    print("="*80)
    
    # Create model
    model = KneeOADetectionModel()
    engine = InferenceEngine(model, None, hardware_config.device, data_config.class_names)
    
    # Create sample images
    print("✓ Creating 3 sample images...")
    image_paths = []
    for i in range(3):
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        path = f"temp_sample_{i}.jpg"
        Image.fromarray(dummy_image).save(path)
        image_paths.append(path)
    
    # Batch prediction
    print("✓ Running batch inference...")
    results = engine.predict_batch(image_paths)
    
    # Print results
    print(f"\nBatch Results ({len(results)} images):")
    for i, result in enumerate(results):
        if 'error' not in result:
            print(f"  {i+1}. {result['predicted_label']} ({result['confidence']:.2%})")
    
    formatted = PredictionFormatter.format_batch_predictions(results)
    print(f"\nFormatted Batch Output:\n{formatted}")
    
    # Cleanup
    for path in image_paths:
        Path(path).unlink()

# ==================== EXAMPLE 5: EXPLAINABILITY ====================
def example_explainability():
    """Example 5: Grad-CAM visualization"""
    print("\n" + "="*80)
    print("EXAMPLE 5: EXPLAINABILITY WITH GRAD-CAM")
    print("="*80)
    
    # Create model
    model = KneeOADetectionModel()
    model = model.to(hardware_config.device)
    model.eval()
    
    # Create Grad-CAM
    print("\n✓ Setting up Grad-CAM...")
    gradcam = GradCAM(model, target_layer_name="features")
    
    # Create sample image
    print("✓ Creating sample image...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(dummy_image).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(hardware_config.device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        pred_class = torch.argmax(logits, dim=1)[0].item()
    
    # Compute Grad-CAM
    print(f"✓ Computing Grad-CAM for class {pred_class} ({data_config.class_names[pred_class]})...")
    cam = gradcam.compute_gradcam(image_tensor, pred_class)
    
    if cam is not None:
        print(f"  Grad-CAM shape: {cam.shape}")
        print(f"  Grad-CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
        print("✓ Grad-CAM visualization generated successfully")
    else:
        print("✗ Grad-CAM generation failed")

# ==================== EXAMPLE 6: METRICS CALCULATION ====================
def example_metrics():
    """Example 6: Calculate evaluation metrics"""
    print("\n" + "="*80)
    print("EXAMPLE 6: EVALUATION METRICS")
    print("="*80)
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(5, data_config.class_names)
    
    print("✓ Creating sample predictions...")
    
    # Generate sample predictions and labels
    num_samples = 100
    true_labels = torch.randint(0, 5, (num_samples,))
    logits = torch.randn(num_samples, 5)
    
    # Add some correct predictions
    for i in range(0, num_samples, 3):
        logits[i, true_labels[i]] = 10.0  # Make it easier to predict
    
    # Update metrics
    metrics_calc.update(logits, true_labels)
    
    # Calculate metrics
    metrics = metrics_calc.calculate()
    
    print("\nCalculated Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Get confusion matrix
    cm = metrics_calc.get_confusion_matrix()
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Get classification report
    report = metrics_calc.get_classification_report()
    print(f"\nClassification Report:\n{report}")

# ==================== EXAMPLE 7: CONFIGURATION ====================
def example_configuration():
    """Example 7: View and modify configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 7: CONFIGURATION MANAGEMENT")
    print("="*80)
    
    print("\nData Configuration:")
    print(f"  Class Names: {data_config.class_names}")
    print(f"  Image Size: {data_config.image_size}")
    print(f"  Batch Size: {data_config.batch_size}")
    print(f"  Augmentation Enabled: {data_config.augmentation_enabled}")
    
    print("\nModel Configuration:")
    print(f"  Backbone: {model_config.backbone}")
    print(f"  Pretrained: {model_config.pretrained}")
    print(f"  Num Classes: {model_config.num_classes}")
    print(f"  Include T-Score Head: {model_config.include_t_score_head}")
    
    print("\nTraining Configuration:")
    print(f"  Num Epochs: {training_config.num_epochs}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Optimizer: {training_config.optimizer}")
    print(f"  Early Stopping: {training_config.early_stopping_enabled}")
    
    print("\nHardware Configuration:")
    print(f"  Device: {hardware_config.device}")
    print(f"  Use GPU: {hardware_config.use_gpu}")
    print(f"  Mixed Precision: {training_config.mixed_precision}")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  KNEE OA DETECTION SYSTEM - USAGE EXAMPLES".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        # Run all examples
        example_configuration()
        example_data_loading()
        example_model_creation()
        example_single_inference()
        example_batch_inference()
        example_explainability()
        example_metrics()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Review ARCHITECTURE.md for system design")
        print("2. Check DEPLOYMENT.md for deployment instructions")
        print("3. Run 'python train.py' to start training")
        print("4. Run 'python ui/app.py' to launch web interface")
        print("\n")
    
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
