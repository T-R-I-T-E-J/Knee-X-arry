# Deployment & Setup Guide

## 🖥️ Environment Setup

### Windows Setup

```batch
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Linux/Mac Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start Checklist

- [ ] Create and activate virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Organize dataset into class folders (0-4)
- [ ] Update dataset paths in `configs/config.py`
- [ ] Run training: `python train.py`
- [ ] Verify model output: `outputs/evaluation_metrics.json`
- [ ] Launch UI: `python ui/app.py`

## 📦 Docker Deployment (Optional)

### Create Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 7860

# Run Gradio app
CMD ["python", "ui/app.py", "--server_name", "0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t knee-oa-ai:latest .

# Run container
docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models knee-oa-ai:latest

# On Windows (PowerShell)
docker run --gpus all -p 7860:7860 -v ${PWD}\models:/app/models knee-oa-ai:latest

# Access at http://localhost:7860
```

## 🌐 Cloud Deployment

### AWS Deployment

```bash
# 1. Create EC2 instance (GPU enabled)
# Instance type: p3.2xlarge or g4dn.xlarge
# AMI: Ubuntu 22.04 LTS + NVIDIA CUDA drivers

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Setup environment
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# 4. Clone repository
git clone your-repo-url
cd KneeOA_AI

# 5. Setup and run
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Download model checkpoint from S3
aws s3 cp s3://your-bucket/models/best_model.pt models/

# 7. Run with Gunicorn
pip install gunicorn
gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker \
         --bind 0.0.0.0:7860 ui.app:demo
```

### Azure Container Instances

```bash
# 1. Build and push image to Azure Container Registry
az acr build --registry your-registry --image knee-oa-ai:latest .

# 2. Deploy container instance
az container create \
  --resource-group your-rg \
  --name knee-oa-ai \
  --image your-registry.azurecr.io/knee-oa-ai:latest \
  --gpu 1 \
  --cpu 4 \
  --memory 16 \
  --ports 7860 \
  --ip-address public
```

### Google Cloud Run (CPU only)

```bash
# Build and deploy
gcloud run deploy knee-oa-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=models/best_model.pt
```

## 🔐 Security Considerations

### Model Security
```python
# Encrypt model weights
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt model
with open('models/best_model.pt', 'rb') as f:
    model_data = f.read()
encrypted = cipher.encrypt(model_data)

with open('models/best_model.pt.encrypted', 'wb') as f:
    f.write(encrypted)

# Decrypt on load
encrypted_data = cipher.decrypt(encrypted)
state_dict = torch.load(io.BytesIO(encrypted_data))
```

### Input Validation
```python
# Validate image inputs
import logging
from PIL import Image

def validate_xray_image(image_path):
    """Validate X-ray image format and size"""
    try:
        img = Image.open(image_path)
        
        # Check format
        if img.format not in ['JPEG', 'PNG', 'TIFF']:
            raise ValueError(f"Invalid format: {img.format}")
        
        # Check dimensions
        if img.size[0] < 256 or img.size[1] < 256:
            raise ValueError("Image too small (min 256x256)")
        
        if img.size[0] > 4096 or img.size[1] > 4096:
            raise ValueError("Image too large (max 4096x4096)")
        
        return True
    
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False
```

## 📊 Performance Tuning

### Batch Size Optimization
```python
# Benchmark different batch sizes
batch_sizes = [8, 16, 32, 64, 128]

for bs in batch_sizes:
    data_config.batch_size = bs
    
    # Measure throughput
    start = time.time()
    for _ in range(100):
        # Run inference
        pass
    throughput = 100 / (time.time() - start)
    print(f"Batch size {bs}: {throughput:.1f} img/s")
```

### Mixed Precision Training
```python
# Enable automatic mixed precision for faster training
training_config.mixed_precision = True
training_config.use_gradient_accumulation = True
training_config.gradient_accumulation_steps = 4

# This reduces memory usage and speeds up training on modern GPUs
```

### Model Quantization
```python
# Post-training quantization (reduces model size 4x)
from torch.quantization import quantize_dynamic

model = KneeOADetectionModel()
model.load_state_dict(torch.load('models/best_model.pt'))

quantized_model = quantize_dynamic(
    model,
    {nn.Linear, nn.BatchNorm1d},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'models/best_model_quantized.pt')
```

## 🧪 Testing & Validation

### Unit Tests
```python
# test_model.py
import unittest
import torch
from src.model import KneeOADetectionModel

class TestModel(unittest.TestCase):
    
    def setUp(self):
        self.model = KneeOADetectionModel()
    
    def test_forward_pass(self):
        x = torch.randn(2, 3, 224, 224)
        outputs = self.model(x)
        self.assertEqual(outputs[0].shape, torch.Size([2, 5]))
    
    def test_inference_shapes(self):
        x = torch.randn(1, 3, 224, 224)
        class_logits, t_scores = self.model(x)
        self.assertEqual(class_logits.shape[1], 5)
        self.assertEqual(t_scores.shape[1], 1)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests
```python
# test_pipeline.py
def test_training_pipeline():
    """Test complete training pipeline"""
    from src.data_loader import KneeXRayDataLoader
    from src.model import create_model
    from src.trainer import setup_training
    
    # Create mock data
    data_loader = KneeXRayDataLoader(...)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # Create and train model
    model = create_model(model_config)
    trainer = setup_training(model, train_loader, val_loader, training_config)
    
    # Run one epoch
    history = trainer.train()
    
    # Verify results
    assert 'train_loss' in history
    assert history['train_loss'][-1] > 0
```

## 📈 Monitoring & Logging

### TensorBoard Monitoring
```bash
# Start TensorBoard
tensorboard --logdir outputs/tensorboard_logs

# Open http://localhost:6006 in browser
```

### Logging Configuration
```python
# Customize logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

## 🔄 Continuous Integration (CI/CD)

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Lint code
      run: |
        flake8 src/
        black --check src/
```

## 📝 Production Checklist

- [ ] Test all components locally
- [ ] Verify GPU/CUDA availability
- [ ] Configure security credentials
- [ ] Setup model versioning
- [ ] Enable logging and monitoring
- [ ] Create backup strategy
- [ ] Document troubleshooting steps
- [ ] Setup alerts for failures
- [ ] Test disaster recovery
- [ ] Review data privacy policies
- [ ] Obtain medical/ethical approval if applicable
- [ ] Setup user authentication (if needed)

## 🆘 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
data_config.batch_size = 8  # Default: 32

# Enable gradient accumulation
training_config.use_gradient_accumulation = True
training_config.gradient_accumulation_steps = 4

# Use smaller model
model_config.backbone = "efficientnet_b0"  # Instead of b7
```

### Slow Training
```python
# Increase number of workers
data_config.num_workers = 8  # Default: 4

# Enable mixed precision
training_config.mixed_precision = True

# Use larger batch size (if memory allows)
data_config.batch_size = 64  # From 32
```

### Poor Model Performance
```python
# Debug data loading
from src.data_loader import KneeXRayDataLoader
loader = KneeXRayDataLoader(...)
train_loader, _, _ = loader.create_dataloaders()

# Inspect batch
images, labels = next(iter(train_loader))
print(f"Image stats: min={images.min()}, max={images.max()}")
print(f"Label distribution: {np.bincount(labels.numpy())}")

# Visualize samples
import matplotlib.pyplot as plt
plt.imshow(images[0].transpose(0, 2).squeeze())
```

---

**Guide Version:** 1.0  
**Last Updated:** April 2026
