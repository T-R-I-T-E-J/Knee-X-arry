import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.landmark_model import KneeLandmarkDetector

class LandmarkDataset(Dataset):
    """
    Expects a JSON file mapping image filename to an array of 6 normalized coordinates:
    {"knee_001.jpg": [0.2, 0.5, 0.5, 0.53, 0.8, 0.55], ... }
    """
    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Annotation file missing: {json_file}. You must create this coordinate file first!")
            
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
        self.filenames = list(self.data.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = self.image_dir / fname
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Target coordinates: [x_medial, y_medial, x_center, y_center, x_lateral, y_lateral]
        coords = torch.tensor(self.data[fname], dtype=torch.float32)
        return image, coords

def train_model(epochs=50, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[#] Initializing training on: {device}")
    
    # 1. Image preprocessing (must match backbone expectation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Setup Dataset & Loader
    try:
        dataset = LandmarkDataset(image_dir='./data/images', json_file='./data/landmark_labels.json', transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError as e:
        print(f"\n[!] DATASET ERROR: {e}")
        print("[!] Stopping script. Please refer to docs/TRAIN_LANDMARK_MODEL.md for setup instructions.")
        sys.exit(1)
        
    print(f"[#] Training loaded with {len(dataset)} annotated images.")
    
    # 3. Model, Loss, Optimizer
    model = KneeLandmarkDetector().to(device)
    # Using L2 Distance (MSE) which penalizes pixel distance errors exponentially
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'MSE_Loss': loss.item()})
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} complete | Average Error Loss: {epoch_loss:.6f}")
        
        # Save best local checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_landmark_model.pt")
            print(" -> Checkpoint saved: New lowest spatial error!")

if __name__ == "__main__":
    train_model()
