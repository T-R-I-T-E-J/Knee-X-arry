import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.segmentation_model import KneeSegmentationUNet

class KneeMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale
        
        if self.transform:
            image = self.transform(image)
            
            # Mask just needs resizing and to be converted to [0, 1] tensor
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
            
        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous()
        target = target.contiguous()
        
        intersection = (predict * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / 
                     (predict.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

def train_unet(epochs=20, batch_size=4, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[#] Initializing Semantic UNet on: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = KneeMaskDataset(
        image_dir='data/segmentation/images',
        mask_dir='data/segmentation/masks',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = KneeSegmentationUNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Dice_Loss': loss.item()})
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} Average Dice Error: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/segmentation_unet.pt")
            print(" -> Checkpoint saved!")

if __name__ == "__main__":
    train_unet()
