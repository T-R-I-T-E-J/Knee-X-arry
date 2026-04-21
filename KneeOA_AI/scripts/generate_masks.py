import cv2
import numpy as np
import os
import glob
from pathlib import Path

def generate_pseudo_mask(img_path, output_mask_path, output_img_path):
    """Uses OpenCV structural morphology to generate clinical bone masks from clean data"""
    image = cv2.imread(img_path)
    if image is None: return False
    
    # 1. Image Enhancement
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_enhanced = clahe.apply(image_gray)
    image_denoised = cv2.bilateralFilter(image_enhanced, d=5, sigmaColor=10, sigmaSpace=10)
    
    h_img, w_img = image_denoised.shape
    
    # 2. Otsu Threshold + Heavy morphological close
    _, binary = cv2.threshold(image_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Contour Filtering (same logic built in the analyzer)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_bones = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < w_img * 0.1 and h > h_img * 0.4: continue 
        if y > h_img * 0.85: continue 
        if cv2.contourArea(c) > 5000:
            valid_bones.append(c)
            
    if not valid_bones:
        return False
        
    valid_bones = sorted(valid_bones, key=cv2.contourArea, reverse=True)
    if w_img > h_img * 1.2:
        valid_bones = [c for c in valid_bones if cv2.boundingRect(c)[0] < w_img // 2]
    
    if len(valid_bones) > 0:
        left_bones = [c for c in valid_bones if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2) < w_img * 0.5]
        right_bones = [c for c in valid_bones if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2) > w_img * 0.5]
        if len(left_bones) > 0 and len(right_bones) > 0:
            valid_bones = left_bones
    else:
        return False
        
    # Limit to top 2 massive constraints (Femur and Tibia)
    valid_bones = valid_bones[:2]

    # Create empty solid black mask and color the bone white
    bone_mask = np.zeros_like(binary)
    cv2.drawContours(bone_mask, valid_bones, -1, 255, -1)
    
    # Only save if it generated a substantial mask
    if np.sum(bone_mask == 255) > 5000:
        cv2.imwrite(output_mask_path, bone_mask)
        cv2.imwrite(output_img_path, cv2.resize(image, (w_img, h_img)))
        return True
    return False

def main():
    source_dir = Path("../MedicalExpert-I")
    mask_out = Path("data/segmentation/masks")
    img_out = Path("data/segmentation/images")
    
    print(f"Scanning for images in {source_dir}...")
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        images.extend(source_dir.rglob(ext))
        
    print(f"Found {len(images)} potential images. Generating Pseudo-label Masks...")
    success_count = 0
    limit = 50 # We only need a starter batch of 50 images to train the U-Net
    
    for img_path in images:
        if success_count >= limit:
            break
            
        fname = img_path.name
        o_mask = str(mask_out / fname)
        o_img = str(img_out / fname)
        
        if generate_pseudo_mask(str(img_path), o_mask, o_img):
            success_count += 1
            print(f"[{success_count}/{limit}] Generated mask for {fname}")

if __name__ == "__main__":
    main()
