"""
Automated Knee Joint Detection and Cropping (Auto-Cutter).
Mimics a YOLO detector by finding the anatomical center of the joint space.
"""

import cv2
import numpy as np
from PIL import Image

class AutoCutter:
    def __init__(self, crop_size=448):
        self.crop_size = crop_size

    def detect_joint_center(self, img_np):
        """
        Heuristic anatomical detector. 
        Knee joints have high horizontal edge density in the center.
        """
        # 1. Convert to gray and enhance contrast
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # 2. Find horizontal edges
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        
        # 3. Project horizontal intensity
        # The joint line has the highest concentration of horizontal edges
        h, w = sobel_y.shape
        row_sums = np.sum(sobel_y, axis=1)
        
        # We look for the peak in the middle 60% of the image (ignore top/bottom labels)
        search_region = slice(int(h * 0.2), int(h * 0.8))
        target_y = np.argmax(row_sums[search_region]) + int(h * 0.2)
        
        # 4. Find vertical center (lateral center)
        # Sum columns in the target row area to find the bone mass center
        col_sums = np.sum(gray[target_y-20:target_y+20, :], axis=0)
        target_x = np.argmax(col_sums)
        
        return target_x, target_y

    def crop(self, image):
        """
        Takes a PIL image and returns a cropped PIL image focused on the joint.
        """
        img_np = np.array(image.convert('RGB'))
        h, w, _ = img_np.shape
        
        # Find the knee center
        cx, cy = self.detect_joint_center(img_np)
        
        # Calculate crop bounds
        half_size = min(h, w, 600) // 2 # Determine a safe crop size
        
        y1 = max(0, cy - half_size)
        y2 = min(h, cy + half_size)
        x1 = max(0, cx - half_size)
        x2 = min(w, cx + half_size)
        
        # Ensure we maintain square aspect ratio if possible
        crop_square = img_np[y1:y2, x1:x2]
        return Image.fromarray(crop_square)

def get_cutter_transform(crop_size=448):
    cutter = AutoCutter(crop_size)
    return cutter.crop
