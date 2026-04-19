"""
Geometric Analysis Engine for Medical Imaging.
Includes:
- Sharpness Calculation (Laplacian Variance, Sobel)
- Femur-Tibia Distance (JSW) Detection and Measurement
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class GeometricAnalyzer:
    """Handles pixel-level image analysis for medical measurements."""
    
    def __init__(self, mm_per_pixel: float = 0.1):
        """
        Args:
            mm_per_pixel: Calibration factor for converting pixels to millimeters.
                         Defaults to 0.1mm/px for standard X-ray detectors.
        """
        self.mm_per_pixel = mm_per_pixel

    def calculate_sharpness(self, image: np.ndarray) -> Dict:
        """
        Calculates image sharpness and quality status.
        Args:
            image: Grayscale image as numpy array.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Laplacian Variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Sobel Edge Magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        # Normalized score (0-100) - based on typical medical imaging thresholds
        # Using a logarithmic-like mapping for Laplacian var
        score = min(100.0, (np.log1p(lap_var) / 8.0) * 100.0)
        
        if score > 70:
            status, rec = "EXCELLENT", "Ready for Analysis"
        elif score > 50:
            status, rec = "GOOD", "Acceptable quality"
        elif score > 30:
            status, rec = "ACCEPTABLE", "Suboptimal; review cautiously"
        else:
            status, rec = "POOR", "Image too blurry; retake recommended"
            
        return {
            "score": round(score, 1),
            "laplacian_variance": round(lap_var, 2),
            "sobel_score": round(sobel_mag, 2),
            "status": status,
            "recommendation": rec
        }

    def measure_jsw(self, image: np.ndarray) -> Dict:
        """
        Detects femoral and tibial surfaces to measure Joint Space Width (JSW).
        Note: Production implementation would use a segmentation model,
              this falls back to classical CV (Active Contours/Edge Tracing).
        """
        # Preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Denoise and enhance
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Detect edges
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Find joint region (central horizontal band)
        h, w = edges.shape
        roi = edges[h//3 : 2*h//3, :]
        
        # Simplified surface detection: 
        # Scan columns to find the two strongest edge pixels in the ROI
        medial_dists = []
        lateral_dists = []
        
        points_count = 0
        for col in range(10, w - 10, 5): # Sample every 5 pixels
            edge_indices = np.where(roi[:, col] > 0)[0]
            if len(edge_indices) >= 2:
                # Assuming top edge is femoral condyle, bottom is tibial plateau
                d = (edge_indices[-1] - edge_indices[0]) * self.mm_per_pixel
                if 1.0 < d < 12.0: # Filter physiologically impossible values
                    if col < w // 2:
                        medial_dists.append(d)
                    else:
                        lateral_dists.append(d)
                    points_count += 1
                    
        # Stats summary
        def get_stats(dists):
            if not dists: return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            return {
                "mean": round(np.mean(dists), 2),
                "min": round(np.min(dists), 2),
                "max": round(np.max(dists), 2),
                "std": round(np.std(dists), 2)
            }
            
        return {
            "medial": get_stats(medial_dists),
            "lateral": get_stats(lateral_dists),
            "points_sampled": points_count,
            "confidence": min(1.0, points_count / 100.0) # Heuristic confidence
        }

    def analyze_contours(self, image: np.ndarray) -> Dict:
        """Analyzes bone contour roughness and deviation."""
        # Simplified: uses laplacian of the edge map as roughness proxy
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        roughness = np.std(edges) / 10.0 # Heuristic normalization
        
        return {
            "severity_grade": 1 if roughness < 5 else 2,
            "roughness_index": round(min(10.0, roughness), 1),
            "shape_deviation": round(roughness * 0.5, 2) # mm proxy
        }
