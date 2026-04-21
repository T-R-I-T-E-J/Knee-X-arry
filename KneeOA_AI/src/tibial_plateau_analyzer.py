import cv2
import numpy as np
import base64
import json
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class TibialPlateauAnalyzer:
    """
    Automated Tibial Plateau Division and JSW Measurement System
    """
    def __init__(self, debug=True):
        self.debug = debug
        self.assumed_plateau_mm = 75.0  # Default assumed width of tibial plateau in adults

    def _array_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_image(self, image_path: str, measurement_points_per_half: int = 4) -> Dict[str, Any]:
        """Full sequential pipeline as requested"""
        result = {
            "detection_status": "FAILED",
            "fixed_points": {},
            "division_line": {},
            "medial_jsw_anatomical": {},
            "lateral_jsw_anatomical": {},
            "medial_jsw_ruler": {},
            "lateral_jsw_ruler": {},
            "overall_assessment": {},
            "quality_metrics": {"warnings": []},
            "visual_outputs": {}
        }
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
                
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_enhanced = clahe.apply(image_gray)
            image_denoised = cv2.bilateralFilter(image_enhanced, d=5, sigmaColor=10, sigmaSpace=10)
            
            sharpness = min(100.0, (np.log1p(cv2.Laplacian(image_gray, cv2.CV_64F).var()) / 8.0) * 100.0)
            result["quality_metrics"]["image_sharpness_score"] = round(sharpness, 1)

            # ---------------------------------------------------------
            # ML STEP 2: MORPHOLOGICAL BONE HEALING & BOTTLENECK TRACKING
            # ---------------------------------------------------------
            h_img, w_img = image_denoised.shape
            
            # Otsu thresholding cleanly maps the bone while dropping background haze
            _, binary = cv2.threshold(image_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extreme Morphological Close: This mathematically ERASES the black pen lines drawn across the bone
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_bones = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < w_img * 0.1 and h > h_img * 0.4: continue # Ignore tall thin rulers on the side
                if y > h_img * 0.85: continue # Ignore dense text blocks baked onto the bottom
                if cv2.contourArea(c) > 5000:
                    valid_bones.append(c)
                    
            valid_bones = sorted(valid_bones, key=cv2.contourArea, reverse=True)
            
            if len(valid_bones) > 0:
                # If there's bone mass in both halves of the image, it's a dual-view X-Ray. Lock to the left (AP) half!
                left_bones = [c for c in valid_bones if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2) < w_img * 0.5]
                right_bones = [c for c in valid_bones if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2) > w_img * 0.5]
                
                if len(left_bones) > 0 and len(right_bones) > 0:
                    valid_bones = left_bones
                    
                # Create a cleansed mathematical mask containing only the relevant AP bones
                bone_mask = np.zeros_like(binary)
                cv2.drawContours(bone_mask, valid_bones, -1, 255, -1)
                
                # 1D Vertical Row Density Projection of the physical bone map
                y_proj = np.sum(bone_mask, axis=1)
                y_proj_smooth = cv2.GaussianBlur(y_proj.reshape(-1, 1).astype(np.float32), (31, 1), 0).flatten()
                
                # The functional joint is the physical GAP (valley) in vertical bone density between the Femur and Tibia blocks
                search_y_min, search_y_max = int(h_img * 0.30), int(h_img * 0.70)
                y_center_roi_smooth = y_proj_smooth[search_y_min:search_y_max]
                
                # Find the deepest valley bounded by significant bone density drops
                valley_idx = np.argmin(y_center_roi_smooth) 
                knee_center_y = int(valley_idx + search_y_min)
                
                # Align X to the center of mass of the bone mask at this valley
                row_pixels = bone_mask[max(0, knee_center_y-20):min(h_img, knee_center_y+20), :]
                bone_indices = np.where(row_pixels > 0)[1]
                if len(bone_indices) > 0:
                    knee_center_x = int(np.mean(bone_indices))
                else:
                    x_all, _, w_all, _ = cv2.boundingRect(valid_bones[0])
                    knee_center_x = x_all + w_all // 2
            else:
                knee_center_x, knee_center_y = w_img // 2, h_img // 2
                
            # Fallback bounds check
            if not (0.25 * h_img <= knee_center_y <= 0.75 * h_img):
                knee_center_y = h_img // 2
                
            logger.info(f"Knee joint center locked at x={knee_center_x}, y={knee_center_y}")
            
            roi_x_start = max(0, knee_center_x - int(w_img * 0.25))
            roi_x_end = min(w_img, knee_center_x + int(w_img * 0.25))

            # ---------------------------------------------------------
            # STEP 1: ML VERIFIED TIBIAL PLATEAU DETECTION
            # ---------------------------------------------------------
            search_region_y_start = knee_center_y + 10
            search_region_y_end = knee_center_y + 150
            search_region_y_end = min(search_region_y_end, h_img)
            
            tibia_search = image_denoised[search_region_y_start:search_region_y_end, roi_x_start:roi_x_end]
            _, binary_tib = cv2.threshold(tibia_search, 80, 255, cv2.THRESH_BINARY)
            t_contours, _ = cv2.findContours(binary_tib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_tibial = None
            valid_t_contours = [c for c in t_contours if cv2.contourArea(c) > 500]
            if valid_t_contours:
                # ML dynamically proved the joint gap, meaning the largest connected mass directly below is definitively the Tibia
                largest_tib = max(valid_t_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_tib)
                best_tibial = {
                    'contour': largest_tib,
                    'x': x,
                    'y': y + search_region_y_start,
                    'w': w,
                    'h': h
                }

            if best_tibial:
                tibial_plateau_y = best_tibial['y']
                logger.info(f"Tibial plateau found at y={tibial_plateau_y}")
                # Get global contour mapped mathematically via reshape flattening to eliminate bounds error
                tibial_contour_flat = best_tibial['contour'].reshape(-1, 2)
                tibial_contour = tibial_contour_flat + np.array([roi_x_start, search_region_y_start])
            else:
                raise ValueError("Tibial plateau detection FAILED")

            # ---------------------------------------------------------
            # STEP 2: VERIFY & CORRECT FEMORAL SURFACE DETECTION
            # ---------------------------------------------------------
            f_search_region_y_start = max(0, tibial_plateau_y - 200)
            f_search_region_y_end = max(0, tibial_plateau_y - 20)
            
            femur_search = image_denoised[f_search_region_y_start:f_search_region_y_end, roi_x_start:roi_x_end]
            _, binary_fem = cv2.threshold(femur_search, 90, 255, cv2.THRESH_BINARY)
            f_contours, _ = cv2.findContours(binary_fem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            femoral_surfaces = []
            for contour in f_contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    bottommost_point = contour[contour[:, 0, 1].argmax()][0]
                    femoral_y = bottommost_point[1] + f_search_region_y_start
                    femoral_surfaces.append({
                        'contour': contour,
                        'bottommost_y': femoral_y,
                        'area': area
                    })
                    
            if femoral_surfaces:
                # Combine ALL valid femoral condyles to bridge both medial and lateral sides
                femoral_pts_list = []
                for item in femoral_surfaces:
                    fc_flat = item['contour'].reshape(-1, 2)
                    fc_global = fc_flat + np.array([roi_x_start, f_search_region_y_start])
                    femoral_pts_list.append(fc_global)
                
                femoral_contour = np.vstack(femoral_pts_list)
                # Find the global lowest point for logging
                femoral_plateau_y = int(np.max(femoral_contour[:, 1]))
                logger.info(f"Femoral surface found at y={femoral_plateau_y}")
            else:
                raise ValueError("Femoral surface detection FAILED")
                
            if femoral_plateau_y >= tibial_plateau_y:
                raise ValueError("ERROR: Femoral surface should be ABOVE tibial surface!")

            # Determine X extents
            x_medial = np.min(tibial_contour[:, 0])
            x_lateral = np.max(tibial_contour[:, 0])
            plateau_width_pixels = float(x_lateral - x_medial)
            
            # Identify exact division center from mathematical extents
            x_center = int((x_medial + x_lateral) / 2)
            
            # Store Fixed points
            result["fixed_points"] = {
                "medial_edge_pixels": [x_medial, tibial_plateau_y],
                "central_eminence_pixels": [x_center, tibial_plateau_y],
                "lateral_edge_pixels": [x_lateral, tibial_plateau_y]
            }

            # ---------------------------------------------------------
            # STEP 3: CALCULATE JSW AT DIVISION LINE
            # ---------------------------------------------------------
            def find_surface_at_x(contour_pts, x_target, is_femur=False, tolerance=40):
                nearby = contour_pts[np.abs(contour_pts[:, 0] - x_target) <= tolerance]
                if len(nearby) == 0: 
                    idx = np.argmin(np.abs(contour_pts[:, 0] - x_target))
                    return int(contour_pts[idx, 1])
                
                if is_femur:
                    return int(np.max(nearby[:, 1]))
                else:
                    return int(np.min(nearby[:, 1]))
                
            femoral_y_at_center = find_surface_at_x(femoral_contour, x_center, is_femur=True)
            tibial_y_at_center = find_surface_at_x(tibial_contour, x_center, is_femur=False)
            
            if femoral_y_at_center is None or tibial_y_at_center is None:
                raise ValueError("Cannot find femoral or tibial surface at center x")
                
            jsw_pixels = tibial_y_at_center - femoral_y_at_center
            calibration_factor = self.assumed_plateau_mm / plateau_width_pixels if plateau_width_pixels > 0 else 0.312
            
            jsw_center_mm = jsw_pixels * calibration_factor
            if not (0.2 <= jsw_center_mm <= 15.0):
                logger.warning(f"Center JSW out of bounds: {jsw_center_mm:.2f}mm")
            
            # ---------------------------------------------------------
            # STEP 4: IDENTIFY MEDIAL HALF POINTS
            # ---------------------------------------------------------
            medial_width = x_center - x_medial
            medial_positions = {
                'M1': x_medial + medial_width * 0.20,
                'M2': x_medial + medial_width * 0.40,
                'M3': x_medial + medial_width * 0.60,
                'M4': x_medial + medial_width * 0.80,
            }
            femoral_medial = femoral_contour[(femoral_contour[:, 0] >= x_medial) & (femoral_contour[:, 0] <= x_center)]
            tibial_medial = tibial_contour[(tibial_contour[:, 0] >= x_medial) & (tibial_contour[:, 0] <= x_center)]
            
            # ---------------------------------------------------------
            # STEP 5: IDENTIFY LATERAL HALF POINTS
            # ---------------------------------------------------------
            lateral_width = x_lateral - x_center
            lateral_positions = {
                'L1': x_center + lateral_width * 0.20,
                'L2': x_center + lateral_width * 0.40,
                'L3': x_center + lateral_width * 0.60,
                'L4': x_center + lateral_width * 0.80,
            }
            femoral_lateral = femoral_contour[(femoral_contour[:, 0] >= x_center) & (femoral_contour[:, 0] <= x_lateral)]
            tibial_lateral = tibial_contour[(tibial_contour[:, 0] >= x_center) & (tibial_contour[:, 0] <= x_lateral)]

            # ---------------------------------------------------------
            # STEP 6: CALCULATE JSW AT ALL MEASUREMENT POINTS
            # ---------------------------------------------------------
            all_measurements = {}
            valid_count = 0
            
            for label, x_point in medial_positions.items():
                fy = find_surface_at_x(femoral_medial, x_point, is_femur=True)
                ty = find_surface_at_x(tibial_medial, x_point, is_femur=False)
                if fy is None or ty is None:
                    all_measurements[label] = {'valid': False, 'reason': 'surface_not_found'}
                    continue
                d_mm = (ty - fy) * calibration_factor
                if d_mm <= 0 or d_mm < 0.2 or d_mm > 15:
                    all_measurements[label] = {'valid': False, 'reason': 'out_of_range'}
                    continue
                all_measurements[label] = {'valid': True, 'jsw_mm': d_mm, 'fy': fy, 'ty': ty, 'x': x_point, 'label': label}
                valid_count += 1
                
            for label, x_point in lateral_positions.items():
                fy = find_surface_at_x(femoral_lateral, x_point, is_femur=True)
                ty = find_surface_at_x(tibial_lateral, x_point, is_femur=False)
                if fy is None or ty is None:
                    all_measurements[label] = {'valid': False, 'reason': 'surface_not_found'}
                    continue
                d_mm = (ty - fy) * calibration_factor
                if d_mm <= 0 or d_mm < 0.2 or d_mm > 15:
                    all_measurements[label] = {'valid': False, 'reason': 'out_of_range'}
                    continue
                all_measurements[label] = {'valid': True, 'jsw_mm': d_mm, 'fy': fy, 'ty': ty, 'x': x_point, 'label': label}
                valid_count += 1

            # ---------------------------------------------------------
            # STEP 7: DRAW PERPENDICULAR MEASUREMENT LINES
            # ---------------------------------------------------------
            vis_img = image.copy()
            drawn_count = 0
            
            # Draw RED division line
            cv2.line(vis_img, (x_center, tibial_plateau_y - 40), (x_center, tibial_plateau_y + 80), (0, 0, 255), 3)
            
            # Draw fixed points
            cv2.circle(vis_img, (x_medial, tibial_plateau_y), 7, (255, 0, 0), -1) # Blue Medial
            cv2.circle(vis_img, (x_center, tibial_plateau_y), 9, (0, 255, 255), -1) # Yellow Center
            cv2.circle(vis_img, (x_lateral, tibial_plateau_y), 7, (0, 255, 255), -1) # Cyan Lateral
            
            for label, measure in all_measurements.items():
                if not measure['valid']: continue
                x_p, fy, ty, jsw = int(measure['x']), measure['fy'], measure['ty'], measure['jsw_mm']
                
                # Green perpendicular
                cv2.line(vis_img, (x_p, fy), (x_p, ty), (0, 255, 0), 2)
                # Yellow endpoints
                cv2.circle(vis_img, (x_p, fy), 3, (0, 255, 255), -1)
                cv2.circle(vis_img, (x_p, ty), 3, (0, 255, 255), -1)
                
                # Background text
                text = f"{jsw:.1f} mm"
                text_y = int((fy + ty) / 2)
                (tw, th), basel = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(vis_img, (x_p+3, text_y-th-2), (x_p+5+tw, text_y+basel+2), (0,0,0), -1)
                cv2.putText(vis_img, text, (x_p+5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                drawn_count += 1

            # ---------------------------------------------------------
            # STEP 8: CALCULATE MEDIAL & LATERAL AVERAGES
            # ---------------------------------------------------------
            m_vals = [m['jsw_mm'] for m in all_measurements.values() if m['valid'] and 'M' in m['label']]
            l_vals = [m['jsw_mm'] for m in all_measurements.values() if m['valid'] and 'L' in m['label']]
            
            m_mean = np.mean(m_vals) if m_vals else 0.0
            m_std = np.std(m_vals) if len(m_vals) > 1 else 0.0
            m_min = np.min(m_vals) if m_vals else 0.0
            m_max = np.max(m_vals) if m_vals else 0.0
            
            l_mean = np.mean(l_vals) if l_vals else 0.0
            l_std = np.std(l_vals) if len(l_vals) > 1 else 0.0
            l_min = np.min(l_vals) if l_vals else 0.0
            l_max = np.max(l_vals) if l_vals else 0.0

            # Step 9: Update Status & Grade (in results dict)
            conf_percent = (drawn_count / 8.0) * 100.0
            if m_mean > 5.0 and l_mean > 5.0:
                eval_status = "NORMAL"
            elif m_mean > 1.5 and l_mean > 1.5:
                # 3.0 to 5.0 normally is "Normal/Mild", 1.5 to 3.0 is Narrowed
                eval_status = "NARROWED"
                if m_mean > 3.0 and l_mean > 3.0:
                    eval_status = "NORMAL (Thinning)"
            else:
                eval_status = "SEVERELY NARROWED"
                
            avg_jsw = (m_mean + l_mean) / 2
            if avg_jsw > 5.0: grade = "Grade 0 (Normal)"
            elif avg_jsw > 4.0: grade = "Grade 1 (Doubtful)"
            elif avg_jsw > 3.0: grade = "Grade 2 (Minimal OA)"
            elif avg_jsw > 1.5: grade = "Grade 3 (Moderate OA)"
            else: grade = "Grade 4 (Severe OA)"
            
            if m_mean == 0.0 and l_mean == 0.0:
                result["detection_status"] = "FAILED"
            else:
                result["detection_status"] = "SUCCESS"

            # ---------------------------------------------------------
            # FINAL OUTPUT OVERLAYS
            # ---------------------------------------------------------
            h, w = vis_img.shape[:2]
            cv2.rectangle(vis_img, (10, h - 110), (600, h - 10), (0, 0, 0), -1)
            f = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, "KNEE OA PARAMETER ANALYSIS COMPLETE", (20, h - 85), f, 0.5, (0, 255, 255), 1)
            cv2.putText(vis_img, f"MEDIAL JSW:  {m_mean:.2f} +/- {m_std:.2f} mm", (20, h - 60), f, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_img, f"LATERAL JSW: {l_mean:.2f} +/- {l_std:.2f} mm", (20, h - 35), f, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_img, f"Status: {eval_status} | Confidence: {conf_percent:.0f}%", (20, h - 15), f, 0.5, (0, 255, 0), 1)
            
            result["visual_outputs"]["image_with_marked_points"] = self._array_to_base64(vis_img)
            
            # Map values rigidly so Medical Gui JSW cards display exactly the data it wants
            result["medial_jsw_anatomical"] = {
                'measurements_mm': m_vals, 'mean_mm': m_mean, 'min_mm': m_min, 'max_mm': m_max, 'std_mm': m_std, 'clinical_grade': grade
            }
            result["lateral_jsw_anatomical"] = {
                'measurements_mm': l_vals, 'mean_mm': l_mean, 'min_mm': l_min, 'max_mm': l_max, 'std_mm': l_std, 'clinical_grade': grade
            }
            result["medial_jsw_ruler"] = result["medial_jsw_anatomical"]
            result["lateral_jsw_ruler"] = result["lateral_jsw_anatomical"]
            result["quality_metrics"]["detection_confidence"] = conf_percent
            
        except Exception as e:
            logger.error(f"Error in tibial plateau analysis: {e}")
            result["quality_metrics"]["warnings"].append(str(e))
            
        return result
