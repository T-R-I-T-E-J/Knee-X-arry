# DETAILED SYSTEM PROMPT FOR CLAUDE/ANTIGRAVITY
## Automatic Tibial Plateau Division & JSW Measurement System

### OBJECTIVE

Create a computer vision system that automatically:
1. **Detects the tibial plateau** from knee X-ray images
2. **Identifies anatomical fixed points** on the tibial surface
3. **Divides the tibial plateau** into medial and lateral halves using mathematical precision
4. **Measures Joint Space Width (JSW)** in each half
5. **Returns structured measurements** with visual overlays

---

## PART 1: ANATOMICAL REFERENCE & MEASUREMENT POINTS

### 1.1 Tibial Plateau Anatomy

**Definition:**
The tibial plateau is the flat or slightly curved upper surface of the tibia (shinbone) that articulates with the femoral condyles. On X-ray (anteroposterior view), it appears as:
- **Medial plateau:** Inner (toward midline) portion of the tibia
- **Lateral plateau:** Outer (toward side) portion of the tibia
- **Central ridge/eminence:** Slight elevation in the middle of the plateau

**Visual Characteristics on X-ray:**
- Appears as a relatively flat or gently curved white/bright area
- Bounded by cortical bone margins (darker lines)
- Shows trabecular pattern within
- Clearly separated from femoral condyles above (joint space)
- Extends down to tibial shaft below

### 1.2 Critical Anatomical Fixed Points

**Point 1: Medial Edge of Tibial Plateau**
- **Definition:** The innermost (medial) boundary of the tibial plateau
- **Location:** Where medial tibial cortex meets the articular surface
- **Visual cue:** Inner edge of the tibial plateau (toward the spine on AP view)
- **Precision:** Identify the lateral border of the medial tibial spines/eminence
- **X-ray appearance:** Clear cortical line marking the medial boundary

**Point 2: Lateral Edge of Tibial Plateau**
- **Definition:** The outermost (lateral) boundary of the tibial plateau
- **Location:** Where lateral tibial cortex meets the articular surface
- **Visual cue:** Outer edge of the tibial plateau (away from spine on AP view)
- **Precision:** Identify just medial to the fibular head insertion area
- **X-ray appearance:** Clear cortical line marking the lateral boundary

**Point 3: Central Tibial Eminence**
- **Definition:** The midline peak/ridge of the tibial plateau
- **Location:** Between medial and lateral plateau, at the highest point
- **Visual cue:** Slight elevation or peaked area in the middle
- **Precision:** Identify the exact center point between medial and lateral edges
- **X-ray appearance:** Small peak or point at approximate midline

**Measurement Logic:**
```
Medial Edge (X₁) ─────────────────── Central Eminence (X₂) ─────────────────── Lateral Edge (X₃)

Distance = X₁ to X₃ (total plateau width)
Midpoint = (X₁ + X₃) / 2 = X₂ (should coincide with central eminence)

Division Line = Vertical line through X₂
  → Creates Medial Half: X₁ to X₂
  → Creates Lateral Half: X₂ to X₃
```

### 1.3 Joint Space Width (JSW) Measurement

**Medial JSW Definition:**
- The perpendicular distance between the femoral condyle (medial) and tibial plateau (medial half)
- Measured at the midpoint of the medial half
- Represents cartilage thickness on the medial side
- Unit: millimeters (mm)
- Normal range: 3.0-5.0 mm

**Lateral JSW Definition:**
- The perpendicular distance between the femoral condyle (lateral) and tibial plateau (lateral half)
- Measured at the midpoint of the lateral half
- Represents cartilage thickness on the lateral side
- Unit: millimeters (mm)
- Normal range: 3.0-5.0 mm

**Measurement Points (5 per half):**
```
For MEDIAL HALF:
  Point M1: 20% into medial half from medial edge
  Point M2: 40% into medial half
  Point M3: 60% into medial half (center of medial half)
  Point M4: 80% into medial half
  
For LATERAL HALF:
  Point L1: 20% into lateral half from center
  Point L2: 40% into lateral half
  Point L3: 60% into lateral half (center of lateral half)
  Point L4: 80% into lateral half
  Point L5: At lateral edge
```

---

## PART 2: IMAGE PROCESSING ALGORITHM

### 2.1 Step 1: Image Preprocessing

**Input:**
- Knee X-ray image (JPG, PNG, TIFF)
- Resolution: Minimum 512×512 pixels (preferably 1024×1024+)
- Format: Grayscale or RGB (will convert to grayscale)

**Processing:**

```python
# Step 1.1: Load and normalize image
image = load_image(image_path)
image_gray = convert_to_grayscale(image)

# Step 1.2: Equalize contrast (improves visibility)
clahe = CLAHE(clipLimit=2.0, tileGridSize=(8,8))
image_enhanced = clahe.apply(image_gray)

# Step 1.3: Denoise while preserving edges
image_denoised = bilateral_filter(image_enhanced, d=5, sigma_color=10, sigma_space=10)

# Step 1.4: Edge enhancement
edges = cv2.Canny(image_denoised, threshold1=50, threshold2=150)
edges_dilated = cv2.dilate(edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
```

**Output:** Enhanced grayscale image with clear edges, ready for bone detection

### 2.2 Step 2: Identify Knee Region (ROI - Region of Interest)

**Objective:** Focus on the knee joint area, exclude non-relevant parts

**Algorithm:**

```python
def identify_knee_region(image_enhanced):
    """
    Identify the knee joint region from X-ray
    Returns: bounding box of knee region
    """
    
    # Apply threshold to find bone regions (white areas)
    _, binary = cv2.threshold(image_enhanced, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours of bone structures
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour (should be femur + tibia + fibula)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Expand ROI slightly for safety
    padding = 50
    roi = {
        'x': max(0, x - padding),
        'y': max(0, y - padding),
        'width': w + 2*padding,
        'height': h + 2*padding
    }
    
    return roi
```

**Output:** Bounding box coordinates of knee region

### 2.3 Step 3: Detect Tibial Plateau

**Objective:** Isolate and identify the tibial plateau region

**Algorithm:**

```python
def detect_tibial_plateau(image_enhanced, roi):
    """
    Detect tibial plateau from knee X-ray
    Tibial plateau = upper flat surface of tibia, bright on X-ray
    
    Returns:
    - tibial_plateau_mask: Binary mask of tibial plateau
    - tibial_plateau_contour: Contour of tibial plateau
    """
    
    # Crop to ROI
    knee_region = image_enhanced[roi['y']:roi['y']+roi['height'], 
                                  roi['x']:roi['x']+roi['width']]
    
    # Adaptive thresholding to find bright bone areas
    thresh = cv2.adaptiveThreshold(knee_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=31, C=10)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify tibial plateau:
    # - Should be in lower half of knee region
    # - Should be relatively horizontal
    # - Should be below the femoral condyles
    
    tibial_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size (tibial plateau should be substantial)
        if area < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tibial plateau characteristics:
        # - Width >> Height (wide, flat)
        # - Aspect ratio > 2.0
        # - Located in lower half of ROI
        
        if w > 0 and h > 0:
            aspect_ratio = w / h
            vertical_position = (y + h/2) / knee_region.shape[0]  # Normalized, 0-1
            
            # Accept if it's flat (high aspect ratio) and in lower position
            if aspect_ratio > 2.0 and vertical_position > 0.4:
                tibial_candidates.append({
                    'contour': contour,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'score': aspect_ratio * area  # Prefer larger, flatter contours
                })
    
    # Select best candidate
    if tibial_candidates:
        tibial_plateau_contour = max(tibial_candidates, key=lambda x: x['score'])['contour']
    else:
        raise ValueError("Could not detect tibial plateau")
    
    # Create mask
    tibial_plateau_mask = np.zeros_like(knee_region)
    cv2.drawContours(tibial_plateau_mask, [tibial_plateau_contour], 0, 255, -1)
    
    return tibial_plateau_mask, tibial_plateau_contour
```

**Output:** 
- Binary mask of tibial plateau
- Contour coordinates of tibial plateau

### 2.4 Step 4: Identify Fixed Points (Medial Edge, Lateral Edge, Center)

**Objective:** Locate exact anatomical fixed points on tibial plateau

**Algorithm:**

```python
def identify_tibial_fixed_points(tibial_plateau_contour):
    """
    Identify three critical fixed points on tibial plateau:
    1. Medial Edge (innermost point)
    2. Lateral Edge (outermost point)
    3. Central Eminence (apex of center peak)
    
    Returns:
    - point_medial: (x, y) coordinates of medial edge
    - point_lateral: (x, y) coordinates of lateral edge
    - point_center: (x, y) coordinates of central eminence
    """
    
    # Get contour points
    contour_points = tibial_plateau_contour.reshape(-1, 2)
    
    # Find medial edge (minimum x)
    point_medial_idx = np.argmin(contour_points[:, 0])
    point_medial = contour_points[point_medial_idx]
    
    # Find lateral edge (maximum x)
    point_lateral_idx = np.argmax(contour_points[:, 0])
    point_lateral = contour_points[point_lateral_idx]
    
    # Find central eminence (peak in middle)
    # Eminence = point closest to the vertical midline with minimum y (highest point)
    
    x_min = point_medial[0]
    x_max = point_lateral[0]
    x_center = (x_min + x_max) / 2
    
    # Find points near center x (within 10% of total width)
    tolerance = (x_max - x_min) * 0.1
    center_region = contour_points[
        (contour_points[:, 0] >= x_center - tolerance) & 
        (contour_points[:, 0] <= x_center + tolerance)
    ]
    
    # Central eminence = highest point (minimum y) in center region
    if len(center_region) > 0:
        point_center_idx = np.argmin(center_region[:, 1])
        point_center = center_region[point_center_idx]
    else:
        # Fallback: use center of x-range
        point_center = np.array([x_center, np.min(contour_points[:, 1])])
    
    # Validate points are in correct order
    assert point_medial[0] < point_center[0] < point_lateral[0], \
        "Points not in correct order (medial < center < lateral)"
    
    return point_medial, point_center, point_lateral
```

**Output:**
- `point_medial`: (x, y) coordinates of medial edge
- `point_center`: (x, y) coordinates of central eminence
- `point_lateral`: (x, y) coordinates of lateral edge

**Precision Notes:**
- Fixed points are identified using contour analysis
- Medial edge = leftmost point (minimum x-coordinate)
- Lateral edge = rightmost point (maximum x-coordinate)
- Central eminence = highest point in center x-range

### 2.5 Step 5: Draw Division Line (Vertical through Center)

**Objective:** Draw a vertical line through the central eminence to divide medial and lateral halves

**Algorithm:**

```python
def draw_division_line(image, point_center):
    """
    Draw vertical division line through central eminence
    
    This line separates:
    - Medial Half: Left of line
    - Lateral Half: Right of line
    
    Returns:
    - image_with_line: Image with drawn division line
    - x_division: x-coordinate of division line
    """
    
    x_division = point_center[0]
    
    # Draw vertical line through entire image height
    y_top = 0
    y_bottom = image.shape[0] - 1
    
    # Draw line in red (for visualization)
    cv2.line(image, (int(x_division), int(y_top)), (int(x_division), int(y_bottom)),
             color=(0, 0, 255), thickness=2)
    
    return image, x_division
```

**Output:**
- Image with drawn division line
- x-coordinate of division line (used for measurements)

---

## PART 3: MEASUREMENT CALCULATIONS

### 3.1 Step 1: Detect Femoral Surfaces (For JSW Calculation)

**Objective:** Identify where femoral condyles meet the joint space

**Algorithm:**

```python
def detect_femoral_surfaces(image_enhanced, tibial_plateau_contour):
    """
    Detect femoral condyle surfaces above the joint space
    These are used to calculate the perpendicular distance (JSW)
    
    Returns:
    - femoral_surface_medial: Contour of medial femoral condyle surface
    - femoral_surface_lateral: Contour of lateral femoral condyle surface
    """
    
    # Find tibial plateau top edge (joint space boundary)
    tibial_top_y = np.min(tibial_plateau_contour[:, 1])
    
    # Search for femoral surfaces above tibial plateau
    # Femoral surfaces = bright white bone areas above joint space
    
    search_region_top = max(0, tibial_top_y - 150)
    search_region = image_enhanced[search_region_top:int(tibial_top_y), :]
    
    # Apply threshold to find femoral bone
    _, binary_femoral = cv2.threshold(search_region, 80, 255, cv2.THRESH_BINARY)
    
    # Find contours of femoral surfaces
    contours, _ = cv2.findContours(binary_femoral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Separate into medial and lateral based on x-coordinate
    x_center = (np.min(tibial_plateau_contour[:, 0]) + np.max(tibial_plateau_contour[:, 0])) / 2
    
    medial_femoral = None
    lateral_femoral = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # Too small to be femoral surface
            continue
        
        contour_center_x = np.mean(contour[:, 0, 0])
        
        if contour_center_x < x_center:
            if medial_femoral is None or cv2.contourArea(contour) > cv2.contourArea(medial_femoral):
                medial_femoral = contour
        else:
            if lateral_femoral is None or cv2.contourArea(contour) > cv2.contourArea(lateral_femoral):
                lateral_femoral = contour
    
    return medial_femoral, lateral_femoral
```

### 3.2 Step 2: Calculate JSW at Multiple Points

**Objective:** Measure perpendicular distances at 4 points in each half

**Algorithm:**

```python
def calculate_jsw_measurements(
    tibial_plateau_contour,
    femoral_surface_medial,
    femoral_surface_lateral,
    x_division,
    points_per_half=4
):
    """
    Calculate Joint Space Width (JSW) in medial and lateral halves
    
    JSW = perpendicular distance from tibial plateau to femoral condyle
    
    Returns:
    - jsw_medial: Dictionary with measurements for medial half
    - jsw_lateral: Dictionary with measurements for lateral half
    """
    
    # Get tibial plateau points in medial half
    medial_plateau_points = tibial_plateau_contour[tibial_plateau_contour[:, 0] <= x_division]
    
    # Get tibial plateau points in lateral half
    lateral_plateau_points = tibial_plateau_contour[tibial_plateau_contour[:, 0] >= x_division]
    
    # ========== MEDIAL JSW ==========
    
    if len(medial_plateau_points) > 0 and femoral_surface_medial is not None:
        # Create sampling points along medial plateau
        medial_distances = []
        
        # Sample 4 points across medial half
        for i in range(points_per_half):
            # Position: 20%, 40%, 60%, 80% along medial half
            position = 0.2 + i * 0.2
            idx = int(len(medial_plateau_points) * position)
            
            if idx >= len(medial_plateau_points):
                idx = len(medial_plateau_points) - 1
            
            tibial_point = medial_plateau_points[idx][0]
            
            # Find closest point on femoral surface at same x-coordinate
            femoral_candidates = femoral_surface_medial[
                np.abs(femoral_surface_medial[:, 0, 0] - tibial_point[0]) < 20
            ]
            
            if len(femoral_candidates) > 0:
                # Find highest point (minimum y) = femoral surface
                femoral_point = femoral_candidates[np.argmin(femoral_candidates[:, 0, 1])][0]
                
                # Calculate perpendicular distance (vertical distance)
                distance_mm = (tibial_point[1] - femoral_point[1]) * PIXEL_TO_MM_RATIO
                medial_distances.append(distance_mm)
        
        jsw_medial = {
            'measurements_mm': medial_distances,
            'min_mm': min(medial_distances) if medial_distances else None,
            'max_mm': max(medial_distances) if medial_distances else None,
            'mean_mm': np.mean(medial_distances) if medial_distances else None,
            'std_mm': np.std(medial_distances) if medial_distances else None,
            'count': len(medial_distances)
        }
    
    # ========== LATERAL JSW ==========
    
    if len(lateral_plateau_points) > 0 and femoral_surface_lateral is not None:
        lateral_distances = []
        
        # Sample 4 points across lateral half
        for i in range(points_per_half):
            position = 0.2 + i * 0.2
            idx = int(len(lateral_plateau_points) * position)
            
            if idx >= len(lateral_plateau_points):
                idx = len(lateral_plateau_points) - 1
            
            tibial_point = lateral_plateau_points[idx][0]
            
            # Find closest point on femoral surface at same x-coordinate
            femoral_candidates = femoral_surface_lateral[
                np.abs(femoral_surface_lateral[:, 0, 0] - tibial_point[0]) < 20
            ]
            
            if len(femoral_candidates) > 0:
                femoral_point = femoral_candidates[np.argmin(femoral_candidates[:, 0, 1])][0]
                
                distance_mm = (tibial_point[1] - femoral_point[1]) * PIXEL_TO_MM_RATIO
                lateral_distances.append(distance_mm)
        
        jsw_lateral = {
            'measurements_mm': lateral_distances,
            'min_mm': min(lateral_distances) if lateral_distances else None,
            'max_mm': max(lateral_distances) if lateral_distances else None,
            'mean_mm': np.mean(lateral_distances) if lateral_distances else None,
            'std_mm': np.std(lateral_distances) if lateral_distances else None,
            'count': len(lateral_distances)
        }
    
    return jsw_medial, jsw_lateral
```

### 3.3 Step 3: Validate Measurements

**Objective:** Ensure measurements are physiologically plausible

**Algorithm:**

```python
def validate_measurements(jsw_medial, jsw_lateral):
    """
    Validate that measurements fall within physiologically plausible ranges
    
    Normal JSW: 2.5-7.5 mm (typically 3.5-5.5 mm)
    
    Returns:
    - is_valid: Boolean indicating if measurements are valid
    - confidence_score: 0-100% confidence in measurements
    - warnings: List of issues found
    """
    
    warnings = []
    confidence_score = 100
    
    # Check medial JSW
    if jsw_medial['mean_mm'] is not None:
        if jsw_medial['mean_mm'] < 0.5:
            warnings.append("Medial JSW unusually small (<0.5mm) - possible measurement error")
            confidence_score -= 30
        elif jsw_medial['mean_mm'] > 10:
            warnings.append("Medial JSW unusually large (>10mm) - possible measurement error")
            confidence_score -= 30
        elif jsw_medial['mean_mm'] < 2.0:
            warnings.append("Medial JSW below normal range (Grade 3-4 OA)")
        
        # Check for high variability
        if jsw_medial['std_mm'] / jsw_medial['mean_mm'] > 0.5:
            warnings.append("High variability in medial JSW measurements")
            confidence_score -= 10
    
    # Check lateral JSW
    if jsw_lateral['mean_mm'] is not None:
        if jsw_lateral['mean_mm'] < 0.5:
            warnings.append("Lateral JSW unusually small (<0.5mm) - possible measurement error")
            confidence_score -= 30
        elif jsw_lateral['mean_mm'] > 10:
            warnings.append("Lateral JSW unusually large (>10mm) - possible measurement error")
            confidence_score -= 30
        elif jsw_lateral['mean_mm'] < 2.0:
            warnings.append("Lateral JSW below normal range (Grade 3-4 OA)")
        
        # Check for high variability
        if jsw_lateral['std_mm'] / jsw_lateral['mean_mm'] > 0.5:
            warnings.append("High variability in lateral JSW measurements")
            confidence_score -= 10
    
    # Check if measurements are too different
    if jsw_medial['mean_mm'] and jsw_lateral['mean_mm']:
        diff = abs(jsw_medial['mean_mm'] - jsw_lateral['mean_mm'])
        if diff > 3.0:
            warnings.append(f"Large difference between medial ({jsw_medial['mean_mm']:.1f}) and lateral ({jsw_lateral['mean_mm']:.1f}) JSW")
    
    is_valid = confidence_score > 50
    
    return is_valid, max(0, confidence_score), warnings
```

---

## PART 4: OUTPUT FORMAT

### 4.1 Measurement Dictionary

**Return Structure:**

```python
measurement_result = {
    "filename": "patient_xray.jpg",
    "timestamp": "2026-04-21T12:34:56",
    "detection_status": "SUCCESS",
    
    # Fixed points detected
    "fixed_points": {
        "medial_edge_pixels": [145, 380],     # (x, y) coordinates
        "lateral_edge_pixels": [420, 385],
        "central_eminence_pixels": [282, 375],
        
        "medial_edge_mm": None,  # Will add if calibration available
        "lateral_edge_mm": None,
        "central_eminence_mm": None
    },
    
    # Division line
    "division_line": {
        "x_coordinate_pixels": 282,
        "type": "vertical_through_central_eminence"
    },
    
    # Medial half measurements
    "medial_jsw": {
        "measurements_mm": [3.2, 3.5, 3.4, 3.1],  # 4 measurement points
        "min_mm": 3.1,
        "max_mm": 3.5,
        "mean_mm": 3.3,
        "std_mm": 0.18,
        "measurement_points": 4,
        "clinical_assessment": "Normal - Grade 0-1"
    },
    
    # Lateral half measurements
    "lateral_jsw": {
        "measurements_mm": [3.6, 3.8, 3.7, 3.5],
        "min_mm": 3.5,
        "max_mm": 3.8,
        "mean_mm": 3.65,
        "std_mm": 0.13,
        "measurement_points": 4,
        "clinical_assessment": "Normal - Grade 0-1"
    },
    
    # Overall assessment
    "overall_assessment": {
        "average_jsw_mm": 3.48,
        "estimated_grade": 1,  # 0-4 scale
        "asymmetry": {
            "medial_vs_lateral_difference_mm": 0.35,
            "interpretation": "Slightly more lateral narrowing"
        },
        "measurement_confidence": 85  # 0-100%
    },
    
    # Quality metrics
    "quality_metrics": {
        "image_sharpness_score": 78.5,  # 0-100
        "detection_confidence": 87.2,   # 0-100
        "measurement_precision": "±0.3mm",
        "warnings": []
    },
    
    # Visual output
    "visual_outputs": {
        "image_with_marked_points": "base64_encoded_image",
        "image_with_division_line": "base64_encoded_image",
        "measurement_diagram": "base64_encoded_image"
    }
}
```

### 4.2 Visual Overlays to Generate

**Overlay 1: Fixed Points**
```
- Draw medial edge: Red circle (radius 10px)
- Draw lateral edge: Blue circle (radius 10px)
- Draw central eminence: Green circle (radius 12px, filled)
- Draw lines connecting points
- Add text labels: "Medial", "Center", "Lateral"
```

**Overlay 2: Division Line**
```
- Draw vertical red line through central eminence
- Extend from tibial plateau top to bottom
- Thickness: 2-3 pixels
- Add label: "Medial | Lateral"
```

**Overlay 3: Measurement Points**
```
- Mark 4 measurement points in medial half: Yellow dots
- Mark 4 measurement points in lateral half: Orange dots
- Draw perpendicular lines from each point to femoral surface
- Label with distance values
```

**Overlay 4: Summary Diagram**
```
- Show both halves with measurements
- Medial side: Show min/max/mean JSW
- Lateral side: Show min/max/mean JSW
- Color coding: Green (normal), Yellow (mild), Red (severe)
```

---

## PART 5: CALIBRATION (PIXEL TO MILLIMETER)

**Critical Step:** Convert pixel measurements to clinical millimeters

**Method 1: Using Ruler in Image (Automatic)**
```python
def detect_ruler_calibration(image):
    """
    If image contains a ruler/scale marker (visible on sides of X-ray):
    - Detect ruler marks
    - Count pixels per mark
    - Calculate pixel-to-mm ratio
    
    Returns:
    - pixels_per_mm: Float value
    """
    
    # Detect ruler markings (regular dark lines on sides)
    # Measure distance between marks in pixels
    # Known: Standard medical X-ray rulers have 1cm marks
    
    # Example: If ruler shows 10 marks over 100 pixels
    # = 10 cm = 100 mm in 100 pixels
    # = 1 mm per pixel
    
    pixels_per_mm = measure_ruler_spacing(image)
    return pixels_per_mm
```

**Method 2: Using Anatomical Reference (If No Ruler)**
```python
def estimate_calibration_from_anatomy(tibial_plateau_contour):
    """
    If no ruler visible, use anatomical reference:
    - Tibial plateau width in healthy adults: 70-80 mm
    
    Returns:
    - estimated_pixels_per_mm: Float value (may have ±10% error)
    """
    
    # Measure detected plateau width in pixels
    plateau_width_pixels = np.max(tibial_plateau_contour[:, 0]) - np.min(tibial_plateau_contour[:, 0])
    
    # Assume normal plateau: 75 mm
    assumed_plateau_mm = 75
    
    pixels_per_mm = plateau_width_pixels / assumed_plateau_mm
    confidence = 0.7  # 70% confidence (±10% error)
    
    return pixels_per_mm, confidence
```

---

## PART 6: ERROR HANDLING & EXCEPTIONS

### 6.1 Common Detection Failures

**Failure 1: Cannot Detect Tibial Plateau**
```
Symptom: No clear flat bone surface found
Causes:
  - Image quality too poor (too blurry)
  - Wrong view (not AP view)
  - Unusual anatomy
  
Recovery:
  - Return error message with image quality score
  - Ask radiologist to confirm if image is valid AP view
  - Suggest retaking X-ray if too blurry
```

**Failure 2: Cannot Identify Fixed Points**
```
Symptom: Medial/lateral edges unclear
Causes:
  - Distorted anatomy (fracture, surgical changes)
  - Image artifacts
  
Recovery:
  - Lower confidence score proportionally
  - Return partial results with available measurements
  - Flag for manual review
```

**Failure 3: Measurements Out of Range**
```
Symptom: JSW <0.5mm or >10mm
Causes:
  - Calibration error
  - Measurement point on cortex instead of joint space
  
Recovery:
  - Flag measurement as questionable
  - Reduce confidence
  - Suggest manual measurement verification
```

### 6.2 Confidence Scoring Algorithm

```python
def calculate_confidence(
    detection_quality,
    measurement_variance,
    value_in_range,
    image_sharpness
):
    """
    Calculate overall confidence (0-100%)
    
    Factors:
    1. Image sharpness (30% weight)
    2. Detection quality (30% weight)
    3. Measurement consistency (20% weight)
    4. Physiological plausibility (20% weight)
    """
    
    score = 0
    
    # Factor 1: Sharpness
    sharpness_contribution = image_sharpness * 0.30
    
    # Factor 2: Detection quality
    if detection_quality > 0.9:
        detection_contribution = 30
    elif detection_quality > 0.7:
        detection_contribution = 25
    else:
        detection_contribution = 15
    
    # Factor 3: Measurement variance
    if measurement_variance < 0.3:
        variance_contribution = 20
    elif measurement_variance < 0.5:
        variance_contribution = 15
    else:
        variance_contribution = 8
    
    # Factor 4: Plausibility
    if value_in_range:
        plausibility_contribution = 20
    else:
        plausibility_contribution = 5
    
    total_confidence = sharpness_contribution + detection_contribution + variance_contribution + plausibility_contribution
    
    return max(0, min(100, total_confidence))
```

---

## PART 7: EXAMPLE OUTPUTS

### Example 1: Successful Detection
```
IMAGE: patient_001_AP.jpg (80-year-old female, Grade 1 OA)

FIXED POINTS DETECTED:
  ✓ Medial edge: x=142px, y=378px
  ✓ Lateral edge: x=423px, y=385px
  ✓ Central eminence: x=282px, y=373px
  
TIBIAL PLATEAU WIDTH: 281 pixels = 87.8mm (estimated)
CALIBRATION: 1 pixel = 0.312mm (from ruler)

DIVISION LINE: x=282px (vertical through central eminence)

MEDIAL JSW (LEFT OF DIVISION LINE):
  Point 1 (20%): 3.2mm ✓
  Point 2 (40%): 3.5mm ✓
  Point 3 (60%): 3.4mm ✓
  Point 4 (80%): 3.1mm ✓
  
  Medial Average: 3.3 ± 0.18mm
  Clinical Grade: Grade 0-1 (Normal)

LATERAL JSW (RIGHT OF DIVISION LINE):
  Point 1 (20%): 3.6mm ✓
  Point 2 (40%): 3.8mm ✓
  Point 3 (60%): 3.7mm ✓
  Point 4 (80%): 3.5mm ✓
  
  Lateral Average: 3.65 ± 0.13mm
  Clinical Grade: Grade 0-1 (Normal)

OVERALL ASSESSMENT:
  Average JSW: 3.48mm
  Estimated KL Grade: 1 (Doubtful OA)
  Asymmetry: 0.35mm (slightly more lateral narrowing)
  
QUALITY METRICS:
  Image Sharpness: 78.5/100 (GOOD)
  Detection Confidence: 87.2% (HIGH)
  Measurement Confidence: 85% (HIGH)
  
STATUS: ✓ PASS - All measurements valid
```

### Example 2: Partial Failure
```
IMAGE: patient_002_AP_ROTATED.jpg (56-year-old male)

⚠️  WARNINGS:
  - Image appears slightly rotated (±5°)
  - Lateral plateau edge unclear (surgical changes?)
  - High measurement variance in lateral half

FIXED POINTS DETECTED:
  ✓ Medial edge: x=165px (HIGH CONFIDENCE)
  ⚠ Lateral edge: x=398px (MEDIUM CONFIDENCE - unclear cortex)
  ✓ Central eminence: x=281px

MEDIAL JSW: 2.8 ± 0.22mm (VALID)
LATERAL JSW: 3.2 ± 0.45mm (QUESTIONABLE - high variance)

QUALITY METRICS:
  Image Sharpness: 65.2/100 (ACCEPTABLE)
  Detection Confidence: 72.3% (MEDIUM)
  Measurement Confidence: 68% (MEDIUM)
  
RECOMMENDATIONS:
  1. Confirm if patient has prior knee surgery (lateral changes)
  2. Retake X-ray with proper alignment if possible
  3. Manual verification recommended for lateral measurements
  
STATUS: ⚠️ PARTIAL - Use with caution
```

---

## PART 8: COMPLETE FUNCTION SIGNATURE

**Main Entry Function:**

```python
def analyze_knee_xray_tibial_plateau(
    image_path: str,
    enable_visualization: bool = True,
    auto_calibration: bool = True,
    measurement_points_per_half: int = 4,
    confidence_threshold: float = 0.6
) -> dict:
    """
    Automatically analyze knee X-ray image:
    - Detect tibial plateau
    - Identify medial/lateral fixed points
    - Draw division line through central eminence
    - Measure Joint Space Width in each half
    - Return structured measurements and visualizations
    
    Parameters:
    -----------
    image_path : str
        Path to knee X-ray image (JPG, PNG, TIFF)
    
    enable_visualization : bool
        If True, generate visual overlays with marked points and measurements
    
    auto_calibration : bool
        If True, attempt to detect ruler/scale for pixel-to-mm conversion
        If False, use anatomical reference (tibial plateau width)
    
    measurement_points_per_half : int
        Number of JSW measurement points per half (default: 4, max: 8)
    
    confidence_threshold : float
        Minimum confidence (0-1) required to return measurements
        If below threshold, return error with reason
    
    Returns:
    --------
    dict : Structured result containing:
        - fixed_points: Medial edge, lateral edge, central eminence coordinates
        - division_line: x-coordinate of medial/lateral division
        - medial_jsw: Measurements for medial half
        - lateral_jsw: Measurements for lateral half
        - overall_assessment: Grade, asymmetry, clinical interpretation
        - quality_metrics: Sharpness, confidence, warnings
        - visual_outputs: Base64-encoded images with overlays
    
    Example:
    --------
    result = analyze_knee_xray_tibial_plateau(
        image_path='patient_xray.jpg',
        enable_visualization=True,
        auto_calibration=True
    )
    
    print(f"Medial JSW: {result['medial_jsw']['mean_mm']:.2f}mm")
    print(f"Lateral JSW: {result['lateral_jsw']['mean_mm']:.2f}mm")
    print(f"Confidence: {result['overall_assessment']['measurement_confidence']:.1f}%")
    """
    
    # Implementation follows all steps outlined above
    pass
```

---

## SUMMARY: KEY SPECIFICATIONS FOR CLAUDE

### What Claude Must Do:

1. ✅ **Detect Tibial Plateau**
   - Input: Knee X-ray image
   - Output: Contour coordinates of tibial plateau
   - Method: Edge detection + contour analysis

2. ✅ **Identify 3 Fixed Points**
   - Medial edge (leftmost point)
   - Lateral edge (rightmost point)
   - Central eminence (highest point in center)
   - Method: Contour analysis + spatial geometry

3. ✅ **Draw Division Line**
   - Type: Vertical line through central eminence
   - Purpose: Separate medial and lateral halves
   - Output: Visual overlay on image + x-coordinate value

4. ✅ **Measure JSW in Each Half**
   - 4 measurement points per half (20%, 40%, 60%, 80%)
   - Calculate perpendicular distances (femur to tibia)
   - Report: min, max, mean ± std deviation (in mm)

5. ✅ **Generate Outputs**
   - Structured JSON with all measurements
   - Visual overlays with marked points and lines
   - Clinical assessment and confidence score
   - Warnings for potential errors

### Input Validation:

- ✅ Image format: JPG, PNG, TIFF (any resolution ≥512×512)
- ✅ View: Anteroposterior (AP) knee X-ray
- ✅ Content: Clear tibial plateau visible
- ✅ Quality: Minimum sharpness score 30/100

### Output Guarantee:

- ✅ Always returns structured JSON
- ✅ Always includes confidence score
- ✅ Always flags edge cases/warnings
- ✅ Always provides visual overlays if enabled
- ✅ Always quantifies measurement uncertainty

---

**This prompt provides Claude with complete specifications to build an accurate, robust tibial plateau analysis system.**
