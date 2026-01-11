from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import csv
import logging
import shutil
import os
import sys

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("server_debug.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. ADVANCED CV ALGORITHMS ---
class HandwritingFeatures:
    def __init__(self, image_path):
        logging.info(f"--- [START] Processing: {image_path} ---")
        self.image_path = image_path
        
        # 1. Load & Normalize
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        
        # Resize to fixed width (1200px) for consistent measurement scale
        h, w = img.shape[:2]
        scale = 1200 / w
        self.original = cv2.resize(img, (1200, int(h * scale)))
        
        # 2. Binarization (Otsu's Method is better for global structure)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # 3. Skeletonization (Thinning) - Critical for Stroke Counting
        # Reduces ink to single-pixel wide lines
        # Check if ximgproc is available (opencv-contrib-python), else fallback
        try:
            self.skeleton = cv2.ximgproc.thinning(self.binary)
        except AttributeError:
            self.skeleton = self._fallback_skeleton(self.binary)
        
        # 4. Calculate Global Baselines
        self.avg_line_height = self._get_avg_line_height()
        logging.info(f"   > Avg Line Height (Px): {self.avg_line_height}")

    def _fallback_skeleton(self, img):
        """Standard thinning algorithm if ximgproc is missing."""
        skeleton = np.zeros(img.shape, np.uint8)
        eroded = img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(eroded, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            eroded = cv2.erode(eroded, kernel)
            if cv2.countNonZero(eroded) == 0: break
        return skeleton

    def _get_avg_line_height(self):
        """Uses Horizontal Projection Profile to find text lines accurately."""
        # Sum pixels along rows (Horizontal Projection)
        proj = np.sum(self.binary, axis=1)
        
        # Find continuous runs of non-zero pixels (Text Lines)
        lines = []
        in_line = False
        start = 0
        for y, val in enumerate(proj):
            if val > 0 and not in_line:
                in_line = True
                start = y
            elif val == 0 and in_line:
                in_line = False
                height = y - start
                if height > 10: # Min line height threshold
                    lines.append(height)
        
        return np.median(lines) if lines else 50.0

    def _clamp(self, val):
        return max(0.0, min(1.0, float(val)))

    # --- FEATURE 1: TREMORS (Smoothness Ratio) ---
    def get_tremors(self):
        """Compares rough contour perimeter vs smooth hull perimeter."""
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        scores = []
        for c in contours:
            if cv2.contourArea(c) > 100:
                # 1. Real Perimeter (Jagged)
                real_perim = cv2.arcLength(c, True)
                
                # 2. Smooth Perimeter (Convex Hull)
                hull = cv2.convexHull(c)
                smooth_perim = cv2.arcLength(hull, True)
                
                if smooth_perim > 0:
                    # Ratio: 1.0 = Perfect Smoothness (Circle/Oval)
                    # Higher = More Jitter/Tremor
                    roughness = (real_perim / smooth_perim) - 1.0
                    scores.append(roughness)
        
        if not scores: return 0.0
        
        avg_roughness = np.mean(scores)
        # Normalize: 0.0 (Smooth) to 0.2 (Very Shaky) scaled to 0-1
        # Multiplier adjusted for sensitivity
        return self._clamp(avg_roughness * 5.0)

    # --- FEATURE 2: PEN LIFTS (Skeleton Discontinuities) ---
    def get_pen_lifts_score(self):
        """Counts connected components in the SKELETON, not the thick ink."""
        # 1. Count separate strokes in skeleton
        num_strokes, _ = cv2.connectedComponents(self.skeleton)
        
        # 2. Count distinct words (using dilation to merge letters)
        kernel = np.ones((5, 15), np.uint8) # Wide kernel merges letters
        dilated = cv2.dilate(self.binary, kernel, iterations=2)
        num_words, _ = cv2.connectedComponents(dilated)
        
        if num_words <= 1: return 0.5 # Default fallback
        
        # Ratio: Strokes / Words
        # Cursive: ~1.2 strokes/word | Print: ~3-4 strokes/word
        ratio = num_strokes / num_words
        
        # Normalize: Map ratio 1.0->4.0 to 0.0->1.0
        norm = (ratio - 1.0) / 3.0
        return self._clamp(norm)

    # --- FEATURE 3: WORD SPACING (Projection Distance) ---
    def get_word_spacing_score(self):
        """Measures distance between word blobs on the same line."""
        # 1. Merge letters into words
        kernel = np.ones((3, 12), np.uint8)
        dilated = cv2.dilate(self.binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 2. Collect bounding boxes
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]
        boxes.sort(key=lambda b: (b[1] // (self.avg_line_height/2), b[0])) # Sort by Line then X
        
        spacings = []
        for i in range(len(boxes) - 1):
            curr = boxes[i]
            next_b = boxes[i+1]
            
            # Check if on same line (Y difference is small)
            y_diff = abs((curr[1] + curr[3]/2) - (next_b[1] + next_b[3]/2))
            if y_diff < self.avg_line_height * 0.5:
                # Calculate horizontal gap
                dist = next_b[0] - (curr[0] + curr[2])
                if 0 < dist < self.avg_line_height * 3: # valid gap
                    spacings.append(dist)
        
        if not spacings: return 0.5
        
        avg_spacing_px = np.median(spacings)
        # Normalize relative to line height
        # 0.2x LineHeight (Normal) to 1.0x LineHeight (Wide)
        norm = avg_spacing_px / self.avg_line_height
        return self._clamp(norm)

    # --- FEATURE 4: EMBELLISHMENTS (Skeleton Branching) ---
    def get_embellishment_score(self):
        """Detects complexity by checking branch points in the skeleton."""
        # Convolution to find pixels with >2 neighbors (intersections/crossings)
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        filtered = cv2.filter2D(self.skeleton, -1, kernel)
        
        # Pixels with value > 12 are branch points (center=10 + >2 neighbors)
        branches = np.sum(filtered > 12)
        total_ink = np.sum(self.skeleton > 0)
        
        if total_ink == 0: return 0.0
        
        # Ratio of branch points to total stroke length
        complexity = branches / total_ink
        # Normalize: 0.02 (Simple) to 0.10 (Highly Decorative)
        return self._clamp((complexity - 0.02) * 15.0)

    # --- FEATURE 5: RETOUCHING (Blob Density) ---
    def get_retouching_score(self):
        """Detects high-density ink blobs (overwriting)."""
        # Distance Transform: Finds thickest parts of the ink
        dist = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)
        # Thickness > 20% of line height usually means blob/overwrite
        threshold = self.avg_line_height * 0.20
        
        blobs = np.sum(dist > threshold)
        total_ink = np.sum(self.binary > 0)
        
        if total_ink == 0: return 0.0
        
        ratio = blobs / total_ink
        return self._clamp(ratio * 5.0) # Scaling factor

    # --- METRIC WRAPPERS (Counts & Stats) ---
    def get_counts_and_stats(self):
        # Loops
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        loop_areas = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1: # Inner contour
                    area = cv2.contourArea(contours[i])
                    if 10 < area < (self.avg_line_height**2):
                        loop_areas.append(area)
        
        loop_count = len(loop_areas)
        loop_mean = self._clamp(np.mean(loop_areas) / (self.avg_line_height**2) * 3) if loop_areas else 0
        loop_var = self._clamp(np.std(loop_areas) / (self.avg_line_height**2) * 5) if loop_areas else 0

        # T-Bars (Horizontal Line Detection)
        cols = self.binary.shape[1]
        horizontal_size = int(cols / 30)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        h_mask = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, h_kernel)
        t_contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_count = len(t_contours)

        # Diacritics (Small Dots)
        d_contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dot_count = 0
        dot_heights = []
        for c in d_contours:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            # Small compact shape high up
            if 5 < area < 100 and w < 20 and h < 20:
                dot_count += 1
                dot_heights.append(y)
        
        dia_offset_mean = 0.5 # Default middle
        if dot_heights:
             # Normalize Y position relative to image height (Rough approx)
             dia_offset_mean = self._clamp(1.0 - (np.mean(dot_heights) / self.binary.shape[0]))

        return {
            "Loop_Count": loop_count, "Loop_Mean": loop_mean, "Loop_Var": loop_var,
            "T_Count": t_count, "T_Mean": 0.5, "T_Var": 0.1, # Placeholders for T stats
            "Dia_Count": dot_count, "Dia_Mean": dia_offset_mean, "Dia_Var": 0.1
        }

    def generate_report(self):
        counts = self.get_counts_and_stats()
        
        return {
            # --- NORMALIZED INDICES (0.0 - 1.0) ---
            "Word_Spacing_Score": self.get_word_spacing_score(),
            "Pen_Lifts_Score": self.get_pen_lifts_score(),
            "Tremor_Index": self.get_tremors(),
            "Embellishment_Index": self.get_embellishment_score(),
            "Retouching_Index": self.get_retouching_score(),
            
            # --- COUNTS ---
            "Loops_Count": counts["Loop_Count"],
            "Diacritics_Count": counts["Dia_Count"],
            "T_Crossing_Count": counts["T_Count"],
            
            # --- ADVANCED VECTORS (For Prediction) ---
            "Loop_Area_Mean": counts["Loop_Mean"],
            "Loop_Area_Var": counts["Loop_Var"],
            "Diacritic_Offset_Mean": counts["Dia_Mean"],
            "Diacritic_Offset_Var": counts["Dia_Var"],
            "T_Bar_Height_Mean": counts["T_Mean"],
            "T_Bar_Height_Var": counts["T_Var"],
            "Word_Spacing_Norm": self.get_word_spacing_score()
        }

# --- 4. API ENDPOINT ---
@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    print("\n" + "="*60)
    logging.info(f"NEW REQUEST: {file.filename}")
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        extractor = HandwritingFeatures(temp_filename)
        results = extractor.generate_report()
        
        logging.info("Features Extracted Successfully.")
        
        # Save CSV
        csv_file = "analysis_results_full.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['Filename'] + list(results.keys()))
            writer.writerow([file.filename] + list(results.values()))
            
        return {"filename": file.filename, "data": results}

    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")