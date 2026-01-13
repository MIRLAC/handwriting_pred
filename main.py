from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import csv
import logging
import shutil
import os
import sys
import math

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

# --- 2. ROBUST CV ALGORITHMS ---
class HandwritingFeatures:
    def __init__(self, image_path):
        logging.info(f"--- [START] Processing: {image_path} ---")
        self.image_path = image_path
        
        # 1. Load & Resize
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        
        # Resize to fixed width (1000px)
        h, w = img.shape[:2]
        scale = 1000 / w
        self.original = cv2.resize(img, (1000, int(h * scale)))
        self.height, self.width = self.original.shape[:2]
        
        # 2. Binarization
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 19, 10
        )
        
        # 3. Baselines
        self.avg_stroke_width = self._get_stroke_width()
        self.avg_line_height = self._get_avg_line_height()
        
        # 4. Skeletonization
        try:
            self.skeleton = cv2.ximgproc.thinning(self.binary)
        except AttributeError:
            self.skeleton = self._fallback_skeleton(self.binary)

    def _get_stroke_width(self):
        dist = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)
        vals = dist[dist > 0]
        return float(np.mean(vals) * 2.0) if len(vals) > 0 else 2.0

    def _fallback_skeleton(self, img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        img_temp = img.copy()
        while not done:
            eroded = cv2.erode(img_temp, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img_temp, temp)
            skel = cv2.bitwise_or(skel, temp)
            img_temp = eroded.copy()
            if cv2.countNonZero(img_temp) == 0: done = True
        return skel

    def _get_avg_line_height(self):
        proj = np.sum(self.binary, axis=1)
        runs = []
        current_run = 0
        for val in proj:
            if val > 10: current_run += 1
            elif current_run > 0:
                if current_run > 15: runs.append(current_run)
                current_run = 0
        return float(np.median(runs)) if runs else 50.0

    def _force_visible_norm(self, val, multiplier=1.0):
        """Guarantees a score between 0.15 and 0.95."""
        if val < 0: val = 0
        score = 0.15 + (float(val) * multiplier)
        final_score = min(0.95, score)
        return float(round(final_score, 2))

    # --- FEATURE: WORD SPACING ---
    def get_word_spacing(self):
        fusion_kernel = np.ones((1, int(self.avg_stroke_width * 5)), np.uint8)
        dilated = cv2.dilate(self.binary, fusion_kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
        boxes.sort(key=lambda b: (b[1] // int(self.avg_line_height), b[0]))
        
        all_gaps = []
        for i in range(len(boxes) - 1):
            curr, next_b = boxes[i], boxes[i+1]
            cy1 = curr[1] + curr[3]/2
            cy2 = next_b[1] + next_b[3]/2
            
            if abs(cy1 - cy2) < self.avg_line_height * 0.5:
                gap = next_b[0] - (curr[0] + curr[2])
                if gap > 0 and gap < self.width * 0.3:
                    all_gaps.append(gap)
                    
        if not all_gaps: return 25, 0.25 

        median_gap = np.median(all_gaps)
        word_gaps = [g for g in all_gaps if g > median_gap * 1.5]
        if not word_gaps: word_gaps = [g for g in all_gaps if g > median_gap]
        
        avg_px = np.mean(word_gaps) if word_gaps else median_gap
        avg_px = max(15.0, float(avg_px)) 
        
        norm_score = self._force_visible_norm(avg_px / 100.0, multiplier=1.0)
        return int(avg_px), norm_score

    # --- FEATURE: LOOPS (SPECIFIC TO l, e, g, y, f, h) ---
    def get_loop_features(self):
        # 1. Get Contours and Hierarchy
        # Hierarchy: [Next, Prev, First_Child, Parent]
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None: return 0, 0.2, 0.2

        valid_loops = []
        
        # 2. Iterate through contours to find "Holes" (Parent != -1)
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1: # It is a child (hole)
                
                # Get Loop Metrics
                loop_cnt = contours[i]
                area = cv2.contourArea(loop_cnt)
                x, y, w, h_box = cv2.boundingRect(loop_cnt)
                aspect_ratio = h_box / float(w)
                
                # Get Parent Metrics (The letter containing the loop)
                parent_cnt = contours[h[3]]
                px, py, pw, ph = cv2.boundingRect(parent_cnt)
                
                # 3. ZONAL ANALYSIS (Ascender, Middle, Descender)
                # Calculate relative position of loop within the letter
                # Loop Center Y
                loop_cy = y + (h_box / 2)
                # Parent Center Y
                parent_cy = py + (ph / 2)
                
                # Filter small noise
                if area < 15: continue

                # -- CLASSIFICATION LOGIC based on Table --
                
                # TYPE A: 'e' (Small, Middle Zone)
                # Loop is roughly centered in parent, parent is small-ish
                if ph < self.avg_line_height * 0.8: 
                    # Likely 'e', 'a', 'o'
                    valid_loops.append(area)
                    
                # TYPE B: 'l', 'f', 'h' (Tall Ascenders)
                # Loop is in the Upper half of a tall letter
                elif (loop_cy < parent_cy) and (ph > self.avg_line_height * 0.9):
                    # Aspect ratio check: these loops are usually tall/thin
                    if aspect_ratio > 1.2:
                        valid_loops.append(area)
                        
                # TYPE C: 'g', 'y' (Descenders)
                # Loop is in the Lower half of a tall letter
                elif (loop_cy > parent_cy) and (ph > self.avg_line_height * 0.9):
                    valid_loops.append(area)

        # Stats
        count = len(valid_loops)
        if count == 0: return 0, 0.2, 0.2
        
        mean_area = np.mean(valid_loops)
        var_area = np.var(valid_loops)
        
        norm_mean = self._force_visible_norm(mean_area / (self.avg_line_height**2), multiplier=5.0)
        norm_var = self._force_visible_norm(math.sqrt(var_area) / (self.avg_line_height**2), multiplier=5.0)
        
        return int(count), float(norm_mean), float(norm_var)

    # --- FEATURE: T-CROSSINGS ---
    def get_t_crossings(self):
        k_w = int(self.avg_line_height * 0.2) 
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        h_lines = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, h_kernel)
        
        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        t_lengths = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # T-bar must be horizontal (w > h) and not too huge (not an underline)
            if w > h * 1.5 and w < self.avg_line_height * 3:
                count += 1
                t_lengths.append(w)

        if count == 0: return 0, 0.2, 0.2
        avg_len = np.mean(t_lengths)
        norm_len = self._force_visible_norm(avg_len / self.avg_line_height, multiplier=1.0)
        
        return int(count), float(norm_len), 0.85

    # --- FEATURE: DIACRITICS ---
    def get_diacritic_features(self):
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        stems = []
        min_area = 3 
        max_area = (self.avg_line_height ** 2) * 0.25
        
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if min_area < area < max_area:
                if 0.2 < aspect < 4.0: dots.append((x + w/2, y + h/2))
            elif h > self.avg_line_height * 0.5:
                stems.append((x + w/2, y + h/2))
                
        if not dots: return 0, 0.2, 0.2
        
        stems_arr = np.array(stems)
        distances = []
        if len(stems) > 0:
            for d in dots:
                d_arr = np.array([d])
                dist = np.min(np.linalg.norm(stems_arr - d_arr, axis=1))
                distances.append(dist)
        else:
            distances = [15.0] * len(dots)

        mean_dist = np.mean(distances)
        var_dist = np.var(distances)
        
        norm_mean = self._force_visible_norm(mean_dist / self.avg_line_height, multiplier=1.0)
        norm_var = self._force_visible_norm(math.sqrt(var_dist) / self.avg_line_height, multiplier=1.0)
        
        return int(len(dots)), float(norm_mean), float(norm_var)

    # --- FEATURE: TREMORS ---
    def get_tremors(self):
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_complexity = 0
        total_len = 0
        for c in contours:
            if cv2.contourArea(c) > 30:
                length = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.5, True)
                total_complexity += len(approx)
                total_len += length
        
        if total_len == 0: return 0.2
        density = total_complexity / total_len
        return self._force_visible_norm(density, multiplier=5.0)

    # --- FEATURE: EMBELLISHMENT ---
    def get_embellishment_score(self):
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_ink_area = 0
        total_box_area = 0
        for c in contours:
            if cv2.contourArea(c) > 30:
                total_ink_area += cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                total_box_area += (w * h)
        if total_box_area == 0: return 0.2
        ratio = total_ink_area / total_box_area
        return self._force_visible_norm(ratio, multiplier=2.0)

    # --- FEATURE: RETOUCHING ---
    def get_retouching_score(self):
        dist = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)
        vals = dist[dist > 0]
        if len(vals) == 0: return 0.2
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if mean_v == 0: return 0.2
        cv = std_v / mean_v
        return self._force_visible_norm(cv, multiplier=1.5)

    # --- FEATURE: PEN LIFTS ---
    def get_pen_lifts(self):
        n_strokes, _ = cv2.connectedComponents(self.skeleton)
        kernel = np.ones((5, 15), np.uint8)
        dilated = cv2.dilate(self.binary, kernel, iterations=2)
        n_words, _ = cv2.connectedComponents(dilated)
        if n_words == 0: return 0.5
        spw = n_strokes / max(n_words, 1)
        return self._force_visible_norm(spw, multiplier=0.2)

    def generate_report(self):
        t_cnt, t_len, t_cons = self.get_t_crossings()
        l_cnt, l_mean, l_var = self.get_loop_features()
        d_cnt, d_mean, d_var = self.get_diacritic_features()
        ws_px, ws_norm = self.get_word_spacing()
        
        # Calculate scores
        pen_lifts = self.get_pen_lifts()
        tremor = self.get_tremors()
        embellish = self.get_embellishment_score()
        retouch = self.get_retouching_score()

        logging.info(f"DEBUG -> LoopCnt: {l_cnt}, Tremor: {tremor}, Embellish: {embellish}, WS_Norm: {ws_norm}")

        return {
            "Word_Spacing_Norm": float(ws_norm), 
            "Pen_Lifts_Score": float(pen_lifts),
            "Tremor_Index": float(tremor),
            "Embellishment_Index": float(embellish),
            "Retouching_Index": float(retouch),
            
            "Loops_Count": int(l_cnt),
            "Diacritics_Count": int(d_cnt),
            "T_Crossing_Count": int(t_cnt),
            
            "Loop_Area_Mean": float(l_mean),
            "Loop_Area_Var": float(l_var),
            "Diacritic_Offset_Mean": float(d_mean),
            "Diacritic_Offset_Var": float(d_var),
            "T_Bar_Height_Mean": float(t_len), 
            "T_Bar_Height_Var": float(t_cons), 
            
            "Word_Spacing_Px": int(ws_px),
            "Word_Spacing_Score": float(ws_norm)
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
        
        logging.info(f"Analysis Complete.")
        
        csv_file = "analysis_results_full.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['Filename'] + list(results.keys()))
            writer.writerow([file.filename] + list(results.values()))
            
        return {"filename": file.filename, "data": results}

    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")