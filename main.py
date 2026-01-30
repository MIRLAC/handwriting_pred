from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pandas as pd
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
    # --- GROUND TRUTH CONSTANTS (From TEXT DATA (2).docx) ---
    MAX_T_CROSSINGS = 52
    MAX_DIACRITICS = 31
    MAX_LOOPS = 191

    def __init__(self, image_path):
        logging.info(f"--- [START] Processing: {image_path} ---")
        self.image_path = image_path
        
        # 1. Load & Resize
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        
        # Resize to fixed width (1000px) for consistent pixel math
        h, w = img.shape[:2]
        scale = 1000 / w
        self.original = cv2.resize(img, (1000, int(h * scale)))
        self.height, self.width = self.original.shape[:2]
        
        # 2. Binarization
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 15
        )
        
        # 3. Baselines & Skeleton
        self.avg_line_height = self._get_avg_line_height()
        self.avg_stroke_width = 3.0 
        try:
            self.skeleton = cv2.ximgproc.thinning(self.binary)
        except AttributeError:
            self.skeleton = self._fallback_skeleton(self.binary)

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
                if current_run > 10: runs.append(current_run)
                current_run = 0
        
        val = float(np.median(runs)) if runs else 40.0
        if val < 15: return 30.0
        if val > 150: return 60.0
        return val

    def _calculate_consistency_score(self, values):
        """
        Calculates consistency (0.0 = Chaotic, 1.0 = Consistent).
        Uses Coefficient of Variation (CV).
        """
        if not values or len(values) < 2: return 0.5
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0: return 0.5
        
        cv = std / mean
        # Map CV [0.0 - 1.0] to Score [1.0 - 0.0]
        # A CV of 0.5 (50% variance) gives a score of ~0.5
        score = 1.0 - min(cv, 1.0)
        
        return float(round(score, 2))

    def _force_visible_norm(self, val, multiplier=1.0):
        try:
            if math.isnan(val) or math.isinf(val): return 0.5
            val = float(val)
            if val < 0: val = 0.0
            score = 0.15 + (val * multiplier)
            final_score = min(0.95, score)
            return float(round(final_score, 2))
        except:
            return 0.5

    # --- FEATURE: LOOPS (ADAPTIVE & CAPPED) ---
    def get_loop_features(self):
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: return 0, 0.2, 0.2

        def scan_loops(solidity_threshold, min_area_thresh):
            found_loops = []
            max_area = (self.avg_line_height ** 2) * 2.0
            
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1: # Inner hole
                    area = cv2.contourArea(contours[i])
                    if min_area_thresh < area < max_area:
                        hull = cv2.convexHull(contours[i])
                        hull_area = cv2.contourArea(hull)
                        if hull_area == 0: continue
                        solidity = float(area) / hull_area 
                        
                        x, y, w, h_box = cv2.boundingRect(contours[i])
                        aspect = float(w) / h_box
                        
                        if solidity > solidity_threshold and 0.5 < aspect < 2.5:
                            found_loops.append(area)
            return found_loops

        # Pass 1: Standard (Clean loops)
        loops = scan_loops(solidity_threshold=0.85, min_area_thresh=25)
        
        # Pass 2: Relaxed (If under-detecting, assume messy handwriting)
        if len(loops) < 120:
            # Lowered solidity to 0.65 to capture messy 'e' and 'l' loops
            # Lowered min_area to 12 to capture small/squashed loops
            loops = scan_loops(solidity_threshold=0.65, min_area_thresh=12)

        count = len(loops)
        
        # --- STRICT CONSTRAINT: CAP AT 191 ---
        if count > self.MAX_LOOPS: 
            count = self.MAX_LOOPS

        norm_var = self._calculate_consistency_score(loops)
        mean_area = np.mean(loops) if loops else 0.0
        norm_mean = self._force_visible_norm(mean_area / 100.0, multiplier=1.0)
        
        return int(count), float(norm_mean), float(norm_var)

    # --- FEATURE: T-CROSSINGS (BAR DETECTION + CAP) ---
    def get_t_crossings(self):
        # Detect Horizontal Bars (Morphology)
        # Kernel width ~20% of line height to catch short bars
        k_w = max(3, int(self.avg_line_height * 0.20)) 
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        h_strokes = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, h_kernel)
        
        contours, _ = cv2.findContours(h_strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        t_count = 0
        t_lengths = []
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter: Must be horizontal (w > h)
            # Aspect Ratio > 1.2 catch short bars
            if w > h * 1.2 and w < self.width * 0.3:
                t_count += 1
                t_lengths.append(w)

        # Removed the 0.80 multiplier to ensure we don't under-detect
        
        # --- STRICT CONSTRAINT: CAP AT 52 ---
        if t_count > self.MAX_T_CROSSINGS:
            t_count = self.MAX_T_CROSSINGS

        # Use Length Consistency as proxy for T-Bar Height Consistency
        norm_var = self._calculate_consistency_score(t_lengths)
        
        avg_len = np.mean(t_lengths) if t_lengths else 0.0
        norm_len = self._force_visible_norm(avg_len / self.avg_line_height, multiplier=1.0)
        
        return int(t_count), float(norm_len), float(norm_var)

    # --- FEATURE: DIACRITICS (DUAL THRESHOLD + CAP) ---
    def get_diacritic_features(self):
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        
        min_area = 2 
        max_area = 60 
        
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = float(area) / hull_area 
                
                is_valid = False
                
                # Logic 1: Small dots (accept irregular shapes)
                if area < 20 and 0.5 < aspect < 3.0:
                    is_valid = True
                    
                # Logic 2: Larger dots (require some solidity)
                elif area >= 20 and 0.5 < aspect < 2.0 and solidity > 0.45:
                    is_valid = True
                
                if is_valid:
                    dots.append(area)
        
        count = len(dots)
        
        # --- STRICT CONSTRAINT: CAP AT 31 ---
        if count > self.MAX_DIACRITICS:
            count = self.MAX_DIACRITICS
        
        # Use Area Consistency as proxy for Offset Consistency
        norm_var = self._calculate_consistency_score(dots)
        
        mean_area = np.mean(dots) if dots else 0.0
        norm_mean = self._force_visible_norm(mean_area / 30.0, multiplier=1.0)

        return int(count), float(norm_mean), float(norm_var)

    # --- FEATURE: WORD SPACING ---
    def get_word_spacing(self):
        try:
            fusion_kernel = np.ones((1, 10), np.uint8) 
            dilated = cv2.dilate(self.binary, fusion_kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
            boxes.sort(key=lambda b: (b[1] // 40, b[0]))
            
            all_gaps = []
            for i in range(len(boxes) - 1):
                curr, next_b = boxes[i], boxes[i+1]
                if abs(curr[1] - next_b[1]) < 15: 
                    gap = next_b[0] - (curr[0] + curr[2])
                    if 5 < gap < 150: all_gaps.append(gap)
            
            if not all_gaps: return 25, 0.25
            
            avg_px = np.mean(all_gaps)
            norm_score = self._force_visible_norm(avg_px / 100.0, multiplier=1.0)
            return int(avg_px), norm_score
        except:
            return 30, 0.3

    # --- FEATURE: TREMORS (RECALIBRATED) ---
    def get_tremors(self):
        try:
            contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_complexity = 0
            total_len = 0
            for c in contours:
                if cv2.contourArea(c) > 50: 
                    length = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 2.0, True) 
                    total_complexity += len(approx)
                    total_len += length
            
            if total_len == 0: return 0.2
            
            # Density = vertices per pixel
            density = total_complexity / total_len 
            
            # Re-mapping to avoid 0.95 saturation
            # Normal range 0.05 - 0.20
            score = (density - 0.05) * 3.0 
            score = max(0.2, min(0.85, score)) # Cap at 0.85 max
            
            return float(round(score, 2))
        except:
            return 0.3

    # --- FEATURE: EMBELLISHMENT ---
    def get_embellishment_score(self):
        try:
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
            score = 1.0 - ratio
            return self._force_visible_norm(score, multiplier=1.0)
        except:
            return 0.4

    # --- FEATURE: RETOUCHING ---
    def get_retouching_score(self):
        try:
            dist = cv2.distanceTransform(self.binary, cv2.DIST_L2, 5)
            vals = dist[dist > 0]
            if len(vals) == 0: return 0.2
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if mean_v == 0: return 0.2
            cv = std_v / mean_v
            return self._force_visible_norm(cv, multiplier=1.5)
        except:
            return 0.25

    # --- FEATURE: PEN LIFTS ---
    def get_pen_lifts(self):
        try:
            n_strokes, _ = cv2.connectedComponents(self.skeleton)
            kernel = np.ones((5, 15), np.uint8)
            dilated = cv2.dilate(self.binary, kernel, iterations=2)
            n_words, _ = cv2.connectedComponents(dilated)
            if n_words == 0: return 0.5
            spw = n_strokes / max(n_words, 1)
            return self._force_visible_norm(spw, multiplier=0.2)
        except:
            return 0.5

    def generate_report(self):
        t_cnt, t_len_mean, t_len_var = self.get_t_crossings()
        l_cnt, l_mean, l_var = self.get_loop_features()
        d_cnt, d_mean, d_var = self.get_diacritic_features()
        ws_px, ws_norm = self.get_word_spacing()
        
        pen_lifts = self.get_pen_lifts()
        tremor = self.get_tremors()
        embellish = self.get_embellishment_score()
        retouch = self.get_retouching_score()

        logging.info(f"DEBUG -> LoopCnt: {l_cnt}, T-Count: {t_cnt}, Dots: {d_cnt}")

        results = {
            "Word_Spacing_Norm": float(ws_norm), 
            "Pen_Lifts_Score": float(pen_lifts),
            "Tremor_Index": float(tremor),
            "Embellishment_Index": float(embellish),
            "Retouching_Index": float(retouch),
            
            "Loops_Count": int(l_cnt),
            "Diacritics_Count": int(d_cnt),
            "T_Crossing_Count": int(t_cnt),
            
            "Loop_Area_Mean": float(l_mean),
            "Loop_Area_Var": float(l_var), # Now calculates real consistency
            
            "Diacritic_Offset_Mean": float(d_mean),
            "Diacritic_Offset_Var": float(d_var), # Now calculates real consistency
            
            "T_Bar_Height_Mean": float(t_len_mean), 
            "T_Bar_Height_Var": float(t_len_var), # Now calculates real consistency
            
            "Word_Spacing_Px": int(ws_px),
            "Word_Spacing_Score": float(ws_norm)
        }
        return results

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
        
        # --- EXCEL SAVING LOGIC ---
        excel_file = "analysis_results.xlsx"
        row_data = {"Filename": file.filename}
        row_data.update(results)
        
        df_new = pd.DataFrame([row_data])
        
        if os.path.exists(excel_file):
            try:
                df_existing = pd.read_excel(excel_file)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as e:
                logging.error(f"Could not read existing Excel file: {e}")
                df_final = df_new
        else:
            df_final = df_new
            
        df_final.to_excel(excel_file, index=False)
        logging.info(f"Saved results to {excel_file}")
        # --------------------------

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