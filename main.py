from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import csv
import logging
import shutil
import os
import sys

# --- 1. SETUP LOGGING (Windows-Safe) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()

# --- 2. ENABLE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. FEATURE EXTRACTION LOGIC ---
class HandwritingFeatures:
    def __init__(self, image_path):
        self.image_path = image_path
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError("Image not found.")
            
        # Normalization
        height, width = original_img.shape[:2]
        target_width = 1000
        scale = target_width / width
        new_height = int(height * scale)
        self.original = cv2.resize(original_img, (target_width, new_height))
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Detail Layer
        self.blur_detail = cv2.GaussianBlur(self.gray, (1, 1), 0)
        self.thresh_detail = cv2.adaptiveThreshold(
            self.blur_detail, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        self.contours_detail, self.hierarchy_detail = cv2.findContours(
            self.thresh_detail, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Structure Layer
        self.blur_struct = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.thresh_struct = cv2.adaptiveThreshold(
            self.blur_struct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 15
        )
        kernel = np.ones((3,3), np.uint8)
        self.thresh_struct = cv2.morphologyEx(self.thresh_struct, cv2.MORPH_OPEN, kernel, iterations=1)
        self.contours_struct, _ = cv2.findContours(
            self.thresh_struct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # State for advanced metrics
        self.loop_circularities = []
        self.spacing_variances = []

    def get_loops(self):
        count = 0
        self.loop_circularities = [] # Reset
        if self.hierarchy_detail is not None:
            for i in range(len(self.contours_detail)):
                if self.hierarchy_detail[0][i][3] != -1:
                    c = self.contours_detail[i]
                    area = cv2.contourArea(c)
                    if 40 < area < 1000:
                        perimeter = cv2.arcLength(c, True)
                        if perimeter == 0: continue
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        if hull_area == 0: continue
                        solidity = area / hull_area
                        
                        if circularity > 0.4 and solidity > 0.6:
                            count += 1
                            self.loop_circularities.append(circularity)
        return count

    def get_diacritics(self):
        count = 0
        if not self.contours_detail: return 0
        areas = [cv2.contourArea(c) for c in self.contours_detail]
        mean_area = np.mean(areas) if areas else 50
        for c in self.contours_detail:
            area = cv2.contourArea(c)
            if 10 < area < (mean_area * 0.5):
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = area / hull_area
                if solidity > 0.5:
                    count += 1
        return count

    def get_tremors(self):
        score = 0
        valid_contours = 0
        for c in self.contours_detail:
            if cv2.contourArea(c) > 50:
                valid_contours += 1
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
                score += len(approx) / (len(c) + 1)
        return round(score / valid_contours, 4) if valid_contours > 0 else 0

    def get_embellishments(self):
        scores = []
        for c in self.contours_detail:
            area = cv2.contourArea(c)
            if area > 30:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    scores.append(1 - circularity)
        return round(np.mean(scores), 4) if scores else 0

    def get_word_spacing_stats(self):
        # Returns (Mean Spacing, Standard Deviation)
        kernel = np.ones((4, 10), np.uint8) 
        dilated = cv2.dilate(self.thresh_struct, kernel, iterations=1)
        word_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = [cv2.boundingRect(c) for c in word_contours if cv2.contourArea(c) > 100]
        boxes.sort(key=lambda x: x[1])
        
        lines = {}
        for box in boxes:
            y_center = box[1] + box[3]//2
            found = False
            for y_key in lines:
                if abs(y_key - y_center) < 20: 
                    lines[y_key].append(box)
                    found = True
                    break
            if not found:
                lines[y_center] = [box]
        
        spacings = []
        for y_key in lines:
            line = sorted(lines[y_key], key=lambda x: x[0])
            for i in range(len(line) - 1):
                dist = line[i+1][0] - (line[i][0] + line[i][2])
                if 5 < dist < 300: 
                    spacings.append(dist)
        
        if not spacings: return 0, 0
        return round(np.mean(spacings), 2), round(np.std(spacings), 2)

    def get_pen_lifts(self):
        valid_strokes = [c for c in self.contours_detail if cv2.contourArea(c) > 15]
        return len(valid_strokes)

    def get_t_crossings(self):
        lines = cv2.HoughLinesP(self.thresh_struct, 1, np.pi / 180, 
                               threshold=50, minLineLength=15, maxLineGap=3)
        count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 20 and 15 < length < 60:
                    count += 1
        return count

    def get_retouching(self):
        edges = cv2.Canny(self.blur_detail, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        ink_pixels = np.count_nonzero(self.thresh_detail)
        return round(edge_pixels / ink_pixels, 4) if ink_pixels > 0 else 0

    # --- PROFESSIONAL INTERPRETATION LOGIC (From DOCX) ---
    def generate_professional_report(self):
        # 1. Gather Raw Metrics
        loop_count = self.get_loops()
        mean_spacing, std_dev_spacing = self.get_word_spacing_stats()
        
        raw = {
            "Loops": loop_count,
            "Embellishments": self.get_embellishments(),
            "Diacritics": self.get_diacritics(),
            "Retouching": self.get_retouching(),
            "Word_Spacing_px": mean_spacing,
            "Spacing_Consistency": std_dev_spacing,
            "T_Crossing": self.get_t_crossings(),
            "Pen_Lifts": self.get_pen_lifts(),
            "Tremors": self.get_tremors()
        }

        # 2. Map to Document Categories
        categories = {}

        # -- LOOPS (Narrow, Wide, Angular, No loops) --
        if raw["Loops"] < 10:
            categories["Loop_Style"] = "No loops / Retracted"
        else:
            avg_circ = np.mean(self.loop_circularities) if self.loop_circularities else 0
            if avg_circ > 0.8: categories["Loop_Style"] = "Wide Loops (Round)"
            elif avg_circ > 0.6: categories["Loop_Style"] = "Narrow Loops"
            else: categories["Loop_Style"] = "Angular / Compressed Loops"

        # -- WORD SPACING (Narrow, Normal, Wide, Inconsistent) --
        # Using Std Dev to detect inconsistency
        if raw["Spacing_Consistency"] > 25: 
            categories["Spacing_Type"] = "Inconsistent"
        elif raw["Word_Spacing_px"] < 40: 
            categories["Spacing_Type"] = "Narrow"
        elif raw["Word_Spacing_px"] > 65: 
            categories["Spacing_Type"] = "Wide"
        else: 
            categories["Spacing_Type"] = "Normal"

        # -- TREMORS (Abundant, Less Prevalent) --
        if raw["Tremors"] > 0.18:
            categories["Tremors_Category"] = "Abundant (Shaky)"
        else:
            categories["Tremors_Category"] = "Less Prevalent (Stable)"

        # -- EMBELLISHMENTS (Abundant, Less Prevalent) --
        if raw["Embellishments"] > 0.70:
            categories["Embellishments_Category"] = "Abundant (Decorated)"
        else:
            categories["Embellishments_Category"] = "Less Prevalent (Simple)"

        # -- PEN LIFTS (Abundant, Less Prevalent) --
        # High count = Print (Abundant lifts). Low count = Cursive (Less lifts).
        if raw["Pen_Lifts"] > 300: 
            categories["Pen_Lifts_Category"] = "Abundant (Disconnected)"
        else:
            categories["Pen_Lifts_Category"] = "Less Prevalent (Fluid)"

        # -- RETOUCHING (Abundant, Less Prevalent) --
        if raw["Retouching"] > 0.75:
            categories["Retouching_Category"] = "Abundant (Overwritten)"
        else:
            categories["Retouching_Category"] = "Less Prevalent (Clean)"

        # -- T-CROSSINGS (High/Middle/Low - Proxy via Count) --
        # Note: Computer vision cannot easily detect "Height on stem" without OCR.
        # We classify by FREQUENCY as a proxy for "Attention to detail".
        if raw["T_Crossing"] > 20:
            categories["T_Crossing_Frequency"] = "Frequent (High Attention)"
        else:
            categories["T_Crossing_Frequency"] = "Sparse (Low Attention)"

        # -- DIACRITICS (Proper/Displaced - Proxy via Count) --
        if raw["Diacritics"] > 20:
            categories["Diacritics_Category"] = "Abundant (Precise)"
        else:
            categories["Diacritics_Category"] = "Less Prevalent (Omitted)"

        return {**raw, **categories}

# --- 4. API ENDPOINTS ---
# --- 4. API ENDPOINTS ---
@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    logging.info(f"[>>] Received Request: {file.filename}")
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        logging.info("[..] Starting professional analysis...")
        
        extractor = HandwritingFeatures(temp_filename)
        results = extractor.generate_professional_report()
        
        logging.info(f"[OK] Analysis Complete.")
        
        # --- MODIFIED CSV SAVING LOGIC ---
        # We explicitly list only the columns we want (A to I).
        # This excludes Column J (Tremors) and all Categories (K onwards).
        csv_columns = [
            "Filename", 
            "Loops", 
            "Embellishments", 
            "Diacritics", 
            "Retouching", 
            "Word_Spacing_px", 
            "Spacing_Consistency", 
            "T_Crossing", 
            "Pen_Lifts"
        ]
        
        csv_file = "analysis_results.csv"
        try:
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write Header if file is new
                if not file_exists:
                    writer.writerow(csv_columns)
                
                # Build the row data strictly based on our allowed columns
                # 1. Filename
                row_data = [file.filename]
                # 2. The rest of the metrics from the 'results' dictionary
                for col in csv_columns[1:]:
                    row_data.append(results.get(col, 0))
                
                writer.writerow(row_data)
                
            logging.info(f"[SAVE] Saved to {csv_file}")
        except PermissionError:
            logging.warning(f"[WARN] CSV File Locked.")
            
        return {"filename": file.filename, "data": results}

    except Exception as e:
        logging.error(f"[!!] CRITICAL ERROR: {str(e)}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        logging.info("[XX] Cleanup done.")