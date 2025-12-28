import io
import cv2
import numpy as np
import traceback
import math
import hashlib
import json
import sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

# --- CONFIGURATION ---
CACHE_DB_PATH = "analysis_cache.db"

# --- DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS results 
                     (hash TEXT PRIMARY KEY, json_data TEXT)''')
        conn.commit()
        conn.close()
        print("Local database initialized.")
    except Exception as e:
        print(f"Database error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Local Graphology Engine", version="5.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- DATA MODELS ---

# 1. The Raw Metrics (Parameters you can upload via JSON)
class HandwritingMetrics(BaseModel):
    total_loops: float
    slant_degrees: float
    spacing_mean_word: float
    retouching_index: float  # 0.0 to 1.0
    tremor_index: float      # 0.0 to 1.0
    pressure_index: float    # 0.0 (Light) to 1.0 (Heavy)
    baseline_angle: float    # degrees
    
# 2. The Output Format (OCEAN Scores)
class PersonalityProfile(BaseModel):
    metrics: HandwritingMetrics
    ocean_scores: Dict[str, float]  # 1.0 to 5.0
    summary: str
    indicators: List[str]

# --- LOCAL COMPUTER VISION ENGINE (No AI API) ---
def extract_local_metrics(image_bytes: bytes) -> HandwritingMetrics:
    # 1. Read Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Invalid Image")
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pressure Estimation (Darker pixels = heavier pressure)
    # We invert gray so ink is bright, measuring mean intensity of ink pixels
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_pixels = gray[binary_inv > 0]
    # Lower value in gray means darker ink. We invert this logic for index.
    # 0 = white paper, 255 = black ink.
    if len(ink_pixels) > 0:
        mean_intensity = np.mean(ink_pixels)
        pressure_idx = 1.0 - (mean_intensity / 255.0) # 1.0 is pure black (heavy)
    else:
        pressure_idx = 0.5

    # 3. Loops Detection (Using Contours)
    # Find contours on the binary inverted image
    contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    loop_count = 0
    # Hierarchy: [Next, Previous, First_Child, Parent]
    # A contour is a "hole" (loop) if it has a parent.
    if hierarchy is not None:
        for i in range(len(contours)):
            parent_idx = hierarchy[0][i][3]
            if parent_idx != -1: # It has a parent, so it's an internal hole
                area = cv2.contourArea(contours[i])
                if 10 < area < 500: # Filter noise and huge shapes
                    loop_count += 1
    
    # 4. Slant Calculation (Skeleton Method)
    skeleton = skeletonize(binary_inv > 0)
    # Sobel gradients
    dy = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 0, 1)
    dx = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 1, 0)
    angles = np.degrees(np.arctan2(dy, dx))
    # Filter for vertical-ish strokes
    valid_angles = angles[((angles > 30) & (angles < 150)) | ((angles < -30) & (angles > -150))]
    slant = 0.0
    if len(valid_angles) > 0:
        raw_slant = np.median(90 - np.abs(valid_angles))
        # Determine direction based on simple logic or assume right slant dominance for average
        slant = raw_slant # Simplified for robustness
        
    # 5. Spacing & Baseline
    # Create bounding boxes for words (using dilation to connect letters)
    kernel = np.ones((5, 20), np.uint8) # Wide kernel to connect letters
    dilated = cv2.dilate(binary_inv, kernel, iterations=1)
    word_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spacings = []
    # Sort contours left-to-right, top-to-bottom logic needed for true text,
    # simplified here to average distance between adjacent boxes in x-plane
    sorted_cnts = sorted(word_contours, key=lambda c: cv2.boundingRect(c)[0])
    for i in range(len(sorted_cnts)-1):
        x1, _, w1, _ = cv2.boundingRect(sorted_cnts[i])
        x2, _, _, _ = cv2.boundingRect(sorted_cnts[i+1])
        dist = x2 - (x1 + w1)
        if 5 < dist < 200: # Valid spacing
            spacings.append(dist)
            
    avg_spacing = np.mean(spacings) if spacings else 20.0
    
    return HandwritingMetrics(
        total_loops=float(loop_count),
        slant_degrees=float(slant),
        spacing_mean_word=float(avg_spacing),
        retouching_index=0.1, # Hard to detect without tablet data, default low
        tremor_index=0.1,     # Hard to detect with simple CV, default low
        pressure_index=float(pressure_idx),
        baseline_angle=0.0
    )

# --- THE RULE ENGINE (Based on your DOCX) ---
def apply_handwriting_rules(m: HandwritingMetrics) -> PersonalityProfile:
    scores = {
        "Openness": 3.0, "Conscientiousness": 3.0, "Extraversion": 3.0,
        "Agreeableness": 3.0, "Neuroticism": 3.0
    }
    indicators = []

    # 1. LOOPS (Reflects expression/fluency)
    # Table: "Closed or semi-closed curved formations... Reflects expressive behavior"
    if m.total_loops > 30:
        scores["Openness"] += 0.8
        scores["Extraversion"] += 0.5
        indicators.append("High loop frequency indicates expressiveness and creativity.")
    elif m.total_loops < 10:
        scores["Conscientiousness"] += 0.5
        scores["Extraversion"] -= 0.3
        indicators.append("Few loops suggest practicality and simplified thinking.")

    # 2. SPACING (Reflects spatial organization/pacing)
    # Table: "Mean spacing distance... Reflects spatial organisation"
    if m.spacing_mean_word > 40: # Wide spacing
        scores["Openness"] += 0.5 # Independence
        scores["Extraversion"] -= 0.4
        indicators.append("Wide word spacing suggests a need for personal space and independence.")
    elif m.spacing_mean_word < 15: # Narrow spacing
        scores["Extraversion"] += 0.6
        scores["Agreeableness"] += 0.4
        indicators.append("Narrow word spacing indicates a desire for social closeness.")

    # 3. PRESSURE (Deduced from text density)
    if m.pressure_index > 0.7: # Heavy
        scores["Neuroticism"] -= 0.2
        scores["Extraversion"] += 0.4
        indicators.append("Heavy pressure suggests high energy and commitment.")
    elif m.pressure_index < 0.4: # Light
        scores["Neuroticism"] += 0.3
        scores["Agreeableness"] += 0.2
        indicators.append("Light pressure may indicate sensitivity or adaptability.")

    # 4. SLANT (Ascender analysis)
    # Table: "Ascender angle detection... Reflects emotional direction"
    if m.slant_degrees > 15: # Right Slant
        scores["Extraversion"] += 0.7
        scores["Openness"] += 0.3
        indicators.append("Rightward slant is strongly linked to sociability and emotional responsiveness.")
    elif m.slant_degrees < -5: # Left Slant
        scores["Extraversion"] -= 0.6
        scores["Conscientiousness"] += 0.4
        indicators.append("Leftward slant often indicates reserve and introspection.")
    else: # Vertical
        scores["Conscientiousness"] += 0.5
        indicators.append("Vertical writing suggests logic and independence.")

    # Clamp scores between 1.0 and 5.0
    final_scores = {k: round(max(1.0, min(5.0, v)), 1) for k, v in scores.items()}
    
    # Generate Summary
    dominant_trait = max(final_scores, key=final_scores.get)
    summary = f"The profile is characterized by high {dominant_trait}, suggesting a personality that is "
    if dominant_trait == "Extraversion": summary += "social and energetic."
    elif dominant_trait == "Conscientiousness": summary += "organized and disciplined."
    elif dominant_trait == "Openness": summary += "creative and open to new ideas."
    elif dominant_trait == "Agreeableness": summary += "cooperative and compassionate."
    else: summary += "sensitive and emotionally reactive."

    return PersonalityProfile(
        metrics=m,
        ocean_scores=final_scores,
        summary=summary,
        indicators=indicators
    )

# --- API ENDPOINTS ---

@app.post("/analyze/image", response_model=PersonalityProfile)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    
    # Check cache
    img_hash = hashlib.sha256(content).hexdigest()
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT json_data FROM results WHERE hash=?", (img_hash,))
    row = cursor.fetchone()
    
    if row:
        conn.close()
        print("Returning cached result.")
        return PersonalityProfile.model_validate_json(row[0])
    
    # Local Processing (No API Key)
    try:
        raw_metrics = extract_local_metrics(content)
        profile = apply_handwriting_rules(raw_metrics)
        
        # Save to cache
        cursor.execute("INSERT OR REPLACE INTO results VALUES (?, ?)", 
                      (img_hash, profile.model_dump_json()))
        conn.commit()
        conn.close()
        
        return profile
    except Exception as e:
        if conn: conn.close()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/json", response_model=PersonalityProfile)
async def analyze_json_endpoint(metrics: HandwritingMetrics):
    """
    Directly upload the parameters (loops, slant, etc.) to get a prediction.
    Useful if you have the data from elsewhere.
    """
    return apply_handwriting_rules(metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)