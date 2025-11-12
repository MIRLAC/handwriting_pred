# main.py
import os
import io
import json
import cv2
import numpy as np
import google.generativeai as genai
import traceback
import hashlib
import math 
import sqlite3 # --- Import sqlite3 for persistent caching ---
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError 
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from paddleocr import PaddleOCR
# --- Imports for CV Slant ---
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
# --- End Imports ---

# This dictionary will hold our models and configurations
ml_models = {}

# --- Define database path ---
CACHE_DB_PATH = "analysis_cache.db"

# --- This 'lifespan' function runs on application startup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env file. Please check your .env file.")

    # Configure Gemini
    genai.configure(
        api_key=api_key,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
    print("Gemini API configured successfully.")

    # Load PaddleOCR model
    print("Loading PaddleOCR model...")
    ml_models["ocr"] = PaddleOCR(use_angle_cls=True, lang='en')
    print("PaddleOCR model loaded successfully.")
    
    # --- Initialize the persistent cache database ---
    try:
        print(f"Initializing persistent cache at {CACHE_DB_PATH}...")
        db = sqlite3.connect(CACHE_DB_PATH) # This line now works
        cursor = db.cursor()
        # Create table if it doesn't exist.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_cache (
            image_hash TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
        )
        """)
        db.commit()
        db.close()
        print("Cache database initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize cache database: {e}")
    
    yield  # The application runs here

    # --- Shutdown ---
    print("Application shutting down.")
    ml_models.clear()

# The FastAPI app instance is defined AFTER the lifespan function.
app = FastAPI(
    title="Hybrid Handwriting Analyzer",
    version="3.9.2", # Version updated for 1-5 OCEAN scale
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- Pydantic Models ---
class Diagnostics(BaseModel):
    overall_summary: str # Technical summary from AI
    personality_summary: Optional[str] = None # Text summary based on OCEAN scores
    personality_indicators: Optional[str] = None # Detailed descriptive text
    total_chars: int
    estimated_pressure: Optional[str] = None
    baseline_consistency: Optional[str] = None
    # --- Fields for OCEAN trait scores ---
    Openness: Optional[float] = None
    Conscientiousness: Optional[float] = None
    Extraversion: Optional[float] = None
    Agreeableness: Optional[float] = None
    Neuroticism: Optional[float] = None


class TraitResult(BaseModel):
    total_loops: float # Renamed field
    diacritics_i_j_detection_rate: float # Keep rate internally for logic if needed
    t_crossing_detection_rate: float # Keep rate internally for logic if needed
    # Add integer counts for display
    total_correct_diacritics: Optional[int] = None # Estimated count
    total_correct_t_crossings: Optional[int] = None # Estimated count
    spacing_mean_char: float # Now estimated via Gemini
    spacing_std_char: float  # Now estimated via Gemini
    spacing_mean_word: float # Now estimated via Gemini
    spacing_std_word: float  # Now estimated via Gemini
    pen_lifts_per_word: float
    retouching_index: float
    tremor_index: float
    slant_degrees: float # Now calculated via CV
    diagnostics: Diagnostics

# --- Helper Functions ---
def read_image(file_bytes: bytes) -> np.ndarray:
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # Ensure image is valid
    if img is None:
        print("Warning: cv2.imdecode returned None. Invalid image data.")
        return None
    # Ensure image has 3 channels (BGR)
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4: # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Warning: Unexpected image shape {img.shape}")
        return None # Or handle differently
    return img


def resize_image_if_needed(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    if img is None: return None # Handle None input
    height, width = img.shape[:2]
    if max(height, width) > max_side:
        scale_factor = max_side / max(height, width)
        new_w = int(width * scale_factor)
        new_h = int(height * scale_factor)
        # Ensure dimensions are at least 1 pixel
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Image resized from ({width}x{height}) to ({new_w}x{new_h})")
        return resized_img
    return img

# --- Function to run OCR and count characters ---
def run_ocr_and_count(image_bgr: np.ndarray) -> Tuple[int, List[Any]]:
    """Runs PaddleOCR and returns the total character count (excluding spaces) and raw OCR results (for potential future use)."""
    ocr_model = ml_models.get("ocr")
    if not ocr_model:
        raise RuntimeError("OCR model not loaded.")

    print("Running OCR...")
    ocr_result = ocr_model.ocr(image_bgr)
    print("OCR complete.")

    total_chars = 0
    raw_results_list = [] # Store the raw line_info structure

    try:
        # Standard list-of-lists structure
        if (ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and
                isinstance(ocr_result[0], list)):
            page_results = ocr_result[0]
            if page_results:
                for line_info in page_results:
                    raw_results_list.append(line_info) # Keep storing raw results
                    if isinstance(line_info, (list, tuple)) and len(line_info) == 2:
                        _bounding_box, text_confidence_tuple = line_info
                        if isinstance(text_confidence_tuple, (tuple, list)) and len(text_confidence_tuple) == 2:
                            text, _confidence = text_confidence_tuple
                            if isinstance(text, str) and text:
                                total_chars += len(text.replace(" ", ""))

        # Dictionary structure
        elif (ocr_result and isinstance(ocr_result, list) and len(ocr_result) == 1 and
              isinstance(ocr_result[0], dict)):
            result_dict = ocr_result[0]
            raw_results_list.append(result_dict) # Store the dict
            if ('rec_texts' in result_dict and isinstance(result_dict['rec_texts'], list) and
                'rec_polys' in result_dict and isinstance(result_dict['rec_polys'], list) and
                'rec_scores' in result_dict and isinstance(result_dict['rec_scores'], list) and
                len(result_dict['rec_texts']) == len(result_dict['rec_polys']) == len(result_dict['rec_scores'])):

                for i, text in enumerate(result_dict['rec_texts']):
                    # Reconstruct line_info if needed later, but just count chars now
                    if isinstance(text, str) and text:
                        total_chars += len(text.replace(" ", ""))
    except Exception as e_proc:
        print(f"ERROR: Exception during OCR result processing: {e_proc}")
        traceback.print_exc()

    print(f"OCR calculated total characters (excluding spaces): {total_chars}")
    return total_chars, raw_results_list


# --- Image Processing Helper Functions ---
def binarize_image(gray_image: np.ndarray) -> np.ndarray:
    """ Converts a grayscale image to binary (black and white). """
    try:
        # Apply Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        thresh_val = threshold_otsu(blurred)
        # Invert: We want ink to be white (1) on black (0) background for skeletonize
        binary_image = blurred < thresh_val 
        return binary_image.astype(np.uint8) # Return 0/1 image
    except Exception as e:
        print(f"Error during binarization: {e}. Using simple threshold.")
        # Fallback to simple threshold if Otsu fails
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        return (binary_image / 255).astype(np.uint8) # Normalize to 0/1


def skeletonize_image(binary_image: np.ndarray) -> np.ndarray:
    """ Applies skeletonization to a binary image (ink = 1). """
    print("Performing skeletonization...")
    # Input to skimage skeletonize should be boolean
    skeleton = skeletonize(binary_image > 0) 
    print("Skeletonization complete.")
    return skeleton.astype(np.uint8) # Return 0/1 skeleton

# --- Computer Vision Slant Calculation ---
def calculate_cv_slant(skeleton_image: np.ndarray) -> float:
    """ Calculates the average slant angle from a skeletonized image using gradients. """
    print("Calculating CV Slant...")
    if np.count_nonzero(skeleton_image) < 50: 
        print("  DEBUG CV Slant: Skeleton too sparse, returning 0.")
        return 0.0

    # Use Sobel operator to find gradients
    dy = cv2.Sobel(skeleton_image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5) # Increased kernel size
    dx = cv2.Sobel(skeleton_image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5) # Increased kernel size
    
    # Calculate angles only for skeleton pixels
    mask = skeleton_image > 0
    angles_rad = np.arctan2(dy[mask], dx[mask])
    angles_deg = np.degrees(angles_rad)

    # Filter for angles corresponding to near-vertical strokes
    vertical_angles = angles_deg[
        ((angles_deg > 30) & (angles_deg < 150)) |  # Range around 90
        ((angles_deg < -30) & (angles_deg > -150)) # Range around -90
    ]

    if vertical_angles.size < 10: # Require a minimum number of samples
        print(f"  DEBUG CV Slant: Too few near-vertical strokes ({vertical_angles.size}) found, returning 0.")
        return 0.0

    slants = 90 - vertical_angles
    # Handle wrap-around near -90 degrees
    slants[slants > 90] -= 180
    slants[slants < -90] += 180 

    median_slant = float(np.median(slants))
    
    print(f"CV Slant Calculation: Found {vertical_angles.size} vertical segments. Median Slant = {median_slant:.2f} degrees.")
    # Clamp slant to a reasonable range if needed, e.g., -60 to +60
    median_slant = max(-60.0, min(60.0, median_slant))
    return round(median_slant, 2)


# --- Gemini Analysis Function ---
async def analyze_with_gemini(image_bytes: bytes) -> dict:
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    pil_image = Image.open(io.BytesIO(image_bytes))

    # --- UPDATED PROMPT: Removed slant_degrees ---
    prompt = """
    Analyze the provided handwriting image as a precise measurement tool.
    Your response MUST be a single, valid JSON object with no markdown formatting.
    Provide the single, most accurate numerical calculation/estimation for each key based on direct visual measurement from the image.
    
    The JSON object must contain these keys:
    
    - "total_loops": Float. Estimate the TOTAL number of closed loops.
    - "diacritics_i_j_detection_rate": Float (0.0-1.0). Proportion of correctly placed 'i'/'j' dots.
    - "t_crossing_detection_rate": Float (0.0-1.0). Proportion of correctly placed 't' crosses.
    - "total_correct_diacritics": Integer. Estimate the TOTAL NUMBER of correct 'i'/'j' dots.
    - "total_correct_t_crossings": Integer. Estimate the TOTAL NUMBER of correct 't' crosses.
    - "spacing_mean_char": Float. Average pixel distance between characters.
    - "spacing_std_char": Float. Standard deviation of distance between characters.
    - "spacing_mean_word": Float. Average pixel distance between words.
    - "spacing_std_word": Float. Standard deviation of distance between words.
    - "pen_lifts_per_word": Float. Average pen lifts within words.
    - "retouching_index": Float (0.0-1.0). Proportion of retraced strokes.
    - "tremor_index": Float (0.0-1.0). Degree of line shakiness (0=smooth).
    # --- SLANT_DEGREES REMOVED ---
    - "diagnostics": Object. MUST include: "overall_summary" (one sentence technical summary), "estimated_pressure" (string: "light", "medium", or "heavy"), "baseline_consistency" (string: "straight", "wavy", or "irregular").
    """
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.0,
        top_k=1,
        top_p=1
    )
    
    request_options = {"timeout": 300}

    try:
        print("Calling Gemini API for analysis (excluding slant)...") 
        response = await model.generate_content_async(
            [prompt, pil_image],
            generation_config=generation_config,
            request_options=request_options
        )
        if not response.text:
             print("Warning: Received empty response from Gemini.")
             # Provide defaults for fields Gemini should return
             return {
                 "total_loops": 0.0, "diacritics_i_j_detection_rate": 0.0, "t_crossing_detection_rate": 0.0,
                 "total_correct_diacritics": 0, "total_correct_t_crossings": 0,
                 "spacing_mean_char": 0.0, "spacing_std_char": 0.0, 
                 "spacing_mean_word": 0.0, "spacing_std_word": 0.0, 
                 "pen_lifts_per_word": 0.0, "retouching_index": 0.0, "tremor_index": 0.0, 
                 "diagnostics": {"overall_summary": "Analysis incomplete.", "estimated_pressure": None, "baseline_consistency": None}
             }

        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("Received response from Gemini.")
        
        try:
            parsed_json = json.loads(json_text)
            # Ensure required fields exist, provide defaults if missing
            keys_to_check_float = [
                "total_loops", "diacritics_i_j_detection_rate", "t_crossing_detection_rate",
                "spacing_mean_char", "spacing_std_char", "spacing_mean_word", "spacing_std_word",
                "pen_lifts_per_word", "retouching_index", "tremor_index" 
            ]
            keys_to_check_int = ["total_correct_diacritics", "total_correct_t_crossings"]

            for key in keys_to_check_float:
                if key not in parsed_json: parsed_json[key] = 0.0
            for key in keys_to_check_int:
                 if key not in parsed_json: parsed_json[key] = 0

            if 'diagnostics' not in parsed_json: parsed_json['diagnostics'] = {}
            if 'overall_summary' not in parsed_json['diagnostics']: parsed_json['diagnostics']['overall_summary'] = 'Summary N/A'
            if 'estimated_pressure' not in parsed_json['diagnostics']: parsed_json['diagnostics']['estimated_pressure'] = 'unknown'
            if 'baseline_consistency' not in parsed_json['diagnostics']: parsed_json['diagnostics']['baseline_consistency'] = 'unknown'

            return parsed_json
        except json.JSONDecodeError as json_err:
             print(f"Error decoding JSON from Gemini: {json_err}")
             print(f"Raw Gemini response text: {json_text}")
             raise HTTPException(status_code=500, detail=f"Failed to parse AI model response: {json_err}")

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        raise HTTPException(status_code=503, detail=f"Gemini API Error: {e}")

# --- UPDATED: Personality Interpretation - Calculates OCEAN scores on 1-5 scale ---
def interpret_traits_for_ocean_scores(traits: TraitResult) -> Dict[str, float]:
    """
    Interprets handwriting traits to estimate OCEAN personality scores.
    Returns numerical scores on a 1-5 scale.
    """
    # Initialize scores in the [-1, 1] range first
    scores_internal = {
        "Openness": 0.0, "Conscientiousness": 0.0, "Extraversion": 0.0,
        "Agreeableness": 0.0, "Neuroticism": 0.0,
    }

    total_chars = traits.diagnostics.total_chars if traits.diagnostics else 0
    total_loops = traits.total_loops
    loops_per_100 = (total_loops / total_chars) * 100 if total_chars > 0 else 0
    
    # Apply rules based on correlation table
    if loops_per_100 > 25: scores_internal["Openness"] += 0.4; scores_internal["Extraversion"] += 0.2
    elif loops_per_100 < 12: scores_internal["Conscientiousness"] += 0.2; scores_internal["Extraversion"] -= 0.1; scores_internal["Openness"] -= 0.2
    if traits.diacritics_i_j_detection_rate > 0.85: scores_internal["Conscientiousness"] += 0.3
    if traits.t_crossing_detection_rate > 0.85: scores_internal["Conscientiousness"] += 0.2
    if traits.retouching_index > 0.25: scores_internal["Neuroticism"] += 0.4
    elif traits.retouching_index < 0.08: scores_internal["Extraversion"] += 0.1; scores_internal["Neuroticism"] -= 0.2
    else: scores_internal["Conscientiousness"] += 0.1 
    spacing_irregular = traits.spacing_mean_word > 0 and traits.spacing_std_word > (traits.spacing_mean_word * 0.4) 
    if spacing_irregular: scores_internal["Neuroticism"] += 0.3
    elif traits.spacing_mean_word > 25: scores_internal["Openness"] += 0.3; scores_internal["Agreeableness"] -= 0.1
    elif traits.spacing_mean_word < 12: scores_internal["Extraversion"] += 0.2; scores_internal["Agreeableness"] += 0.1
    if traits.pen_lifts_per_word > 1.5: scores_internal["Conscientiousness"] += 0.2
    elif traits.pen_lifts_per_word < 0.8: scores_internal["Agreeableness"] += 0.1; scores_internal["Extraversion"] += 0.1
    if traits.tremor_index > 0.15: scores_internal["Neuroticism"] += 0.5
    elif traits.tremor_index < 0.06: scores_internal["Conscientiousness"] += 0.1; scores_internal["Neuroticism"] -= 0.3
    if traits.slant_degrees > 10: scores_internal["Extraversion"] += 0.3; scores_internal["Agreeableness"] += 0.1 
    elif traits.slant_degrees < -5: scores_internal["Extraversion"] -= 0.2
    
    # Clamp scores to [-1, 1]
    for trait in scores_internal: scores_internal[trait] = max(-1.0, min(1.0, scores_internal[trait]))

    # --- NEW: Map [-1, 1] to [1, 5] scale ---
    # 1 = Low (-1.0), 3 = Neutral (0.0), 5 = High (1.0)
    scores_1_to_5 = {}
    for trait in scores_internal:
        score_minus1_to_1 = scores_internal[trait]
        # Linear mapping: y = 2x + 3
        score_1_to_5 = (score_minus1_to_1 * 2) + 3
        scores_1_to_5[trait] = round(score_1_to_5, 1) # Round to one decimal place

    return scores_1_to_5 # Return scores on 1-5 scale

# --- Function to generate summary from scores ---
def generate_personality_summary_from_scores(ocean_scores: Dict[str, float]) -> str:
    """ Generates a brief summary based on the highest and lowest 1-5 scale OCEAN scores. """
    if not ocean_scores: return "Personality profile could not be determined."
    
    # Sort traits by absolute deviation from the midpoint (3)
    sorted_traits = sorted(ocean_scores.items(), key=lambda item: abs(item[1] - 3.0), reverse=True) 
    
    primary_trait, primary_score = sorted_traits[0]
    
    summary_parts = []
    
    trait_desc_high = { "Openness": "openness to experience", "Conscientiousness": "conscientiousness", "Extraversion": "extraversion", "Agreeableness": "agreeableness", "Neuroticism": "sensitivity/neuroticism" }
    trait_desc_low = { "Openness": "a more practical focus", "Conscientiousness": "spontaneity", "Extraversion": "introversion", "Agreeableness": "independence", "Neuroticism": "emotional stability" }

    # --- UPDATED: Use 1-5 scale thresholds ---
    # High score > 3.8 (was 0.4 on [-1,1])
    # Low score < 2.2 (was -0.4 on [-1,1]) - Adjusted slightly
    # Neutral range is roughly 2.2 to 3.8
    if primary_score > 3.8:
        summary_parts.append(f"indicates notable {trait_desc_high.get(primary_trait, primary_trait)}")
    elif primary_score < 2.2:
         summary_parts.append(f"suggests {trait_desc_low.get(primary_trait, primary_trait)}")
    else: 
         summary_parts.append("suggests a relatively balanced profile")
              
    if len(sorted_traits) > 1:
        secondary_trait, secondary_score = sorted_traits[1]
        # Check if secondary trait is also strong
        if (secondary_score > 3.8 or secondary_score < 2.2) and primary_trait != secondary_trait: 
             desc = trait_desc_high if secondary_score > 3.8 else trait_desc_low
             # Check if they are in the same 'direction' (both high or both low)
             primary_is_strong = primary_score > 3.8 or primary_score < 2.2
             primary_is_high = primary_score > 3.0
             secondary_is_high = secondary_score > 3.0
             
             prefix = "along with"
             if primary_is_strong and (primary_is_high != secondary_is_high):
                  prefix = "but also hints of" # Contrasting
             
             summary_parts.append(f"{prefix} {desc.get(secondary_trait, secondary_trait)}")


    return "The personality profile " + ", ".join(summary_parts) + "."

# --- Function to generate detailed indicators text ---
def interpret_personality_indicators(traits: TraitResult) -> str:
    """ Generates a descriptive string of personality indicators based on traits. """
    indicators = []
    total_chars = traits.diagnostics.total_chars if traits.diagnostics else 0
    total_loops = traits.total_loops
    loops_per_100 = (total_loops / total_chars) * 100 if total_chars > 0 else 0
    if loops_per_100 > 25: indicators.append("Large loops may indicate creativity and expressiveness.")
    elif loops_per_100 < 12: indicators.append("Small/absent loops may suggest discipline or practicality.")
    else: indicators.append("Moderate loop size suggests balance.")
    diacritic_acc = traits.diacritics_i_j_detection_rate > 0.85
    t_cross_acc = traits.t_crossing_detection_rate > 0.85
    if diacritic_acc and t_cross_acc: indicators.append("Accurate details (dots/crosses) suggest precision and organization.")
    elif diacritic_acc or t_cross_acc: indicators.append("Good attention to detail (dots/crosses).")
    else: indicators.append("Less consistent details might indicate impulsiveness.")
    if traits.retouching_index > 0.25: indicators.append("Frequent retouching might indicate self-doubt.")
    elif traits.retouching_index < 0.08: indicators.append("Lack of retouching can suggest confidence.")
    else: indicators.append("Occasional retouching may point towards caution.")
    spacing_irregular = traits.spacing_mean_word > 0 and traits.spacing_std_word > (traits.spacing_mean_word * 0.4)
    if spacing_irregular: indicators.append("Irregular word spacing might correlate with inconsistency.")
    elif traits.spacing_mean_word > 25: indicators.append("Wide word spacing can suggest independence.")
    elif traits.spacing_mean_word < 12: indicators.append("Narrow word spacing may indicate sociability.")
    else: indicators.append("Average word spacing suggests balance.")
    if traits.pen_lifts_per_word > 1.5: indicators.append("Frequent pen lifts could suggest an analytical approach.")
    elif traits.pen_lifts_per_word < 0.8: indicators.append("Connected writing often suggests fluidity.")
    else: indicators.append("Moderate pen lifts suggest balance.")
    if traits.tremor_index > 0.15: indicators.append("Pronounced tremors might indicate tension.")
    elif traits.tremor_index < 0.06: indicators.append("Smooth lines can suggest calmness.")
    else: indicators.append("Slight tremor may indicate sensitivity.")
    if traits.slant_degrees > 10: indicators.append("A rightward slant is often associated with extraversion.")
    elif traits.slant_degrees < -5: indicators.append("A leftward slant may suggest introversion.")
    else: indicators.append("A vertical slant often indicates independence.")
    if not indicators: return "Balanced mix of indicators observed."
    else: return "- " + "\n- ".join(indicators)


# --- API Endpoint ---
@app.post("/analyze", response_model=TraitResult)
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image_hash = hashlib.sha256(content).hexdigest()

    # --- UPDATED CACHE CHECK ---
    db = sqlite3.connect(CACHE_DB_PATH)
    cursor = db.cursor()
    try:
        cursor.execute("SELECT result_json FROM analysis_cache WHERE image_hash = ?", (image_hash,))
        row = cursor.fetchone()
        if row:
            print(f"Returning cached result for image hash: {image_hash[:10]}...")
            cached_data = json.loads(row[0])
            db.close()
            try:
                 # Ensure all fields expected by Pydantic model are present
                 # (This handles if the model changed since caching)
                 cached_data.setdefault('slant_degrees', 0.0) # Ensure CV slant field exists
                 if 'diagnostics' not in cached_data: cached_data['diagnostics'] = {}
                 cached_data['diagnostics'].setdefault('personality_summary', None)
                 cached_data['diagnostics'].setdefault('personality_indicators', None)
                 for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
                     cached_data['diagnostics'].setdefault(trait, 3.0) # Default to neutral 3.0
                 # Ensure new count fields exist
                 cached_data.setdefault('total_correct_diacritics', None)
                 cached_data.setdefault('total_correct_t_crossings', None)
                
                 return TraitResult(**cached_data)
            except Exception as e_cache_val:
                print(f"Cache validation error (model mismatch?), re-analyzing: {e_cache_val}")
                # If validation fails, proceed to re-analyze
                pass
    except Exception as e_db_read:
        print(f"Database read error: {e_db_read}")
    finally:
        if db: db.close() # Ensure db is closed if read fails
            
    print(f"New image detected. Performing analysis for hash: {image_hash[:10]}...")
    
    bgr_image = read_image(content)
    if bgr_image is None: raise HTTPException(status_code=400, detail="Invalid image file.")

    bgr_image_resized = resize_image_if_needed(bgr_image)
    if bgr_image_resized is None: raise HTTPException(status_code=400, detail="Image resize failed.")

    try:
        ocr_char_count, ocr_raw_results = run_ocr_and_count(bgr_image_resized)
        print(f"DEBUG: OCR function returned char count: {ocr_char_count}") 
    except Exception as e:
         print(f"Error during OCR: {e}"); traceback.print_exc() 
         raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # --- Image Processing for Slant ---
    try:
        gray_image = cv2.cvtColor(bgr_image_resized, cv2.COLOR_BGR2GRAY)
        binary_image = binarize_image(gray_image) # Ink is 1, background is 0
        skeleton_image = skeletonize_image(binary_image)
        cv_slant_degrees = calculate_cv_slant(skeleton_image)
    except Exception as e_img_proc:
        print(f"Error during CV processing or slant calc: {e_img_proc}")
        traceback.print_exc()
        cv_slant_degrees = 0.0 # Default slant if CV fails
        
    success, buffer = cv2.imencode('.png', bgr_image_resized)
    if not success: raise HTTPException(status_code=500, detail="Image encode failed.")
    image_bytes = buffer.tobytes()

    gemini_result = await analyze_with_gemini(image_bytes)

    # --- Combine OCR count, CV slant, and Gemini results ---
    final_result_dict = gemini_result # Start with Gemini results
    final_result_dict['slant_degrees'] = cv_slant_degrees # Overwrite/add CV slant result
    
    if 'diagnostics' not in final_result_dict or not isinstance(final_result_dict.get('diagnostics'), dict):
        final_result_dict['diagnostics'] = {} 
        
    final_result_dict['diagnostics']['total_chars'] = int(ocr_char_count) 
    print(f"DEBUG: Value assigned to final_result_dict['diagnostics']['total_chars']: {final_result_dict['diagnostics']['total_chars']}") 
    
    if 'overall_summary' not in final_result_dict['diagnostics']: final_result_dict['diagnostics']['overall_summary'] = 'Summary N/A.'
    if 'estimated_pressure' not in final_result_dict['diagnostics']: final_result_dict['diagnostics']['estimated_pressure'] = 'unknown'
    if 'baseline_consistency' not in final_result_dict['diagnostics']: final_result_dict['diagnostics']['baseline_consistency'] = 'unknown'

    print("\n" + "-"*20 + " FINAL DICT BEFORE INTERPRETATION " + "-"*20) 
    print(repr(final_result_dict)) 
    print("-"*64 + "\n")

    try:
        # --- Prepare data for interpretation (ensure all fields exist) ---
        temp_data_for_interpretation = final_result_dict.copy()
        # Add placeholders for fields needed by interpretation/validation if missing
        if 'total_loops' not in temp_data_for_interpretation: temp_data_for_interpretation['total_loops'] = 0.0
        if 'diacritics_i_j_detection_rate' not in temp_data_for_interpretation: temp_data_for_interpretation['diacritics_i_j_detection_rate'] = 0.0
        if 't_crossing_detection_rate' not in temp_data_for_interpretation: temp_data_for_interpretation['t_crossing_detection_rate'] = 0.0
        if 'total_correct_diacritics' not in temp_data_for_interpretation: temp_data_for_interpretation['total_correct_diacritics'] = None 
        if 'total_correct_t_crossings' not in temp_data_for_interpretation: temp_data_for_interpretation['total_correct_t_crossings'] = None 
        if 'spacing_mean_char' not in temp_data_for_interpretation: temp_data_for_interpretation['spacing_mean_char'] = 0.0
        if 'spacing_std_char' not in temp_data_for_interpretation: temp_data_for_interpretation['spacing_std_char'] = 0.0
        if 'spacing_mean_word' not in temp_data_for_interpretation: temp_data_for_interpretation['spacing_mean_word'] = 0.0
        if 'spacing_std_word' not in temp_data_for_interpretation: temp_data_for_interpretation['spacing_std_word'] = 0.0 
        if 'pen_lifts_per_word' not in temp_data_for_interpretation: temp_data_for_interpretation['pen_lifts_per_word'] = 0.0
        if 'retouching_index' not in temp_data_for_interpretation: temp_data_for_interpretation['retouching_index'] = 0.0
        if 'tremor_index' not in temp_data_for_interpretation: temp_data_for_interpretation['tremor_index'] = 0.0
        if 'slant_degrees' not in temp_data_for_interpretation: temp_data_for_interpretation['slant_degrees'] = 0.0 
        
        if 'diagnostics' not in temp_data_for_interpretation: temp_data_for_interpretation['diagnostics'] = {}
        if 'personality_summary' not in temp_data_for_interpretation['diagnostics']: temp_data_for_interpretation['diagnostics']['personality_summary'] = None
        if 'personality_indicators' not in temp_data_for_interpretation['diagnostics']: temp_data_for_interpretation['diagnostics']['personality_indicators'] = None
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
             if trait not in temp_data_for_interpretation['diagnostics']: temp_data_for_interpretation['diagnostics'][trait] = None
        if 'total_chars' not in temp_data_for_interpretation['diagnostics']: temp_data_for_interpretation['diagnostics']['total_chars'] = int(ocr_char_count)


        temp_validated_data = TraitResult(**temp_data_for_interpretation)

        # --- Generate OCEAN scores (1-5 scale), summary, and descriptive indicators ---
        ocean_scores_1_to_5 = interpret_traits_for_ocean_scores(temp_validated_data)
        personality_summary = generate_personality_summary_from_scores(ocean_scores_1_to_5)
        personality_indicators_text = interpret_personality_indicators(temp_validated_data) # Generate detailed text

        # Add all personality results back into the final dictionary
        final_result_dict['diagnostics']['personality_summary'] = personality_summary
        final_result_dict['diagnostics']['personality_indicators'] = personality_indicators_text
        final_result_dict['diagnostics'].update(ocean_scores_1_to_5) # Add OCEAN scores (1-5 scale)
        
        print("\n" + "-"*15 + " FINAL DICT BEFORE FINAL VALIDATION " + "-"*15)
        print(repr(final_result_dict)) 
        print("-"*64 + "\n")
        
        # Final validation with Pydantic model
        validated_result = TraitResult(**final_result_dict)
        
        print(f"DEBUG: Validated Total Chars: {validated_result.diagnostics.total_chars}")
        print(f"DEBUG: Validated Slant Degrees (CV): {validated_result.slant_degrees}") 
        
        # --- NEW: Save to persistent cache ---
        json_to_cache = validated_result.model_dump_json()
        try:
            db_save = sqlite3.connect(CACHE_DB_PATH)
            cursor_save = db_save.cursor()
            # Use INSERT OR REPLACE to update if hash already exists
            cursor_save.execute("INSERT OR REPLACE INTO analysis_cache (image_hash, result_json) VALUES (?, ?)", (image_hash, json_to_cache))
            db_save.commit()
            db_save.close()
            print(f"Saved result to persistent cache for hash: {image_hash[:10]}...")
        except Exception as e_db_write:
            print(f"WARNING: Failed to save result to cache database: {e_db_write}")
            if db_save: db_save.close() 
            
        return validated_result
        
    except ValidationError as e_val: 
        print(f"Pydantic Validation Error: {e_val}")
        print(f"Final data structure causing validation error: {repr(final_result_dict)}")
        raise HTTPException(status_code=500, detail=f"Data structure validation failed: {e_val}")
    except Exception as e: 
        print(f"Interpretation or Caching error: {e}")
        print(f"Final data structure causing error: {repr(final_result_dict)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Data processing failed after AI analysis: {e}")

# Added basic check for direct script run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

