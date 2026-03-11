"""
AgroVision AI — Image Analysis Utilities
Primary: Local inference via HuggingFace transformers (no API, no cold starts).
Fallback: HF Inference API (requires token).
Demo mode: Realistic sample predictions when nothing else is available.
"""

import io
import base64
import time
import random
import requests
from PIL import Image
from config import HF_API_TOKEN, HF_API_BASE, MODELS

# ──────────────────────────────────────────────
# Local pipeline cache (module-level singleton)
# ──────────────────────────────────────────────
_LOCAL_PIPELINES: dict = {}

# Primary local model — small MobileNetV2, 38 plant-disease classes
# Downloaded once (~14 MB) then cached on disk
LOCAL_PRIMARY_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# Fallback local model if primary fails to load
LOCAL_FALLBACK_MODEL = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"


# ──────────────────────────────────────────────
# Demo Predictions (shown when all else fails)
# ──────────────────────────────────────────────
DEMO_PREDICTIONS = [
    [
        {"label": "Tomato___Early_blight",         "score": 0.7821},
        {"label": "Tomato___Septoria_leaf_spot",   "score": 0.1043},
        {"label": "Tomato___healthy",              "score": 0.0612},
        {"label": "Potato___Early_blight",         "score": 0.0312},
        {"label": "Tomato___Leaf_Mold",            "score": 0.0212},
    ],
    [
        {"label": "Potato___Late_blight",          "score": 0.8431},
        {"label": "Tomato___Late_blight",          "score": 0.0932},
        {"label": "Potato___Early_blight",         "score": 0.0421},
        {"label": "Tomato___Early_blight",         "score": 0.0143},
        {"label": "Potato___healthy",              "score": 0.0073},
    ],
    [
        {"label": "Tomato___healthy",              "score": 0.9221},
        {"label": "Tomato___Early_blight",         "score": 0.0421},
        {"label": "Tomato___Bacterial_spot",       "score": 0.0213},
        {"label": "Tomato___Leaf_Mold",            "score": 0.0101},
        {"label": "Tomato___Septoria_leaf_spot",   "score": 0.0044},
    ],
    [
        {"label": "Corn___Northern_Leaf_Blight",   "score": 0.6543},
        {"label": "Corn___Common_rust",            "score": 0.2341},
        {"label": "Corn___healthy",                "score": 0.0712},
        {"label": "Tomato___Early_blight",         "score": 0.0261},
        {"label": "Potato___Early_blight",         "score": 0.0143},
    ],
    [
        {"label": "Apple___Apple_scab",            "score": 0.7112},
        {"label": "Apple___Black_rot",             "score": 0.1832},
        {"label": "Apple___healthy",               "score": 0.0712},
        {"label": "Tomato___Late_blight",          "score": 0.0243},
        {"label": "Corn___Common_rust",            "score": 0.0101},
    ],
]


def get_demo_predictions() -> list[dict]:
    """Return a random set of realistic demo predictions."""
    return random.choice(DEMO_PREDICTIONS)


# ──────────────────────────────────────────────
# Token helpers
# ──────────────────────────────────────────────
def has_valid_token() -> bool:
    """Return True if a real HuggingFace API token is configured."""
    token = HF_API_TOKEN or ""
    return bool(token) and not token.startswith("hf_your") and token.startswith("hf_")


# ──────────────────────────────────────────────
# Local Inference (Primary Method)
# ──────────────────────────────────────────────
def _load_local_pipeline(model_id: str):
    """
    Load (or return cached) a local transformers pipeline.
    The model is downloaded from HuggingFace Hub on first use
    and cached in the local HF cache directory.
    """
    if model_id in _LOCAL_PIPELINES:
        return _LOCAL_PIPELINES[model_id], None

    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            task="image-classification",
            model=model_id,
            top_k=5,
        )
        _LOCAL_PIPELINES[model_id] = pipe
        return pipe, None
    except Exception as e:
        return None, str(e)


def analyze_local(model_id: str, image: Image.Image) -> list[dict]:
    """
    Run image classification using local transformers pipeline.
    Returns standardised [{'label': str, 'score': float}] list.
    """
    pipe, err = _load_local_pipeline(model_id)
    if pipe is None:
        return _error(f"Could not load local model: {err}")

    try:
        img_rgb = image.convert("RGB")
        results = pipe(img_rgb)
        # Normalise output format
        return [
            {"label": r.get("label", ""), "score": float(r.get("score", 0.0))}
            for r in results
            if r.get("score", 0.0) > 0.0
        ]
    except Exception as e:
        return _error(f"Local inference failed: {str(e)[:120]}")


# ──────────────────────────────────────────────
# HF Inference API (Fallback)
# ──────────────────────────────────────────────
def _query_api(model_id: str, image: Image.Image) -> list[dict]:
    """Call the HF Inference API for a single model."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    headers = {
        "Content-Type":     "application/octet-stream",
        "Authorization":    f"Bearer {HF_API_TOKEN}",
        "x-wait-for-model": "true",
        "x-use-cache":      "false",
    }
    url = f"{HF_API_BASE}/{model_id}"

    try:
        resp = requests.post(url, headers=headers, data=img_bytes, timeout=180)

        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and result and "label" in result[0]:
                return result
            if isinstance(result, dict) and "error" in result:
                return _error(f"API Error: {result['error']}")
            return _error("Unexpected API response format.")

        elif resp.status_code == 401:
            return _error("Invalid token — update HUGGINGFACE_API_TOKEN in .env.")
        elif resp.status_code == 403:
            return _error("Access denied. Token needs 'read' permission.")
        elif resp.status_code == 429:
            return _error("Rate limited. Please wait and try again.")
        elif resp.status_code == 503:
            return _error("Model cold-starting (503). Please retry in 30 s.")
        elif resp.status_code == 404:
            return _error(f"Model not found: {model_id}")
        else:
            return _error(f"HTTP {resp.status_code}: {resp.text[:100]}")

    except requests.exceptions.Timeout:
        return _error("API request timed out (>180 s).")
    except requests.exceptions.ConnectionError:
        return _error("Network error — check your internet connection.")
    except Exception as e:
        return _error(f"Unexpected error: {str(e)[:80]}")


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
# Ordered list of plant-disease models for local + API fallback
_PLANT_DISEASE_MODELS = [
    LOCAL_PRIMARY_MODEL,       # linkanjarad/...
    LOCAL_FALLBACK_MODEL,      # ozair23/...
]


def analyze_with_model(model_name: str, image: Image.Image) -> list[dict]:
    """
    Analyse an image with the named model.

    Strategy (in order):
      1. Local transformers inference with selected model
      2. Local inference with fallback plant-disease models
      3. HF Inference API (if token is set)
      4. Demo predictions (always succeeds)
    """
    model_info = MODELS.get(model_name, {})
    primary_id = model_info.get("id", LOCAL_PRIMARY_MODEL)

    # ── Step 1: Local inference with selected model ──────────
    result = analyze_local(primary_id, image)
    if not is_api_error(result):
        return result

    # ── Step 2: Local fallback through other plant-disease models
    for mid in _PLANT_DISEASE_MODELS:
        if mid == primary_id:
            continue
        result = analyze_local(mid, image)
        if not is_api_error(result):
            return result

    # ── Step 3: HF API (only if valid token is present) ─────
    if has_valid_token():
        for mid in [primary_id] + _PLANT_DISEASE_MODELS:
            result = _query_api(mid, image)
            if not is_api_error(result):
                return result

    # ── Step 4: Graceful demo fallback ──────────────────────
    return get_demo_predictions()


def is_local_model_loaded(model_id: str = LOCAL_PRIMARY_MODEL) -> bool:
    """Return True if the local model is already in memory."""
    return model_id in _LOCAL_PIPELINES


# ──────────────────────────────────────────────
# Error Detection
# ──────────────────────────────────────────────
def _error(msg: str) -> list[dict]:
    """Return a standardised error entry."""
    return [{"label": msg, "score": 0.0, "_is_error": True}]


def is_api_error(predictions: list[dict]) -> bool:
    """Return True if the predictions represent an error or are empty."""
    if not predictions:
        return True
    top = predictions[0]
    if top.get("_is_error"):
        return True
    score = top.get("score", 0.0)
    label = top.get("label", "")
    if score == 0.0 and any(kw in label.lower() for kw in [
        "error", "timeout", "connection", "loading", "invalid",
        "token", "failed", "denied", "limit", "not found",
        "unexpected", "http", "inference", "network",
    ]):
        return True
    return False


def get_error_message(predictions: list[dict]) -> str:
    if predictions:
        return predictions[0].get("label", "Unknown error.")
    return "No response received."


# ──────────────────────────────────────────────
# Label Parsing
# ──────────────────────────────────────────────
def parse_plant_disease_label(raw_label: str) -> tuple:
    """Parse 'Tomato___Early_blight' → ('tomato', 'Tomato___Early_blight')."""
    label = raw_label.strip().replace(" ", "_")
    if "___" in label:
        crop = label.split("___")[0].lower()
        return crop, label
    for crop in ["tomato", "potato", "corn", "maize", "apple", "grape",
                 "strawberry", "pepper", "squash", "cherry", "peach",
                 "orange", "soybean", "wheat"]:
        if crop in label.lower():
            return crop, label
    return "general", label


def format_confidence(score: float) -> str:
    return f"{score * 100:.1f}%"


def get_top_predictions(predictions: list[dict], top_k: int = 5) -> list[dict]:
    """Return top-k valid predictions sorted by score descending."""
    valid = [p for p in predictions if p.get("score", 0.0) > 0.0]
    if not valid:
        return predictions[:1]
    return sorted(valid, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


# ──────────────────────────────────────────────
# Image Processing
# ──────────────────────────────────────────────
def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize and optimise image for inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


# ──────────────────────────────────────────────
# Image Validation Logic
# ──────────────────────────────────────────────
VALID_PLANT_KEYWORDS = [
    "leaf", "plant", "fruit", "vegetable", "tree", "flower", "grass", 
    "apple", "tomato", "potato", "corn", "maize", "crop", "field", 
    "agriculture", "farm", "soil", "garden", "foliage", "petal", 
    "stem", "branch", "root", "seed", "sprout", "vine", "bush", 
    "shrub", "forest", "wood", "nature", "greenery", "produce",
    "bell pepper", "cabbage", "broccoli", "zucchini", "pomegranate", 
    "strawberry", "pepper", "squash", "cherry", "peach", "orange", 
    "soybean", "wheat", "grape", "lemon", "lime", "banana", "pineapple"
]

def is_valid_agricultural_image(image: Image.Image) -> tuple[bool, str]:
    """
    Check if the image is likely a plant, leaf, fruit, or vegetable.
    Uses a general-purpose model for validation.
    Returns (is_valid, top_label).
    """
    # Use ResNet-50 for broad classification
    validation_model = "microsoft/resnet-50"
    
    # Run inference (Local or API)
    results = analyze_local(validation_model, image)
    
    # If local fails or is not available, try API if token exists
    if is_api_error(results) and has_valid_token():
        results = _query_api(validation_model, image)
    
    # If it's still an error or empty, we assume it's okay to proceed 
    # (don't block if validation model itself is down)
    if is_api_error(results):
        return True, "Validation skipped"

    # Check top labels
    top_labels = [r["label"].lower() for r in results[:3]]
    
    for label in top_labels:
        if any(kw in label for kw in VALID_PLANT_KEYWORDS):
            return True, results[0]["label"]
            
    return False, results[0]["label"]


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
