"""
Knowledge base lookup utilities for the Agriculture Image Analyzer.
Maps model predictions to disease information and treatment recommendations.
"""

from config import DISEASE_KNOWLEDGE, SEVERITY_COLOURS
from hf_utils import parse_plant_disease_label


# ──────────────────────────────────────────────
# Core Lookup
# ──────────────────────────────────────────────
def get_disease_info(crop_key: str, label_key: str) -> dict | None:
    """
    Retrieve disease information from the knowledge base.
    Returns None if no match is found (not the general fallback).
    """
    # 1. Exact crop + exact label
    crop_db = DISEASE_KNOWLEDGE.get(crop_key, {})
    if label_key in crop_db:
        return crop_db[label_key]

    # 2. Partial match within the same crop
    label_lower = label_key.lower()
    for key, info in crop_db.items():
        if key.lower() in label_lower or label_lower in key.lower():
            return info

    # 3. Search all other crops
    for crop, other_db in DISEASE_KNOWLEDGE.items():
        if crop in ("general",):
            continue
        for key, info in other_db.items():
            if key.lower() in label_lower or label_lower in key.lower():
                return info

    # 4. Healthy label detection
    if "healthy" in label_lower:
        # Return a generic healthy entry
        return {
            "display":    "Healthy Plant",
            "severity":   "None",
            "pathogen":   "N/A",
            "symptoms":   "No visible disease symptoms detected.",
            "causes":     "N/A — plant appears healthy.",
            "treatment":  ["No treatment required. Maintain current care."],
            "prevention": [
                "Continue regular monitoring every 5–7 days.",
                "Maintain consistent watering and fertilisation.",
                "Implement Integrated Pest Management (IPM) proactively.",
            ],
            "organic":    ["Compost tea as foliar feed", "Neem oil as preventive spray"],
            "fertilizer": "Balanced NPK fertiliser; conduct a soil test annually.",
            "recovery":   "N/A — plant is healthy!",
        }

    # 5. No match found — return None so the UI can show the right message
    return None


def get_general_fallback() -> dict:
    """Return the general knowledge-base fallback entry."""
    return DISEASE_KNOWLEDGE.get("general", {}).get("default", {})


def get_severity_color(severity: str) -> str:
    """Return the hex color code for a given severity level."""
    return SEVERITY_COLOURS.get(severity, "#6b7280")


# ──────────────────────────────────────────────
# Label Cleaning
# ──────────────────────────────────────────────
def clean_label(raw_label: str) -> str:
    """Convert raw model label to a human-readable string."""
    label = raw_label.strip()
    if "___" in label:
        parts = label.split("___")
        crop = parts[0].replace("_", " ").title()
        rest = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
        return f"{crop} — {rest}" if rest else crop
    return label.replace("_", " ").title()


# ──────────────────────────────────────────────
# Analysis Summary Builder
# ──────────────────────────────────────────────
def build_analysis_summary(predictions: list[dict]) -> dict:
    """
    Build a structured analysis summary from a list of predictions.

    Returns a dict with:
      - top_prediction, raw_label, confidence
      - disease_info (or None)
      - is_error (bool)
      - all_predictions list
    """
    if not predictions:
        return {
            "top_prediction":  "No prediction returned",
            "raw_label":       "",
            "confidence":      0.0,
            "disease_info":    None,
            "is_error":        True,
            "error_message":   "The model returned no predictions.",
            "all_predictions": [],
        }

    top = predictions[0]
    raw_label = top.get("label", "unknown")
    score     = top.get("score", 0.0)

    # Detect API errors (score=0.0 + error keyword in label)
    is_error = score == 0.0 and any(kw in raw_label.lower() for kw in [
        "error", "timeout", "connection", "loading", "invalid", "token",
        "failed", "denied", "limit", "not found", "unexpected", "http"
    ])

    if is_error:
        return {
            "top_prediction":  "Analysis failed",
            "raw_label":       raw_label,
            "confidence":      0.0,
            "disease_info":    None,
            "is_error":        True,
            "error_message":   raw_label,
            "all_predictions": [],
        }

    # Parse and look up disease info
    crop_key, label_key = parse_plant_disease_label(raw_label)
    disease_info = get_disease_info(crop_key, label_key)

    return {
        "top_prediction":  clean_label(raw_label),
        "raw_label":       raw_label,
        "confidence":      score,
        "crop_key":        crop_key,
        "label_key":       label_key,
        "disease_info":    disease_info,
        "is_error":        False,
        "error_message":   "",
        "all_predictions": [
            {
                "label":       p.get("label", ""),
                "clean_label": clean_label(p.get("label", "")),
                "score":       p.get("score", 0.0),
            }
            for p in predictions
            if p.get("score", 0.0) > 0.0
        ],
    }
