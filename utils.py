"""
utils.py
--------
Utility functions: logging setup, Grad-CAM, result formatting, reporting.
Cloud-compatible: all file writes go to /tmp (writable on Streamlit Cloud).
"""

import os
import logging
import datetime
import numpy as np
import cv2
from PIL import Image

# ── Cloud-safe writable base directory ────────────────────────────────────────
# Streamlit Cloud project directory is read-only; /tmp is always writable.
TMP_BASE = "/tmp/autism_screening"


# ── Logging Setup ──────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(TMP_BASE, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"autism_screening_{date_str}.log")
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    except OSError:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("autism_screening")


# ── Risk Scoring ───────────────────────────────────────────────────────────────

INDIVIDUAL_THRESHOLD = 0.50

RISK_BANDS = [
    (0.00, 0.30, "Low",       "#28a745"),
    (0.30, 0.60, "Moderate",  "#ffc107"),
    (0.60, 0.80, "High",      "#fd7e14"),
    (0.80, 1.00, "Very High", "#dc3545"),
]

IMAGE_LABELS = {
    "eye_gaze": "Eye Gaze",
    "facial":   "Facial Expression",
    "social":   "Social / Play Interaction",
    "gesture":  "Gesture Behaviour",
}

BEHAVIORAL_SIGNS = {
    "eye_gaze": "Poor / avoidant eye contact",
    "facial":   "Atypical facial expression",
    "social":   "Limited social interaction",
    "gesture":  "Repetitive / stereotyped gestures",
}


def compute_risk_score(predictions: dict) -> float:
    if not predictions:
        return 0.0
    return float(np.mean(list(predictions.values())))


def get_risk_band(score: float) -> tuple:
    for low, high, label, colour in RISK_BANDS:
        if low <= score < high:
            return label, colour
    return "Very High", "#dc3545"


def get_behavioral_indicators(predictions: dict) -> list:
    return [
        BEHAVIORAL_SIGNS[key]
        for key, conf in predictions.items()
        if conf >= INDIVIDUAL_THRESHOLD and key in BEHAVIORAL_SIGNS
    ]


def format_report(predictions: dict) -> dict:
    score         = compute_risk_score(predictions)
    label, colour = get_risk_band(score)
    indicators    = get_behavioral_indicators(predictions)
    per_image = {
        key: {
            "label":      IMAGE_LABELS.get(key, key),
            "confidence": round(float(conf) * 100, 1),
            "flagged":    conf >= INDIVIDUAL_THRESHOLD,
        }
        for key, conf in predictions.items()
    }
    return {
        "risk_score":            round(score * 100, 1),
        "risk_label":            label,
        "risk_colour":           colour,
        "behavioral_indicators": indicators,
        "per_image":             per_image,
        "timestamp":             datetime.datetime.now().isoformat(),
    }


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    try:
        import tensorflow as tf
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads   = tape.gradient(class_channel, conv_outputs)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        logging.getLogger(__name__).warning(f"Grad-CAM failed: {e}")
        return np.zeros((7, 7), dtype=np.float32)


def overlay_gradcam(original_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    h, w          = original_img.shape[:2]
    hm_resized    = cv2.resize(heatmap, (w, h))
    hm_uint8      = np.uint8(255 * hm_resized)
    hm_colour     = cv2.applyColorMap(hm_uint8, colormap)
    hm_rgb        = cv2.cvtColor(hm_colour, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 1 - alpha, hm_rgb, alpha, 0)


def generate_gradcam_overlay(model, img_array, original_display, last_conv_layer_name):
    heatmap = make_gradcam_heatmap(model, img_array, last_conv_layer_name)
    return overlay_gradcam(original_display, heatmap)


# ── Image Helpers ──────────────────────────────────────────────────────────────

def array_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))


def pil_to_array(img):
    return np.array(img)


def save_report_json(report: dict, output_dir: str = None) -> str:
    import json
    if output_dir is None:
        output_dir = os.path.join(TMP_BASE, "reports")
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"report_{ts}.json")
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    return filepath
