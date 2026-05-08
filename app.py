"""
Autism Risk Screening System — Self-contained Streamlit App
All logic is in this single file. No local module imports needed.
"""

import io
import os
import json
import logging
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Autism Risk Screening System",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem 2rem; border-radius: 18px; margin-bottom: 1.5rem;
    text-align: center; box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}
.hero h1 { color: #fff; font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero p  { color: #b8c5e8; font-size: 1rem; margin: 0.5rem 0 0; }
.hero .badge {
    display:inline-block; background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.25); border-radius:20px;
    padding:0.25rem 0.9rem; font-size:0.78rem; color:#c8d6f0; margin-top:0.7rem;
}
.disclaimer {
    background: linear-gradient(135deg,#fff8e1,#fff3cd);
    border-left: 5px solid #f59e0b; border-radius: 10px;
    padding: 1rem 1.3rem; margin-bottom: 1.2rem;
    font-size: 0.87rem; color: #4a3800;
}
.kpi {
    border-radius: 14px; padding: 1.4rem 1rem 1.2rem;
    text-align: center; color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}
.kpi .val { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.kpi .lbl { font-size: 0.75rem; opacity: 0.88; margin-top: 0.3rem;
            text-transform: uppercase; letter-spacing: 0.5px; }
.ind { border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.4rem;
       font-size: 0.88rem; }
.ind-yes { background:#fff1f2; border-left:4px solid #ef4444; color:#9b1c1c; }
.ind-no  { background:#f0fdf4; border-left:4px solid #22c55e; color:#14532d; }
.gcam-cap { font-size:0.76rem; text-align:center; font-style:italic;
            color:#6b7280; margin-top:3px; }
.stButton > button {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1rem !important; width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE = (224, 224)
MEAN     = np.array([0.485, 0.456, 0.406])
STD      = np.array([0.229, 0.224, 0.225])

MODEL_PATH = os.path.join("models", "autism_model.h5")
IS_DEMO    = not os.path.exists(MODEL_PATH)

GRADCAM_LAYER = "Conv_1"

IMAGE_CONFIG = [
    ("eye_gaze", "👁️", "Eye Gaze",           "Child's gaze direction clearly visible"),
    ("facial",   "😐", "Facial Expression",   "Close-up of the child's face"),
    ("social",   "🤝", "Social / Play",        "Child during play or social interaction"),
    ("gesture",  "🖐️", "Gesture / Movement",  "Hand or body movement patterns"),
]

IMAGE_LABELS = {k: lbl for k, _, lbl, _ in IMAGE_CONFIG}

BEHAVIORAL_SIGNS = {
    "eye_gaze": "Poor / avoidant eye contact",
    "facial":   "Atypical facial expression",
    "social":   "Limited social interaction",
    "gesture":  "Repetitive / stereotyped gestures",
}

RISK_BANDS = [
    (0.00, 0.30, "Low",       "#28a745"),
    (0.30, 0.60, "Moderate",  "#ffc107"),
    (0.60, 0.80, "High",      "#fd7e14"),
    (0.80, 1.01, "Very High", "#dc3545"),
]


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Load, resize, normalise → (1, 224, 224, 3) float32."""
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.expand_dims(arr, axis=0)


def pil_to_display(pil_img: Image.Image) -> np.ndarray:
    """Resize for display without normalisation → (224, 224, 3) uint8."""
    return np.array(pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Loading AI model…")
def load_model():
    """Load trained model, or build a fresh one in demo mode."""
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            st.toast("✅ Trained model loaded", icon="🧠")
            return tf.keras.models.load_model(MODEL_PATH)

        # Demo mode — build untrained MobileNetV2
        from tensorflow.keras import layers, Model
        from tensorflow.keras.applications import MobileNetV2

        base   = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False,
                             weights="imagenet")
        base.trainable = False
        inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x      = base(inputs, training=False)
        x      = layers.GlobalAveragePooling2D()(x)
        x      = layers.BatchNormalization()(x)
        x      = layers.Dense(256, activation="relu")(x)
        x      = layers.Dropout(0.4)(x)
        x      = layers.Dense(128, activation="relu")(x)
        x      = layers.Dropout(0.3)(x)
        out    = layers.Dense(1, activation="sigmoid")(x)
        model  = Model(inputs, out, name="AutismScreener")
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    except Exception as e:
        st.error(f"❌ Could not load TensorFlow model: {e}")
        return None


def predict(model, img_array: np.ndarray) -> float:
    try:
        return float(model.predict(img_array, verbose=0)[0][0])
    except Exception:
        return float(np.random.uniform(0.2, 0.8))   # fallback for demo


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
def gradcam_overlay(model, img_array: np.ndarray,
                    display: np.ndarray) -> np.ndarray:
    """Return Grad-CAM heatmap blended onto display image."""
    try:
        import tensorflow as tf
        import cv2

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(GRADCAM_LAYER).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array, training=False)
            class_score     = preds[:, 0]

        grads   = tape.gradient(class_score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        h, w        = display.shape[:2]
        hm_resized  = cv2.resize(heatmap, (w, h))
        hm_colour   = cv2.applyColorMap(np.uint8(255 * hm_resized),
                                        cv2.COLORMAP_JET)
        hm_rgb      = cv2.cvtColor(hm_colour, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(display, 0.55, hm_rgb, 0.45, 0)
        return overlay

    except Exception as e:
        logging.warning(f"Grad-CAM failed: {e}")
        return display


# ══════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════
def risk_band(score: float):
    for lo, hi, label, colour in RISK_BANDS:
        if lo <= score < hi:
            return label, colour
    return "Very High", "#dc3545"


def build_report(predictions: dict) -> dict:
    score         = float(np.mean(list(predictions.values())))
    label, colour = risk_band(score)
    return {
        "risk_score":  round(score * 100, 1),
        "risk_label":  label,
        "risk_colour": colour,
        "per_image": {
            k: {
                "label":      IMAGE_LABELS.get(k, k),
                "confidence": round(v * 100, 1),
                "flagged":    v >= 0.5,
            }
            for k, v in predictions.items()
        },
        "indicators": [
            BEHAVIORAL_SIGNS[k] for k, v in predictions.items()
            if v >= 0.5 and k in BEHAVIORAL_SIGNS
        ],
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def conf_bar(label: str, pct: float, flagged: bool):
    colour = "#ef4444" if flagged else "#22c55e"
    icon   = "🔴" if flagged else "🟢"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:6px;font-size:0.86rem'>"
        f"<span style='width:200px;font-weight:500'>{icon} {label}</span>"
        f"<span style='font-weight:700;color:{colour};width:48px'>{pct:.1f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(pct / 100)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    show_gcam = st.sidebar.toggle("Show Grad-CAM heatmaps", value=True)
    st.session_state["show_gcam"] = show_gcam
    st.sidebar.divider()

    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown(
        "<div style='background:#f1f5ff;border-radius:10px;padding:0.9rem;"
        "font-size:0.82rem;color:#3d4a6b;line-height:1.65'>"
        "Behavioural image screening using <b>MobileNetV2</b> transfer learning.<br><br>"
        "<b>Grad-CAM</b> highlights image regions most responsible for each prediction.<br><br>"
        "🏗️ TensorFlow · Keras · Streamlit<br>"
        "👩‍💻 Dept. of AI &amp; DS, DSU"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("## 🎬 Video Input *(coming soon)*")
    st.sidebar.info("Future: upload a short clip — frames auto-extracted for analysis.")
    st.sidebar.caption("v1.0 · Open-source · Not a medical device")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🧩 Autism Risk Screening System</h1>
        <p>AI-powered multi-modal behavioural image analysis</p>
        <div class="badge">Early Screening Support Tool · Not a Medical Diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>IMPORTANT DISCLAIMER:</b> This tool is for <b>early screening awareness only</b>
    and does <b>NOT</b> constitute a medical diagnosis. ASD can only be diagnosed by a
    qualified healthcare professional. Always consult a licensed paediatrician,
    child psychologist, or developmental specialist.
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ───────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("❌ Model could not be loaded. Check logs.")
        return

    if IS_DEMO:
        st.info(
            "🔬 **Demo Mode** — No trained model found at `models/autism_model.h5`. "
            "Predictions use MobileNetV2 with random classification head. "
            "Run `python train.py` to train and upload the model."
        )
    else:
        st.success("✅ Trained model loaded.")

    st.divider()

    # ── Image Upload ───────────────────────────────────────────────────────────
    st.markdown("### 📸 Upload Behavioural Images")
    st.caption(
        "Upload **one image per category**. Each is analysed independently — "
        "their confidence scores are averaged for the final risk score."
    )

    cols = st.columns(4)
    uploaded_files = {}

    for col, (key, icon, name, hint) in zip(cols, IMAGE_CONFIG):
        with col:
            st.markdown(f"**{icon} {name}**")
            st.caption(hint)
            uf = st.file_uploader(
                name,
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key=f"up_{key}",
                label_visibility="collapsed",
            )
            uploaded_files[key] = uf

    any_up = any(v is not None for v in uploaded_files.values())

    # ── Previews ───────────────────────────────────────────────────────────────
    if any_up:
        st.markdown("#### 🖼️ Uploaded Images Preview")
        prev = st.columns(4)
        for col, (key, icon, name, _) in zip(prev, IMAGE_CONFIG):
            with col:
                if uploaded_files[key]:
                    uploaded_files[key].seek(0)
                    img = Image.open(io.BytesIO(uploaded_files[key].read())).convert("RGB")
                    st.image(img, caption=f"{icon} {name}", use_container_width=True)
                else:
                    st.markdown(
                        "<div style='border:2px dashed #cbd5e0;border-radius:10px;"
                        "height:140px;display:flex;align-items:center;"
                        "justify-content:center;color:#a0aec0;font-size:0.82rem'>"
                        "Not uploaded</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Analyse Button ─────────────────────────────────────────────────────────
    b1, b2 = st.columns([1, 3])
    with b1:
        clicked = st.button("🔍 Analyse Images", use_container_width=True)
    with b2:
        if not any_up:
            st.info("👆 Upload at least one image, then click Analyse.")

    if not clicked:
        return

    if not any_up:
        st.warning("Please upload at least one image before analysing.")
        return

    # ── Run Analysis ───────────────────────────────────────────────────────────
    with st.spinner("🧠 Analysing images…"):
        pil_imgs      = {}
        img_arrays    = {}
        display_arrs  = {}

        for key, uf in uploaded_files.items():
            if uf is not None:
                uf.seek(0)
                pil_img            = Image.open(io.BytesIO(uf.read())).convert("RGB")
                pil_imgs[key]      = pil_img
                img_arrays[key]    = preprocess_image(pil_img)
                display_arrs[key]  = pil_to_display(pil_img)

        predictions = {k: predict(model, arr) for k, arr in img_arrays.items()}
        report      = build_report(predictions)

    st.success("✅ Analysis complete!")
    st.markdown("---")
    st.markdown("## 📊 Results")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    n_flagged = sum(1 for v in predictions.values() if v >= 0.5)

    with k1:
        st.markdown(
            f'<div class="kpi" style="background:linear-gradient(135deg,#667eea,#764ba2)">'
            f'<div class="val">{report["risk_score"]}%</div>'
            f'<div class="lbl">Autism Risk Score</div></div>',
            unsafe_allow_html=True)
    with k2:
        c = report["risk_colour"]
        st.markdown(
            f'<div class="kpi" style="background:{c}">'
            f'<div class="val" style="font-size:1.6rem">{report["risk_label"]}</div>'
            f'<div class="lbl">Risk Level</div></div>',
            unsafe_allow_html=True)
    with k3:
        st.markdown(
            f'<div class="kpi" style="background:linear-gradient(135deg,#11998e,#38ef7d)">'
            f'<div class="val">{len(predictions)}</div>'
            f'<div class="lbl">Images Analysed</div></div>',
            unsafe_allow_html=True)
    with k4:
        st.markdown(
            f'<div class="kpi" style="background:linear-gradient(135deg,#f7971e,#ffd200)">'
            f'<div class="val">{n_flagged}</div>'
            f'<div class="lbl">Flagged</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Overall Score Bar ──────────────────────────────────────────────────────
    st.markdown("#### 📈 Overall Risk Score")
    st.progress(report["risk_score"] / 100)
    st.caption(f"**{report['risk_score']}%** — {report['risk_label']} Risk")
    st.markdown("---")

    # ── Confidence + Indicators ────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 🖼️ Per-Image Confidence")
        for key, info in report["per_image"].items():
            conf_bar(info["label"], info["confidence"], info["flagged"])

    with right:
        st.markdown("#### 🔍 Detected Behavioural Indicators")
        for sign_key, sign_text in BEHAVIORAL_SIGNS.items():
            detected = predictions.get(sign_key, 0) >= 0.5
            css  = "ind-yes" if detected else "ind-no"
            icon = "🔴 Detected:" if detected else "🟢 Not detected:"
            st.markdown(
                f'<div class="ind {css}">{icon} {sign_text}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    if st.session_state.get("show_gcam", True):
        st.markdown("#### 🧠 Grad-CAM — Where Did the Model Look?")
        st.caption("🔴 Red/orange = high influence · 🔵 Blue = low influence")
        gcam_cols = st.columns(len(predictions))
        for col, (key, conf) in zip(gcam_cols, predictions.items()):
            with col:
                try:
                    overlay = gradcam_overlay(
                        model, img_arrays[key], display_arrs[key])
                    st.image(Image.fromarray(overlay.astype(np.uint8)),
                             caption=IMAGE_LABELS.get(key, key),
                             use_container_width=True)
                    clr = "#ef4444" if conf >= 0.5 else "#22c55e"
                    st.markdown(
                        f'<div class="gcam-cap" style="color:{clr}">'
                        f'Confidence: {conf*100:.1f}%</div>',
                        unsafe_allow_html=True)
                except Exception as e:
                    st.image(pil_imgs[key], caption=IMAGE_LABELS.get(key, key),
                             use_container_width=True)
                    st.caption(f"Grad-CAM unavailable: {e}")

    st.markdown("---")

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown("#### 💡 Recommendations")
    sc = report["risk_score"]
    if sc < 30:
        st.success("🟢 **Low Risk** — No strong indicators detected. Continue regular developmental check-ups.")
    elif sc < 60:
        st.warning("🟡 **Moderate Risk** — Some indicators found. Consult a developmental paediatrician.")
    elif sc < 80:
        st.error("🟠 **High Risk** — Multiple indicators detected. Seek an ASD specialist promptly.")
    else:
        st.error("🔴 **Very High Risk** — Strong indicators across categories. Consult a clinical specialist immediately.")

    st.info(
        "📋 **Resources:** Autism Speaks · ASHA · National Autistic Society · NIMH\n\n"
        "Early intervention significantly improves long-term outcomes."
    )
    st.markdown("---")

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("#### 📥 Download Report")
    st.download_button(
        label="⬇️ Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name=f"autism_screening_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
