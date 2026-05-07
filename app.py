"""
app.py  ──  Autism Risk Screening System
Streamlit dashboard · Cloud-ready · Demo-mode when no model weights exist.

Deploy:  streamlit run app.py
"""

import os
import io
import json
import logging
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ── Local imports ──────────────────────────────────────────────────────────────
from preprocess import preprocess_image, pil_to_display_array
from model import load_model, predict_all_images, GRADCAM_LAYER, DEFAULT_MODEL_PATH
from utils import (
    format_report,
    generate_gradcam_overlay,
    setup_logging,
    array_to_pil,
    IMAGE_LABELS,
    TMP_BASE,
)

# ── Logging ─────────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("autism_screening.app")

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autism Risk Screening System",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.8rem 2rem 2.2rem;
    border-radius: 20px;
    margin-bottom: 1.6rem;
    text-align: center;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(102,126,234,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(118,75,162,0.15) 0%, transparent 60%);
}
.hero h1 { color: #ffffff; font-size: 2.4rem; font-weight: 800;
           margin: 0; letter-spacing: -0.5px; position: relative; }
.hero p  { color: #b8c5e8; font-size: 1rem; margin: 0.5rem 0 0;
           position: relative; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 0.25rem 0.9rem; font-size: 0.78rem; color: #c8d6f0;
    margin-top: 0.8rem; position: relative;
}

/* ── Disclaimer ── */
.disclaimer {
    background: linear-gradient(135deg, #fff8e1, #fff3cd);
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.4rem;
    font-size: 0.87rem;
    color: #4a3800;
    box-shadow: 0 2px 8px rgba(245,158,11,0.15);
}

/* ── Upload cards ── */
.upload-card {
    background: linear-gradient(145deg, #f0f4ff, #e8eeff);
    border: 2px dashed #667eea;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #764ba2; }
.upload-card .icon { font-size: 2rem; margin-bottom: 0.3rem; }
.upload-card .cat-name { font-weight: 700; font-size: 0.95rem; color: #2d3748; }
.upload-card .cat-hint { font-size: 0.78rem; color: #718096; margin-top: 0.15rem; }

/* ── KPI metric cards ── */
.kpi {
    border-radius: 14px;
    padding: 1.4rem 1rem 1.2rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18);
}
.kpi .val { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.kpi .lbl { font-size: 0.78rem; opacity: 0.88; margin-top: 0.3rem; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Confidence bars ── */
.conf-row {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 0.45rem; font-size: 0.85rem;
}
.conf-lbl { width: 200px; color: #2d3748; flex-shrink: 0; font-weight: 500; }
.conf-pct { width: 48px; font-weight: 700; text-align: right; }

/* ── Behavioural indicators ── */
.ind {
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.45rem;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ind-yes { background: #fff1f2; border-left: 4px solid #ef4444; color: #9b1c1c; }
.ind-no  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }

/* ── Grad-CAM caption ── */
.gcam-cap { font-size: 0.76rem; text-align: center; font-style: italic;
            color: #6b7280; margin-top: 3px; }

/* ── Sidebar info box ── */
.sbox { background: #f1f5ff; border-radius: 10px; padding: 0.9rem 1rem;
        font-size: 0.82rem; color: #3d4a6b; line-height: 1.65; }

/* ── Demo mode banner ── */
.demo-banner {
    background: linear-gradient(90deg, #1e3a5f, #2d5986);
    color: #90caf9; border-radius: 10px;
    padding: 0.7rem 1.1rem; font-size: 0.85rem;
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;
}

/* ── Button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.65rem 2rem !important;
    width: 100% !important; transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading AI model …")
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


IS_DEMO = not os.path.exists(DEFAULT_MODEL_PATH)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def to_pil(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.read())).convert("RGB")


def confidence_bar(label: str, pct: float, flagged: bool) -> None:
    colour = "#ef4444" if flagged else "#22c55e"
    st.markdown(
        f'<div class="conf-row">'
        f'<span class="conf-lbl">{"🔴" if flagged else "🟢"} {label}</span>'
        f'<span class="conf-pct" style="color:{colour}">{pct:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(pct / 100)


# ── Sidebar ─────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    show_gcam = st.sidebar.toggle("Show Grad-CAM heatmaps", value=True,
                                   help="Highlight regions that drove the prediction")
    threshold = st.sidebar.slider(
        "Flag threshold (per-image)",
        min_value=0.30, max_value=0.80, value=0.50, step=0.05,
        help="Images with confidence ≥ this value are flagged",
    )
    st.session_state["show_gcam"] = show_gcam
    st.session_state["threshold"] = threshold

    st.sidebar.divider()
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown(
        '<div class="sbox">'
        "Behavioural image screening using <b>MobileNetV2</b> Transfer Learning.<br><br>"
        "<b>Grad-CAM</b> highlights image regions most responsible for each prediction.<br><br>"
        "🏗️ Built with TensorFlow · Keras · Streamlit<br>"
        "👩‍💻 Dept. of AI &amp; DS, DSU"
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.markdown("## 🎬 Video Input *(coming soon)*")
    st.sidebar.info("Upload a short clip — the system will auto-extract frames for analysis.")
    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0 · Open-source · Not a medical device")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    sidebar()

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🧩 Autism Risk Screening System</h1>
        <p>AI-powered multi-modal behavioural image analysis</p>
        <div class="badge">Early Screening Support Tool · Not a Medical Diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>IMPORTANT DISCLAIMER:</b> This tool is for <b>early screening awareness only</b>
    and does <b>NOT</b> constitute a medical diagnosis of Autism Spectrum Disorder.
    ASD can only be diagnosed by a qualified healthcare professional following a
    comprehensive clinical evaluation. Always consult a licensed paediatrician,
    child psychologist, or developmental specialist.
    </div>
    """, unsafe_allow_html=True)

    # Model status
    model = get_model()
    if IS_DEMO:
        st.markdown(
            '<div class="demo-banner">🔬 <b>Demo Mode</b> — '
            'No trained model found. Predictions use random weights. '
            'Run <code>python train.py</code> to train and upload <code>models/autism_model.h5</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ Trained model loaded successfully.")

    st.divider()

    # ── Upload section ─────────────────────────────────────────────────────────
    st.markdown("### 📸 Upload Behavioural Images")
    st.caption(
        "Upload **one image per category**. All four are analysed independently — "
        "their confidence scores are averaged for the final risk score."
    )

    UPLOAD_CONFIG = [
        ("eye_gaze", "👁️", "Eye Gaze",
         "Child's gaze direction clearly visible"),
        ("facial",   "😐", "Facial Expression",
         "Close-up of the child's face / expression"),
        ("social",   "🤝", "Social / Play",
         "Child during play or social interaction"),
        ("gesture",  "🖐️", "Gesture / Movement",
         "Hands or body movement (e.g. hand-flapping)"),
    ]

    cols = st.columns(4)
    uploaded_files = {}
    for col, (key, icon, name, hint) in zip(cols, UPLOAD_CONFIG):
        with col:
            st.markdown(
                f'<div class="upload-card">'
                f'<div class="icon">{icon}</div>'
                f'<div class="cat-name">{name}</div>'
                f'<div class="cat-hint">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            uf = st.file_uploader(
                name, type=["jpg","jpeg","png","bmp","webp"],
                key=f"up_{key}", label_visibility="collapsed",
            )
            uploaded_files[key] = uf

    # Preview
    any_up = any(v is not None for v in uploaded_files.values())
    if any_up:
        st.markdown("#### 🖼️ Uploaded Images Preview")
        prev_cols = st.columns(4)
        for col, (key, icon, name, _) in zip(prev_cols, UPLOAD_CONFIG):
            with col:
                if uploaded_files[key]:
                    uploaded_files[key].seek(0)
                    st.image(to_pil(uploaded_files[key]),
                             caption=f"{icon} {name}", use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="border:2px dashed #cbd5e0;border-radius:10px;'
                        f'height:160px;display:flex;align-items:center;justify-content:center;'
                        f'color:#a0aec0;font-size:0.82rem;">Not uploaded</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Analyse button ─────────────────────────────────────────────────────────
    btn_col, hint_col = st.columns([1, 3])
    with btn_col:
        clicked = st.button("🔍 Analyse Images", type="primary", use_container_width=True)
    with hint_col:
        if not any_up:
            st.info("👆 Upload at least one image, then click Analyse.")

    # ── Analysis ───────────────────────────────────────────────────────────────
    if clicked:
        if not any_up:
            st.warning("Please upload at least one image before analysing.")
            st.stop()

        with st.spinner("🧠 Running AI analysis …"):
            image_arrays   = {}
            display_arrays = {}
            pil_imgs       = {}

            for key, uf in uploaded_files.items():
                if uf is not None:
                    uf.seek(0)
                    pil_img              = to_pil(uf)
                    pil_imgs[key]        = pil_img
                    image_arrays[key]    = preprocess_image(pil_img)
                    display_arrays[key]  = pil_to_display_array(pil_img)

            predictions = predict_all_images(model, image_arrays)
            report      = format_report(predictions)
            logger.info(f"Risk score: {report['risk_score']}% | {report['risk_label']}")

        st.success("✅ Analysis complete!")
        st.markdown("---")
        st.markdown("## 📊 Results")

        # ── KPI Row ────────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        n_flagged = sum(1 for v in predictions.values() if v >= 0.5)
        colour    = report["risk_colour"]

        with k1:
            st.markdown(
                f'<div class="kpi" style="background:linear-gradient(135deg,#667eea,#764ba2)">'
                f'<div class="val">{report["risk_score"]}%</div>'
                f'<div class="lbl">Autism Risk Score</div></div>',
                unsafe_allow_html=True)
        with k2:
            st.markdown(
                f'<div class="kpi" style="background:{colour}">'
                f'<div class="val" style="font-size:1.7rem">{report["risk_label"]}</div>'
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
                f'<div class="lbl">Images Flagged</div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Overall progress bar ───────────────────────────────────────────────
        st.markdown("#### 📈 Overall Risk Score")
        st.progress(report["risk_score"] / 100)
        st.caption(f"**{report['risk_score']}%** — {report['risk_label']} Risk")

        st.markdown("---")

        # ── Confidence + Indicators ────────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.markdown("#### 🖼️ Per-Image Confidence Scores")
            for key, info in report["per_image"].items():
                confidence_bar(info["label"], info["confidence"], info["flagged"])

        with right:
            st.markdown("#### 🔍 Detected Behavioural Indicators")
            signs = {
                "Poor / avoidant eye contact":       predictions.get("eye_gaze", 0) >= 0.5,
                "Atypical facial expression":         predictions.get("facial",   0) >= 0.5,
                "Limited social interaction":         predictions.get("social",   0) >= 0.5,
                "Repetitive / stereotyped gestures":  predictions.get("gesture",  0) >= 0.5,
            }
            for sign, detected in signs.items():
                css  = "ind-yes" if detected else "ind-no"
                icon = "🔴" if detected else "🟢"
                txt  = "Detected" if detected else "Not detected"
                st.markdown(
                    f'<div class="ind {css}">{icon} <b>{txt}:</b> {sign}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Grad-CAM ───────────────────────────────────────────────────────────
        if st.session_state.get("show_gcam", True):
            st.markdown("#### 🧠 Grad-CAM — Where Did the Model Look?")
            st.caption(
                "🔴 Red/orange = high influence on prediction  ·  "
                "🔵 Blue = low influence"
            )
            gcam_cols = st.columns(len(predictions))
            for col, (key, conf) in zip(gcam_cols, predictions.items()):
                with col:
                    try:
                        overlay = generate_gradcam_overlay(
                            model, image_arrays[key],
                            display_arrays[key], GRADCAM_LAYER)
                        st.image(array_to_pil(overlay),
                                 caption=IMAGE_LABELS.get(key, key),
                                 use_container_width=True)
                        pct    = conf * 100
                        clr    = "#ef4444" if conf >= 0.5 else "#22c55e"
                        st.markdown(
                            f'<div class="gcam-cap" style="color:{clr}">'
                            f'Confidence: {pct:.1f}%</div>',
                            unsafe_allow_html=True)
                    except Exception as e:
                        logger.warning(f"Grad-CAM error [{key}]: {e}")
                        if key in pil_imgs:
                            st.image(pil_imgs[key], use_container_width=True)
                            st.caption(f"Grad-CAM unavailable: {e}")

        st.markdown("---")

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown("#### 💡 Recommendations")
        sc = report["risk_score"]
        if sc < 30:
            st.success("🟢 **Low Risk** — No strong indicators detected. Continue regular developmental check-ups.")
        elif sc < 60:
            st.warning("🟡 **Moderate Risk** — Some indicators found. Consult a developmental paediatrician for a thorough evaluation.")
        elif sc < 80:
            st.error("🟠 **High Risk** — Multiple indicators detected. Please seek an ASD specialist assessment promptly.")
        else:
            st.error("🔴 **Very High Risk** — Strong indicators across categories. Consult a licensed clinical specialist immediately.")

        st.info(
            "📋 **Next Steps:** Early intervention significantly improves outcomes. "
            "Resources: **Autism Speaks** · **ASHA** · **National Autistic Society** · "
            "**NIMH** (National Institute of Mental Health)"
        )

        st.markdown("---")

        # ── Download Report ────────────────────────────────────────────────────
        st.markdown("#### 📥 Download Report")
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="⬇️ Download JSON Report",
            data=report_json,
            file_name=f"autism_screening_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        logger.info("Results displayed successfully.")


if __name__ == "__main__":
    main()
