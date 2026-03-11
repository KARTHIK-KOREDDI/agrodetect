"""
🌾 AgroVision AI — Agriculture Image Analyzer
Main Streamlit application entry point.
"""

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import base64
import io
import os

from config import MODELS, GENERAL_TIPS, SEVERITY_COLOURS, HF_API_TOKEN
from hf_utils import (
    analyze_with_model,
    preprocess_image,
    get_top_predictions,
    format_confidence,
    image_to_base64,
    has_valid_token,
    is_local_model_loaded,
    LOCAL_PRIMARY_MODEL,
    is_valid_agricultural_image,
)
from knowledge_base import build_analysis_summary, get_severity_color, clean_label, get_general_fallback

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroVision AI — Plant Disease Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Inject CSS
# ─────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-content">
        <div class="hero-icon">🌾</div>
        <h1 class="hero-title">AgroVision AI</h1>
        <p class="hero-subtitle">
            Advanced Agricultural Image Analysis · Powered by Hugging Face Models
        </p>
        <div class="hero-tags">
            <span class="tag tag-green">🦠 Disease Detection</span>
            <span class="tag tag-blue">🔬 AI Diagnosis</span>
            <span class="tag tag-amber">💊 Treatment Plans</span>
            <span class="tag tag-purple">📊 Confidence Scores</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="sidebar-icon">⚙️</span>
        <span class="sidebar-title">Configuration</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Inference Mode Status ────────────────────────
    _model_cached = is_local_model_loaded(LOCAL_PRIMARY_MODEL)
    _live_api     = has_valid_token()

    if _model_cached:
        st.markdown("""
        <div class="token-status token-ok">
            <span>🧠</span>
            <div>
                <div style="font-weight:700">Local AI — Ready</div>
                <div style="font-size:0.72rem;opacity:0.8">Model loaded in memory</div>
            </div>
            <span class="token-dot dot-green"></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="token-status token-warn">
            <span>⏳</span>
            <div>
                <div style="font-weight:700">Local AI — First Run</div>
                <div style="font-size:0.72rem;opacity:0.8">Model downloads on first analysis</div>
            </div>
            <span class="token-dot dot-amber"></span>
        </div>
        """, unsafe_allow_html=True)

    if _live_api:
        st.markdown("""
        <div style="margin-top:0.4rem">
        <div class="token-status token-ok" style="padding:0.35rem 0.8rem">
            <span>🔑</span>
            <span style="font-size:0.78rem">HF API token: <strong>active</strong></span>
            <span class="token-dot dot-green"></span>
        </div></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:0.4rem">
        <div class="token-status token-demo" style="padding:0.35rem 0.8rem">
            <span>🎥</span>
            <span style="font-size:0.78rem">HF API: <strong>Demo Mode</strong></span>
            <span class="token-dot dot-blue"></span>
        </div></div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.3rem">
            <a href="https://huggingface.co/settings/tokens"
            target="_blank" style="color:#3b82f6">↗ Add free HF token</a>
            to enable API fallback
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Model selector
    st.markdown("### 🤖 Analysis Model")
    model_names = list(MODELS.keys())
    selected_model = st.selectbox(
        "Choose a model",
        model_names,
        index=0,
        help="Each model is optimised for different plant categories",
    )
    model_info = MODELS[selected_model]
    st.markdown(f"""
    <div class="model-card">
        <div class="model-icon">{model_info['icon']}</div>
        <div class="model-name">{selected_model}</div>
        <div class="model-desc">{model_info['description']}</div>
        <code class="model-id">{model_info['id']}</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Settings
    st.markdown("### 🎛️ Settings")
    top_k = st.slider("Top predictions to show", 3, 10, 5)
    show_raw = st.checkbox("Show raw API response", False)
    auto_scroll = st.checkbox("Auto-scroll to results", True)

    st.divider()

    # Tips
    st.markdown("### 🌿 Crop Care Tips")
    for tip in GENERAL_TIPS:
        st.markdown(f"<div class='tip-item'>{tip}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class="sidebar-footer">
        <p>🌾 AgroVision AI v1.0</p>
        <p>Built with Streamlit + Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────
upload_col, info_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📸</span>
        <span class="section-title">Upload Plant Image</span>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    uploaded_file = st.file_uploader(
        "Drop your plant/crop image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Supports JPG, PNG, WebP. For best results, upload clear close-up images of plant leaves or affected areas.",
    )

    # Camera input
    st.markdown("<p style='text-align:center;color:#6b7280;margin:8px 0'>— or —</p>", unsafe_allow_html=True)
    camera_image = st.camera_input("📷 Take a photo with your camera")

    # Choose source
    source_image = None
    if uploaded_file:
        source_image = Image.open(uploaded_file)
    elif camera_image:
        source_image = Image.open(camera_image)

    if source_image:
        # Preprocess
        processed = preprocess_image(source_image)
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(processed, use_column_width=True, caption="Plant image ready for analysis")

        # Image metadata
        w, h = processed.size
        st.markdown(f"""
        <div class="img-meta">
            <span>📐 {w} × {h} px</span>
            <span>🎨 {processed.mode}</span>
            <span>✅ Ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-placeholder">
            <div class="placeholder-icon">🌿</div>
            <p class="placeholder-text">Upload a plant or crop image to begin analysis</p>
            <p class="placeholder-hint">Supported: Tomato, Potato, Corn, Apple, and many more</p>
        </div>
        """, unsafe_allow_html=True)

with info_col:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">ℹ️</span>
        <span class="section-title">How AgroVision Works</span>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Upload Image", "Take or upload a clear photo of the affected plant leaf, stem, or fruit."),
        ("2️⃣", "Select Model", "Choose the AI model best suited for your crop type from the sidebar."),
        ("3️⃣", "AI Analysis", "Our Hugging Face models analyse the image in seconds."),
        ("4️⃣", "Get Diagnosis", "Receive the top disease predictions with confidence scores."),
        ("5️⃣", "Treatment Plan", "View detailed treatment, prevention, and organic remedies."),
    ]
    for emoji, title, desc in steps:
        st.markdown(f"""
        <div class="how-card">
            <span class="how-emoji">{emoji}</span>
            <div>
                <div class="how-title">{title}</div>
                <div class="how-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="supported-crops">
        <div class="sc-title">🌱 Supported Crops</div>
        <div class="sc-tags">
            <span class="crop-tag">🍅 Tomato</span>
            <span class="crop-tag">🥔 Potato</span>
            <span class="crop-tag">🌽 Corn/Maize</span>
            <span class="crop-tag">🍎 Apple</span>
            <span class="crop-tag">🍇 Grape</span>
            <span class="crop-tag">🍓 Strawberry</span>
            <span class="crop-tag">🌾 Wheat</span>
            <span class="crop-tag">🌿 General Crops</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Analyse Button
# ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
with col_btn_c:
    analyse_btn = st.button(
        "🔬 Analyse Plant Image",
        use_container_width=True,
        type="primary",
        disabled=(source_image is None),
    )

# ─────────────────────────────────────────────────────────
# Analysis Results
# ─────────────────────────────────────────────────────────
if analyse_btn and source_image:
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-title">Analysis Results</span>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    progress_bar = st.progress(0, text="🤖 Starting analysis…")
    status_text  = st.empty()
    _model_ready = is_local_model_loaded(LOCAL_PRIMARY_MODEL)

    with st.spinner(""):
        if _model_ready:
            progress_bar.progress(20, text="🧠 Local model ready — running inference…")
        else:
            progress_bar.progress(10, text="⬇️ First run: downloading AI model (~14 MB). Please wait…")
            time.sleep(0.5)
            progress_bar.progress(20, text="📦 Downloading model from HuggingFace Hub…")

        # ── Step 1: Image Validation ──────────────────
        status_text.info("🔍 Validating image content...")
        is_valid, top_cat = is_valid_agricultural_image(processed)
        
        if not is_valid:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Invalid Image Detected**")
            st.warning(f"The uploaded image appears to be: **{top_cat}**.")
            st.info("Please upload a clear image of a **plant leaf, fruit, or vegetable** for disease analysis.")
            if st.button("🔄 Try Another Image"):
                st.rerun()
            st.stop()

        progress_bar.progress(50, text="🔬 Image verified — running disease analysis…")
        predictions_raw = analyze_with_model(selected_model, processed)

        progress_bar.progress(85, text="🧠 Building diagnosis…")
        time.sleep(0.2)

        top_preds = get_top_predictions(predictions_raw, top_k=top_k)
        summary   = build_analysis_summary(top_preds)
        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.2)

    progress_bar.empty()
    status_text.empty()

    # ── Demo Banner (only shown when local model also failed) ──
    _used_demo = all(p.get("label", "").startswith(("Tomato","Potato","Corn","Apple"))
                     for p in predictions_raw[:1]) and not _model_ready
    if _used_demo and not is_local_model_loaded(LOCAL_PRIMARY_MODEL):
        st.markdown("""
        <div class="demo-banner">
            🎥 <strong>Demo Mode</strong> &mdash; Local model could not be loaded.
            Results below are sample predictions for demonstration.
            Make sure <code>transformers</code> and <code>torch</code> are installed:
            <code>pip install transformers torch</code>
        </div>
        """, unsafe_allow_html=True)

    # ── API Error State ───────────────────────────
    if summary.get("is_error"):
        st.markdown(f"""
        <div class="error-card">
            <div class="error-icon">⚠️</div>
            <div class="error-title">Analysis Could Not Complete</div>
            <div class="error-msg">{summary.get('error_message', 'Unknown error')}</div>
            <div class="error-hints">
                <strong>Common fixes:</strong><br>
                • Add your <code>HUGGINGFACE_API_TOKEN</code> in the <code>.env</code> file<br>
                • If the model is loading (503), wait 30 seconds and try again<br>
                • Use the <strong>Plant Disease Classifier</strong> model for best results<br>
                • Check your internet connection
            </div>
        </div>
        """, unsafe_allow_html=True)
        if show_raw:
            with st.expander("📄 Raw API Response"):
                st.json(predictions_raw)
        st.stop()

    # ── Top Metrics ──────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    disease_info  = summary.get("disease_info") or {}
    severity      = disease_info.get("severity", "Unknown")
    sev_colour    = get_severity_color(severity)
    confidence_pct = summary["confidence"] * 100

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Top Prediction</div>
            <div class="metric-value" style="font-size:1rem">{summary['top_prediction']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        conf_color = "#22c55e" if confidence_pct > 70 else "#f59e0b" if confidence_pct > 40 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Confidence</div>
            <div class="metric-value" style="color:{conf_color}">{confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Severity</div>
            <div class="metric-value" style="color:{sev_colour}">{severity}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        pathogen = disease_info.get("pathogen", "N/A")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🦠 Pathogen</div>
            <div class="metric-value" style="font-size:0.85rem;font-style:italic">{pathogen}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence Chart ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, detail_col = st.columns([1, 1], gap="large")

    all_preds = summary.get("all_predictions", [])

    with chart_col:
        st.markdown("#### 📊 Confidence Score Distribution")
        if all_preds:
            labels = [p["clean_label"] for p in all_preds]
            scores = [round(p["score"] * 100, 2) for p in all_preds]
            colors = ["#22c55e" if i == 0 else "#3b82f6" if i == 1 else "#8b5cf6"
                      for i in range(len(labels))]

            fig = go.Figure(go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{s:.1f}%" for s in scores],
                textposition="outside",
                textfont=dict(color="white", size=12),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=12),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                    title="Confidence (%)", color="#94a3b8",
                    range=[0, max(scores) * 1.2],
                ),
                yaxis=dict(
                    showgrid=False, color="#94a3b8",
                    autorange="reversed",
                ),
                margin=dict(l=20, r=60, t=20, b=40),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence scores to display.")

    with detail_col:
        st.markdown("#### 🔍 Prediction Details")
        if all_preds:
            for i, pred in enumerate(all_preds):
                rank_emoji = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
                pct = pred["score"] * 100
                bar_w = min(int(pct), 100)
                bar_color = "#22c55e" if i == 0 else "#3b82f6" if i == 1 else "#8b5cf6"
                st.markdown(f"""
                <div class="pred-row">
                    <div class="pred-rank">{rank_emoji[i]}</div>
                    <div class="pred-info">
                        <div class="pred-label">{pred['clean_label']}</div>
                        <div class="pred-bar-wrap">
                            <div class="pred-bar" style="width:{bar_w}%;background:{bar_color}"></div>
                        </div>
                    </div>
                    <div class="pred-score">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No predictions available.")

    # ── Disease Theory & Treatment ────────────────
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🩺</span>
        <span class="section-title">Disease Diagnosis & Treatment Plan</span>
    </div>
    """, unsafe_allow_html=True)

    if disease_info:
        disp_name = disease_info.get("display", summary["top_prediction"])
        # Banner
        sev_bg = sev_colour + "22"
        st.markdown(f"""
        <div class="disease-banner" style="border-left:5px solid {sev_colour};background:{sev_bg}">
            <div class="disease-banner-title">{disp_name}</div>
            <div class="disease-banner-meta">
                <span class="badge" style="background:{sev_colour}22;color:{sev_colour};border:1px solid {sev_colour}">
                    ⚠️ {severity} Severity
                </span>
                <span class="badge badge-gray">🦠 {disease_info.get('pathogen','N/A')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for each section
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔬 Symptoms & Causes",
            "💊 Chemical Treatment",
            "🌿 Organic Remedies",
            "🛡️ Prevention",
            "🌱 Fertilizer & Recovery",
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="info-box info-red">
                    <div class="info-box-title">🔴 Symptoms</div>
                """, unsafe_allow_html=True)
                st.markdown(disease_info.get("symptoms", "N/A"))
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="info-box info-amber">
                    <div class="info-box-title">🟡 Causes</div>
                """, unsafe_allow_html=True)
                st.markdown(disease_info.get("causes", "N/A"))
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("""<div class="info-box info-blue">
            <div class="info-box-title">💊 Recommended Chemical Treatments</div>""",
            unsafe_allow_html=True)
            for i, step in enumerate(disease_info.get("treatment", []), 1):
                st.markdown(f"""
                <div class="treatment-step">
                    <div class="step-num">{i}</div>
                    <div class="step-text">{step}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.warning("⚠️ Always read product labels and follow local agricultural guidelines before applying pesticides.")

        with tab3:
            st.markdown("""<div class="info-box info-green">
            <div class="info-box-title">🌿 Organic & Bio-based Remedies</div>""",
            unsafe_allow_html=True)
            for remedy in disease_info.get("organic", []):
                st.markdown(f"""
                <div class="remedy-item">
                    <span class="remedy-icon">🍃</span>
                    <span>{remedy}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.info("🌱 Organic options are safer for the environment and beneficial organisms.")

        with tab4:
            st.markdown("""<div class="info-box info-purple">
            <div class="info-box-title">🛡️ Prevention Strategies</div>""",
            unsafe_allow_html=True)
            for i, prev in enumerate(disease_info.get("prevention", []), 1):
                st.markdown(f"""
                <div class="treatment-step">
                    <div class="step-num" style="background:#8b5cf6">{i}</div>
                    <div class="step-text">{prev}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab5:
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("""<div class="info-box info-green">
                <div class="info-box-title">🌱 Fertilizer Recommendation</div>""",
                unsafe_allow_html=True)
                st.markdown(disease_info.get("fertilizer", "N/A"))
                st.markdown("</div>", unsafe_allow_html=True)
            with fc2:
                st.markdown("""<div class="info-box info-blue">
                <div class="info-box-title">⏱️ Expected Recovery Time</div>""",
                unsafe_allow_html=True)
                st.markdown(disease_info.get("recovery", "N/A"))
                st.markdown("</div>", unsafe_allow_html=True)

            # Severity gauge chart
            st.markdown("#### Severity Gauge")
            sev_map = {"None": 0, "Low": 20, "Medium": 45, "High": 70, "Critical": 95, "Unknown": 50}
            sev_val = sev_map.get(severity, 50)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sev_val,
                title={"text": "Severity Level", "font": {"color": "#e2e8f0", "size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar":  {"color": sev_colour},
                    "steps": [
                        {"range": [0, 25],  "color": "#14532d"},
                        {"range": [25, 50], "color": "#713f12"},
                        {"range": [50, 75], "color": "#7f1d1d"},
                        {"range": [75, 100],"color": "#581c87"},
                    ],
                    "threshold": {
                        "line":  {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": sev_val,
                    },
                },
                number={"suffix": "%", "font": {"color": "#e2e8f0"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        # No KB match — show general agricultural advice using fallback
        fallback = get_general_fallback()
        st.markdown(f"""
        <div class="info-box info-amber">
            <div class="info-box-title">🔍 Prediction: <em>{summary['top_prediction']}</em></div>
            <p>This label is not in the local plant-disease knowledge base (it may be a general
            ImageNet class from a non-disease model). Try switching to the
            <strong>Plant Disease Classifier</strong> model in the sidebar for
            crop-specific disease detection.</p>
        </div>
        """, unsafe_allow_html=True)

        if fallback:
            st.markdown("#### 🌿 General Agricultural Recommendations")
            fb_t1, fb_t2 = st.tabs(["💊 General Treatment", "🛡️ Prevention"])
            with fb_t1:
                st.markdown("<div class='info-box info-blue'><div class='info-box-title'>General Steps</div>", unsafe_allow_html=True)
                for i, step in enumerate(fallback.get("treatment", []), 1):
                    st.markdown(f"<div class='treatment-step'><div class='step-num'>{i}</div><div class='step-text'>{step}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with fb_t2:
                st.markdown("<div class='info-box info-purple'><div class='info-box-title'>Prevention Strategies</div>", unsafe_allow_html=True)
                for i, prev in enumerate(fallback.get("prevention", []), 1):
                    st.markdown(f"<div class='treatment-step'><div class='step-num' style='background:#8b5cf6'>{i}</div><div class='step-text'>{prev}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Raw JSON (optional) ───────────────────────
    if show_raw:
        with st.expander("📄 Raw API Response"):
            st.json(predictions_raw)

    # ── Action buttons ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    act1, act2, act3 = st.columns(3)
    with act1:
        report_lines = [
            f"# AgroVision AI Report",
            f"**Prediction:** {summary['top_prediction']}",
            f"**Confidence:** {summary['confidence']*100:.1f}%",
            f"**Severity:** {severity}",
            f"**Pathogen:** {disease_info.get('pathogen','N/A') if disease_info else 'N/A'}",
            f"",
            f"## Symptoms",
            disease_info.get("symptoms", "N/A") if disease_info else "N/A",
            f"",
            f"## Treatment",
        ] + (["- " + t for t in disease_info.get("treatment", [])] if disease_info else ["N/A"]) + [
            f"",
            f"## Prevention",
        ] + (["- " + p for p in disease_info.get("prevention", [])] if disease_info else ["N/A"])

        report_text = "\n".join(report_lines)
        st.download_button(
            "📥 Download Report",
            data=report_text,
            file_name="agrovision_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with act2:
        if st.button("🔄 Analyse Another Image", use_container_width=True):
            st.rerun()
    with act3:
        if st.button("📋 Copy Prediction", use_container_width=True):
            st.toast(f"Copied: {summary['top_prediction']} ({summary['confidence']*100:.1f}%)", icon="✅")

# ─────────────────────────────────────────────────────────
# Stats Section (always visible)
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="section-header">
    <span class="section-icon">📈</span>
    <span class="section-title">Platform Statistics</span>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
stats = [
    (s1, "🤖", "4", "AI Models"),
    (s2, "🦠", "38+", "Disease Classes"),
    (s3, "🌱", "8+", "Crop Types"),
    (s4, "⚡", "<2s", "Avg. Analysis Time"),
]
for col, icon, val, label in stats:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Model Cards Section
# ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-icon">🧠</span>
    <span class="section-title">Available AI Models</span>
</div>
""", unsafe_allow_html=True)

model_cols = st.columns(len(MODELS))
for col, (name, info) in zip(model_cols, MODELS.items()):
    with col:
        is_selected = (name == selected_model)
        border_style = "border:2px solid #22c55e" if is_selected else "border:1px solid rgba(255,255,255,0.08)"
        st.markdown(f"""
        <div class="model-showcase-card" style="{border_style}">
            <div class="msc-icon">{info['icon']}</div>
            <div class="msc-name">{name}</div>
            <div class="msc-desc">{info['description']}</div>
            <code class="msc-id">{info['id'][:35]}…</code>
            {"<div class='msc-active'>✅ Active</div>" if is_selected else ""}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <p>🌾 <strong>AgroVision AI</strong> — Agriculture Image Analyzer</p>
        <p>Powered by <strong>Hugging Face</strong> Inference API · Built with <strong>Streamlit</strong></p>
        <p style="color:#6b7280;font-size:0.8rem">⚠️ AI predictions are for guidance only. Always consult a certified agronomist for critical decisions.</p>
    </div>
</div>
""", unsafe_allow_html=True)
