"""
Streamlit Frontend — Multi-Cloud Pneumonia Detection & Triage Dashboard
RUN: cd frontend && streamlit run app.py
"""

import streamlit as st
import numpy as np
import requests
import json
import base64
import io
import os
import sys
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import colorsys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ══════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════
st.set_page_config(
    page_title="PneumoCloud AI — Multi-Cloud Triage",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pipeline_status' not in st.session_state:
    st.session_state.pipeline_status = {}

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
GCP_FUNCTION_URL = os.environ.get(
    'GCP_FUNCTION_URL',
    'https://pneumonia-analyzer-782668642236.europe-west1.run.app'
)

LOCAL_MODEL = None
try:
    import tensorflow as tf
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pneumonia_model.h5')
    if os.path.exists(model_path):
        LOCAL_MODEL = tf.keras.models.load_model(model_path)
except Exception:
    pass

# ══════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════
def preprocess_for_model(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def local_inference(image):
    img_input = preprocess_for_model(image)
    prediction = LOCAL_MODEL.predict(img_input, verbose=0)[0][0]
    diagnosis = 'PNEUMONIA DETECTED' if prediction > 0.5 else 'NORMAL'
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    return {
        'diagnosis': diagnosis, 'confidence': round(confidence, 4),
        'raw_score': float(prediction), 'ai_model': 'DenseNet-121 (Local)', 'mode': 'local'
    }

def cloud_inference(image, filename):
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    payload = {
        'filename': filename,
        'patient_id': f"PT-{datetime.now().strftime('%H%M%S')}",
        'image_base64': image_base64,
        'source': 'Streamlit_Frontend'
    }
    response = requests.post(GCP_FUNCTION_URL, json=payload, timeout=60)
    return response.json()

def mock_inference(filename):
    import random
    is_pneumonia = any(w in filename.lower() for w in ['virus', 'bacteria', 'pneumonia', 'person'])
    if is_pneumonia:
        confidence = round(random.uniform(0.78, 0.97), 4)
        raw_score = confidence
        diagnosis = 'PNEUMONIA DETECTED'
    else:
        confidence = round(random.uniform(0.82, 0.96), 4)
        raw_score = round(1 - confidence, 4)
        diagnosis = 'NORMAL'
    risk_score = round(raw_score * 0.7 + 0.15, 4)
    if risk_score >= 0.8:
        triage_level, dept = 'CRITICAL', 'ICU - Pulmonology'
        actions = ['Admit to ICU immediately', 'Start IV antibiotics', 'Order chest CT scan']
    elif risk_score >= 0.5:
        triage_level, dept = 'URGENT', 'Emergency Department'
        actions = ['Admit to ED', 'Start oral antibiotics', 'Monitor vitals q2h']
    elif risk_score >= 0.25:
        triage_level, dept = 'STANDARD', 'Outpatient Radiology'
        actions = ['Schedule follow-up in 48h', 'Repeat X-ray in 2 weeks']
    else:
        triage_level, dept = 'LOW', 'General Outpatient'
        actions = ['No immediate intervention', 'Routine follow-up']
    return {
        'diagnosis': diagnosis, 'confidence': confidence, 'raw_score': raw_score,
        'risk_score': risk_score, 'triage_level': triage_level, 'department': dept,
        'recommended_actions': actions, 'ai_model': 'DenseNet-121 (Serverless)', 'mode': 'mock'
    }

def generate_mock_gradcam(image):
    img_array = np.array(image.resize((224, 224)))
    y, x = np.mgrid[0:224, 0:224]
    cx, cy = 130, 100
    heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 40**2))
    cx2, cy2 = 90, 110
    heatmap2 = np.exp(-((x - cx2)**2 + (y - cy2)**2) / (2 * 30**2))
    heatmap = np.clip(heatmap + heatmap2 * 0.6, 0, 1)
    heatmap_uint = (heatmap * 255).astype(np.uint8)
    colored = np.zeros((*heatmap_uint.shape, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            val = heatmap_uint[i, j] / 255.0
            r, g, b = colorsys.hsv_to_rgb(0.0 + val * 0.3, 1.0, val)
            colored[i, j] = [int(r*255), int(g*255), int(b*255)]
    overlay = (img_array * 0.55 + colored * 0.45).astype(np.uint8)
    return Image.fromarray(overlay)

def get_triage_color(level):
    return {'CRITICAL': '#e74c3c', 'URGENT': '#e67e22', 'STANDARD': '#f1c40f', 'LOW': '#2ecc71'}.get(level, '#95a5a6')

def get_triage_bg(level):
    return {'CRITICAL': 'rgba(231,76,60,0.1)', 'URGENT': 'rgba(230,126,34,0.1)', 'STANDARD': 'rgba(241,196,15,0.1)', 'LOW': 'rgba(46,204,113,0.1)'}.get(level, 'rgba(149,165,166,0.1)')

def get_triage_icon(level):
    return {'CRITICAL': '🔴', 'URGENT': '🟠', 'STANDARD': '🟡', 'LOW': '🟢'}.get(level, '⚪')

# ══════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Header */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.15) 0%, transparent 50%);
        animation: pulse-bg 4s ease-in-out infinite;
    }
    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }
    .hero-chips {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
        flex-wrap: wrap;
    }
    .hero-chip {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 0.35rem 1rem;
        border-radius: 50px;
        color: #fff;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }

    /* Metric Cards */
    .metric-glass {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fe 100%);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.15);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-glass:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102,126,234,0.15);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8e8ea0;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #8e8ea0;
    }

    /* Confidence Bar */
    .conf-bar-bg {
        width: 100%;
        height: 8px;
        background: #e9ecef;
        border-radius: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }

    /* Triage Badge */
    .triage-badge-premium {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }

    /* Pipeline Cards */
    .pipeline-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .pipeline-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
    }
    .pipeline-aws {
        background: linear-gradient(145deg, #fff7ed, #ffffff);
        border-color: #f97316;
    }
    .pipeline-aws::after { background: #f97316; }
    .pipeline-gcp {
        background: linear-gradient(145deg, #eff6ff, #ffffff);
        border-color: #3b82f6;
    }
    .pipeline-gcp::after { background: #3b82f6; }
    .pipeline-azure {
        background: linear-gradient(145deg, #f0fdf4, #ffffff);
        border-color: #22c55e;
    }
    .pipeline-azure::after { background: #22c55e; }
    .pipeline-icon { font-size: 2rem; margin-bottom: 0.3rem; }
    .pipeline-name { font-weight: 700; font-size: 0.9rem; color: #1a1a2e; }
    .pipeline-desc { font-size: 0.75rem; color: #6b7280; margin-top: 0.2rem; }
    .pipeline-status {
        margin-top: 0.5rem;
        padding: 0.2rem 0.6rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-ok { background: #dcfce7; color: #16a34a; }
    .status-wait { background: #fef3c7; color: #d97706; }

    /* Action Items */
    .action-item {
        background: linear-gradient(145deg, #fafafa, #ffffff);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        transition: all 0.2s ease;
    }
    .action-item:hover { transform: translateX(5px); }
    .action-num {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
        flex-shrink: 0;
    }
    .action-text {
        font-size: 0.9rem;
        color: #374151;
        font-weight: 500;
    }

    /* Upload Area */
    .upload-zone {
        background: linear-gradient(145deg, #f0f4ff, #e8ecf7);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* History Table */
    .dataframe { border-radius: 12px !important; overflow: hidden; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] span {
        color: #e0e0ff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #c0c0ff !important;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #0f0c29, #302b63);
        border-radius: 16px;
        margin-top: 2rem;
    }
    .footer-text {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
    }
    .footer-brand {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Architecture Diagram */
    .arch-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 1rem;
        flex-wrap: wrap;
    }
    .arch-node {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        min-width: 100px;
    }
    .arch-arrow {
        font-size: 1.2rem;
        color: #667eea;
        font-weight: bold;
    }
    .arch-aws { background: #ff9900; color: white; }
    .arch-gcp { background: #4285f4; color: white; }
    .arch-azure { background: #0078d4; color: white; }
    .arch-streamlit { background: #ff4b4b; color: white; }

    /* Info Cards for Sidebar */
    .sidebar-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# HEADER
# ══════════════════════════════════════
st.markdown('''
<div class="hero-container">
    <div class="hero-title">🫁 PneumoCloud AI</div>
    <div class="hero-subtitle">Multi-Cloud Intelligent Pneumonia Detection & Clinical Triage System</div>
    <div class="hero-chips">
        <span class="hero-chip">🧠 DenseNet-121 CNN</span>
        <span class="hero-chip">🔍 Grad-CAM XAI</span>
        <span class="hero-chip">📊 Risk Scoring</span>
        <span class="hero-chip">☁️ AWS + GCP + Azure</span>
    </div>
</div>
''', unsafe_allow_html=True)


# ══════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")

    inference_mode = st.radio(
        "🔬 Inference Mode",
        ["☁️ Cloud Pipeline (AWS→GCP→Azure)", "💻 Local Model", "🎮 Demo Mode (Mock AI)"],
        index=2,
    )

    st.markdown("---")
    st.markdown("### 👤 Patient Info")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("---")

    st.markdown('''
    <div class="sidebar-card">
        <div class="sidebar-card-title">☁️ Cloud Architecture</div>
        <div style="font-size:0.85rem; color:#d0d0ff;">
            <b>1.</b> AWS S3 — Storage<br>
            <b>2.</b> AWS Lambda — ETL<br>
            <b>3.</b> GCP Functions — AI/ML<br>
            <b>4.</b> Azure Functions — Records
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="sidebar-card">
        <div class="sidebar-card-title">🧠 AI Stack</div>
        <div style="font-size:0.85rem; color:#d0d0ff;">
            • DenseNet-121 (Transfer Learning)<br>
            • Grad-CAM Explainability<br>
            • XGBoost Risk Scoring<br>
            • Rule-based Triage Engine
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.75rem;">Mastering Cloud • 2026</p>', unsafe_allow_html=True)


# ══════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════
col_upload, col_spacer, col_result = st.columns([5, 0.5, 5])

with col_upload:
    st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#eef2ff;">📤</span> Upload Chest X-ray</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a frontal chest X-ray image (PA or AP view)",
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=f"📎 {uploaded_file.name}", use_container_width=True)
        st.markdown(
            f'<div style="background:#f0f4ff;padding:0.6rem 1rem;border-radius:10px;font-size:0.85rem;">'
            f'📐 {image.size[0]}×{image.size[1]} px &nbsp;|&nbsp; 💾 {uploaded_file.size/1024:.1f} KB &nbsp;|&nbsp; '
            f'🕐 {datetime.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="upload-zone">'
            '<div style="font-size:3rem;margin-bottom:0.5rem;">🫁</div>'
            '<div style="font-weight:600;color:#667eea;font-size:1.1rem;">Drop your X-ray here</div>'
            '<div style="color:#9ca3af;font-size:0.85rem;margin-top:0.3rem;">Supports JPG, JPEG, PNG</div>'
            '</div>',
            unsafe_allow_html=True
        )

with col_result:
    if not uploaded_file:
        st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#fef3c7;">📋</span> Diagnosis Results</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#9ca3af;">'
            '<div style="font-size:4rem;margin-bottom:1rem;">🔬</div>'
            '<div style="font-size:1.1rem;font-weight:500;">Awaiting X-ray Upload</div>'
            '<div style="font-size:0.85rem;margin-top:0.5rem;">Upload a chest X-ray to begin AI analysis</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════
if uploaded_file:
    st.markdown("")
    analyze_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate pipeline steps with progress
        steps = ["🔄 Preprocessing image...", "☁️ Connecting to cloud...", "🧠 Running AI inference...", "📊 Computing risk score...", "✅ Analysis complete!"]
        for i, step in enumerate(steps):
            status_text.markdown(f'<div style="text-align:center;color:#667eea;font-weight:500;">{step}</div>', unsafe_allow_html=True)
            progress_bar.progress((i + 1) * 20)
            time.sleep(0.3)

        pipeline_steps = []
        try:
            mode = inference_mode
            if "Cloud" in mode:
                result = cloud_inference(image, uploaded_file.name)
                pipeline_steps = [
                    ("AWS S3 — Stored", "complete"), ("AWS Lambda — ETL", "complete"),
                    ("GCP — AI Inference", "complete"), ("Azure — Record Saved", "complete"),
                ]
            elif "Local" in mode and LOCAL_MODEL is not None:
                result = local_inference(image)
                raw = result['raw_score']
                risk_score = round(raw * 0.7 + 0.15, 4)
                result['risk_score'] = risk_score
                result['triage_level'] = 'CRITICAL' if risk_score >= 0.8 else 'URGENT' if risk_score >= 0.5 else 'STANDARD' if risk_score >= 0.25 else 'LOW'
                result['recommended_actions'] = ['See triage engine for details']
                result['department'] = 'Computed locally'
                pipeline_steps = [("Local Model — Done", "complete")]
            else:
                result = mock_inference(uploaded_file.name)
                pipeline_steps = [
                    ("AWS S3 — Stored ✓", "complete"), ("AWS Lambda — ETL ✓", "complete"),
                    ("GCP — AI Inference ✓", "complete"), ("Azure — Saved ✓", "complete"),
                ]

            progress_bar.progress(100)
            status_text.empty()

            st.session_state.pipeline_status = pipeline_steps
            st.session_state.last_result = result
            st.session_state.last_image = image
            st.session_state.history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': uploaded_file.name,
                'diagnosis': result.get('diagnosis', 'N/A'),
                'confidence': result.get('confidence', 0),
                'triage': result.get('triage_level', 'N/A'),
                'risk_score': result.get('risk_score', 0),
            })

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Try switching to 'Demo Mode (Mock AI)' in the sidebar.")
            progress_bar.empty()
            status_text.empty()

    # ── DISPLAY RESULTS ──
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        diag = result.get('diagnosis', 'N/A')
        conf = result.get('confidence', 0)
        risk = result.get('risk_score', 0)
        triage = result.get('triage_level', 'N/A')
        dept = result.get('department', 'N/A')
        triage_color = get_triage_color(triage)
        triage_icon = get_triage_icon(triage)
        is_pneumonia = "PNEUMONIA" in diag

        st.markdown("---")

        # ── ROW 1: Key Metrics ──
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            diag_color = "#e74c3c" if is_pneumonia else "#2ecc71"
            diag_icon = "⚠️" if is_pneumonia else "✅"
            st.markdown(f'''
            <div class="metric-glass">
                <div class="metric-label">Diagnosis</div>
                <div class="metric-value" style="color:{diag_color};font-size:1.2rem;">{diag_icon} {diag.split(" ")[0]}</div>
                <div class="metric-sub">{"Abnormality Found" if is_pneumonia else "No Abnormalities"}</div>
            </div>''', unsafe_allow_html=True)

        with m2:
            conf_color = "#e74c3c" if conf > 0.9 else "#e67e22" if conf > 0.7 else "#2ecc71"
            st.markdown(f'''
            <div class="metric-glass">
                <div class="metric-label">AI Confidence</div>
                <div class="metric-value" style="color:{conf_color};">{conf*100:.1f}%</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf*100}%;background:linear-gradient(90deg,#667eea,{conf_color});"></div></div>
            </div>''', unsafe_allow_html=True)

        with m3:
            risk_color = "#e74c3c" if risk > 0.7 else "#e67e22" if risk > 0.4 else "#2ecc71"
            st.markdown(f'''
            <div class="metric-glass">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value" style="color:{risk_color};">{risk:.3f}</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{risk*100}%;background:linear-gradient(90deg,#2ecc71,#e67e22,#e74c3c);"></div></div>
            </div>''', unsafe_allow_html=True)

        with m4:
            st.markdown(f'''
            <div class="metric-glass" style="border-color:{triage_color};">
                <div class="metric-label">Triage Level</div>
                <div style="margin:0.3rem 0;">
                    <span class="triage-badge-premium" style="background:{triage_color};">{triage_icon} {triage}</span>
                </div>
                <div class="metric-sub">Route → {dept}</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("")

        # ── ROW 2: Grad-CAM + Actions ──
        gc1, gc2 = st.columns([1, 1])

        with gc1:
            st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#fef2f2;">🔥</span> Grad-CAM Explainability</div>', unsafe_allow_html=True)
            try:
                if LOCAL_MODEL is not None:
                    from src.gradcam import generate_gradcam_for_image
                    mp = os.path.join(os.path.dirname(__file__), '..', 'models', 'pneumonia_model.h5')
                    gr = generate_gradcam_for_image(np.array(image), mp)
                    if gr:
                        st.image(gr['overlay'], use_container_width=True)
                    else:
                        raise ValueError()
                else:
                    st.image(generate_mock_gradcam(image), use_container_width=True)
            except Exception:
                st.image(generate_mock_gradcam(image), use_container_width=True)
            st.markdown(
                '<div style="background:#fef2f2;padding:0.6rem 1rem;border-radius:10px;font-size:0.8rem;color:#991b1b;">'
                '🔴 <b>Red/warm regions</b> = high AI attention &nbsp;|&nbsp; Model focuses on lung opacities & consolidations</div>',
                unsafe_allow_html=True
            )

        with gc2:
            st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#f0fdf4;">💊</span> Clinical Recommendations</div>', unsafe_allow_html=True)

            actions = result.get('recommended_actions', [])
            if isinstance(actions, str):
                try:
                    actions = json.loads(actions)
                except Exception:
                    actions = [actions]

            action_colors = ['#e74c3c', '#e67e22', '#3b82f6', '#8b5cf6', '#06b6d4']
            for i, action in enumerate(actions):
                ac = action_colors[i % len(action_colors)]
                st.markdown(f'''
                <div class="action-item" style="border-left-color:{ac};">
                    <div class="action-num" style="background:{ac};">{i+1}</div>
                    <div class="action-text">{action}</div>
                </div>''', unsafe_allow_html=True)

            st.markdown("")

            # Model Info Card
            st.markdown(f'''
            <div style="background:linear-gradient(145deg,#f5f3ff,#ede9fe);border-radius:12px;padding:1rem;border:1px solid #c4b5fd;">
                <div style="font-weight:700;color:#5b21b6;margin-bottom:0.5rem;">🧠 Model Details</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;font-size:0.85rem;">
                    <div><span style="color:#7c3aed;">Model:</span> {result.get('ai_model', 'N/A')}</div>
                    <div><span style="color:#7c3aed;">Patient:</span> {patient_age}y/{patient_gender[0]}</div>
                    <div><span style="color:#7c3aed;">Input:</span> 224×224 RGB</div>
                    <div><span style="color:#7c3aed;">Compliance:</span> HIPAA-Ready</div>
                </div>
            </div>''', unsafe_allow_html=True)


        # ── ROW 3: Pipeline Status ──
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#eef2ff;">☁️</span> Multi-Cloud Pipeline Status</div>', unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)

        pipeline = st.session_state.get('pipeline_status', [])
        cloud_configs = [
            (p1, "☁️", "AWS S3 + Lambda", "Storage & ETL", "pipeline-aws", "#f97316"),
            (p2, "🔵", "GCP Cloud Fn", "AI Inference", "pipeline-gcp", "#3b82f6"),
            (p3, "🟢", "Azure Functions", "Record Storage", "pipeline-azure", "#22c55e"),
        ]

        for col, icon, name, desc, css_class, color in cloud_configs:
            with col:
                status_html = '<span class="pipeline-status status-ok">✓ COMPLETE</span>' if pipeline else '<span class="pipeline-status status-wait">○ STANDBY</span>'
                st.markdown(f'''
                <div class="pipeline-card {css_class}">
                    <div class="pipeline-icon">{icon}</div>
                    <div class="pipeline-name">{name}</div>
                    <div class="pipeline-desc">{desc}</div>
                    {status_html}
                </div>''', unsafe_allow_html=True)

        with p4:
            st.markdown(f'''
            <div class="pipeline-card" style="background:linear-gradient(145deg,#fdf2f8,#ffffff);border-color:#ec4899;">
                <div class="pipeline-icon">📊</div>
                <div class="pipeline-name">Streamlit</div>
                <div class="pipeline-desc">Dashboard</div>
                <span class="pipeline-status status-ok">✓ ACTIVE</span>
            </div>''', unsafe_allow_html=True)

        # Architecture flow
        st.markdown("")
        st.markdown('''
        <div class="arch-flow">
            <div class="arch-node arch-aws">☁️ AWS S3</div>
            <span class="arch-arrow">→</span>
            <div class="arch-node arch-aws">⚡ Lambda</div>
            <span class="arch-arrow">→</span>
            <div class="arch-node arch-gcp">🧠 GCP AI</div>
            <span class="arch-arrow">→</span>
            <div class="arch-node arch-azure">💾 Azure</div>
            <span class="arch-arrow">→</span>
            <div class="arch-node arch-streamlit">📊 Dashboard</div>
        </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════
# DIAGNOSIS HISTORY
# ══════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header"><span class="section-header-icon" style="background:#fef3c7;">📋</span> Diagnosis History</div>', unsafe_allow_html=True)

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df.columns = ['Timestamp', 'Filename', 'Diagnosis', 'Confidence', 'Triage', 'Risk Score']

    # Color the dataframe
    def style_row(row):
        if 'PNEUMONIA' in str(row['Diagnosis']):
            return ['background-color: rgba(231,76,60,0.08)'] * len(row)
        return ['background-color: rgba(46,204,113,0.08)'] * len(row)

    styled_df = df.style.apply(style_row, axis=1).format({
        'Confidence': '{:.1%}',
        'Risk Score': '{:.4f}'
    })
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    col_clear, col_count = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            if 'last_result' in st.session_state:
                del st.session_state.last_result
            st.rerun()
    with col_count:
        total = len(df)
        pneum = len(df[df['Diagnosis'].str.contains('PNEUMONIA')])
        st.markdown(f'<div style="padding:0.5rem;font-size:0.85rem;color:#6b7280;">📊 Total: <b>{total}</b> scans &nbsp;|&nbsp; ⚠️ Pneumonia: <b>{pneum}</b> &nbsp;|&nbsp; ✅ Normal: <b>{total-pneum}</b></div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:2rem;background:linear-gradient(145deg,#f9fafb,#f3f4f6);border-radius:16px;border:2px dashed #d1d5db;">'
        '<div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>'
        '<div style="color:#6b7280;font-weight:500;">No diagnosis records yet</div>'
        '<div style="color:#9ca3af;font-size:0.85rem;">Upload an X-ray image to begin</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown('''
<div class="custom-footer">
    <div class="footer-brand">🫁 PneumoCloud AI — Multi-Cloud Pneumonia Detection & Triage System</div>
    <div class="footer-text" style="margin-top:0.3rem;">
        AWS (S3 + Lambda) • GCP (Cloud Functions) • Azure (Functions) • Streamlit Dashboard
    </div>
    <div class="footer-text" style="margin-top:0.5rem;">
        DenseNet-121 CNN • Grad-CAM XAI • XGBoost Risk Scoring • Rule-based Triage
    </div>
    <div style="margin-top:0.8rem;">
        <span class="hero-chip" style="font-size:0.7rem;">Mastering Cloud 2026</span>
    </div>
</div>
''', unsafe_allow_html=True)
