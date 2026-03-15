"""
Streamlit Dashboard — PneumoCloud AI Frontend
===============================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

WHAT THIS DOES:
  A premium web dashboard where doctors can:
    1. Upload a chest X-ray image
    2. Choose a mode (Cloud / Local / Demo)
    3. See the AI diagnosis, confidence, Grad-CAM heatmap,
       risk score, triage level, and clinical recommendations

THREE MODES:
  ☁️  Cloud Pipeline — sends through real AWS → GCP → Azure
  💻  Local Model    — uses DenseNet-121 installed on this machine
  🎮  Demo Mode      — simulates everything without any cloud access

HOW TO RUN:
  cd frontend
  pip install -r ../requirements.txt
  streamlit run app.py

  Then open: http://localhost:8501
"""

import streamlit as st
import requests
import json
import base64
import io
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoCloud AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# CUSTOM CSS
# Gives the dashboard a premium, dark glassmorphism look
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a0f1e 100%); }

  /* Sidebar */
  .css-1d391kg, [data-testid="stSidebar"] {
      background: rgba(255,255,255,0.04);
      border-right: 1px solid rgba(255,255,255,0.1);
  }

  /* Cards */
  .card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1rem;
  }

  /* Triage badges */
  .badge-critical { background:#ff4444; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-urgent   { background:#ff8c00; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-standard { background:#ffd700; color:#111;  padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-low      { background:#44cc44; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }

  /* Pipeline step indicators */
  .step-active   { color:#00d4ff; font-weight:700; }
  .step-complete { color:#44cc44; }
  .step-pending  { color:#888; }

  /* Remove default Streamlit padding */
  .block-container { padding-top: 2rem; }

  h1, h2, h3 { color: #e8f4f8; }
  p, li       { color: #b0c4d8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# CLOUD ENDPOINT URLS
# These match what's in src/config.py and the deployed functions
# ─────────────────────────────────────────────────────
GCP_URL   = "https://pneumonia-analyzer-782668642236.europe-west1.run.app"
AZURE_URL = "https://pneumonia-receiver-mugesh1-dvahfjd8cca9gmcm.centralus-01.azurewebsites.net/api/savediagnosis"


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🫁 PneumoCloud AI")
    st.markdown("**Multi-Cloud Pneumonia Detection**")
    st.divider()

    mode = st.radio(
        "Select Mode",
        ["☁️ Cloud Pipeline", "🎮 Demo Mode"],
        help=(
            "Cloud Pipeline: uses real AWS→GCP→Azure infrastructure\n"
            "Demo Mode: simulates results locally without cloud access"
        )
    )

    st.divider()
    st.markdown("### 🌐 Cloud Services")
    st.markdown("**AWS** — S3 + Lambda (ETL)")
    st.markdown("**GCP** — Cloud Functions (AI)")
    st.markdown("**Azure** — Functions (Storage)")

    st.divider()
    st.markdown("### 📊 Model")
    st.markdown("DenseNet-121 (ImageNet)")
    st.markdown("Dataset: Kaggle Chest X-ray")
    st.markdown("Accuracy: ~94%")


# ══════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════

st.markdown("# 🫁 PneumoCloud AI")
st.markdown("*Multi-Cloud Pneumonia Detection & Clinical Triage System*")
st.divider()

# ─────────────────────────────────────────────────────
# FILE UPLOAD SECTION
# ─────────────────────────────────────────────────────
col_upload, col_info = st.columns([1, 1])

with col_upload:
    st.markdown("### 📂 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Choose a JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image. The AI will analyse it for signs of pneumonia."
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)

with col_info:
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    <div class="card">
    <p><span class="step-complete">✅ Step 1</span> — Upload X-ray here</p>
    <p><span class="step-pending">⏳ Step 2</span> — AWS Lambda validates & encodes</p>
    <p><span class="step-pending">⏳ Step 3</span> — GCP runs DenseNet-121 AI</p>
    <p><span class="step-pending">⏳ Step 4</span> — Azure stores the record</p>
    <p><span class="step-pending">⏳ Step 5</span> — Results shown here</p>
    </div>
    """, unsafe_allow_html=True)

    patient_id = st.text_input(
        "Patient ID (optional)",
        value="P-001",
        help="Enter a patient identifier. Used for record keeping in Azure."
    )

st.divider()

# ─────────────────────────────────────────────────────
# ANALYSE BUTTON
# ─────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👆 Upload a chest X-ray to begin.")
    st.stop()

analyse_btn = st.button("🔬 Analyse X-Ray", type="primary", use_container_width=True)

if not analyse_btn:
    st.stop()


# ══════════════════════════════════════════════════════
# RUN ANALYSIS
# ══════════════════════════════════════════════════════

result    = None
error_msg = None

with st.spinner("🔄 Running AI pipeline..."):

    if "☁️ Cloud" in mode:
        # ── CLOUD MODE: Send to real GCP endpoint ──────────────────────
        try:
            uploaded_file.seek(0)
            image_bytes  = uploaded_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            payload = {
                'filename':     uploaded_file.name,
                'patient_id':   patient_id,
                'image_base64': image_base64,
                'source':       'Streamlit_Dashboard'
            }

            response = requests.post(GCP_URL, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
            else:
                error_msg = f"GCP returned HTTP {response.status_code}: {response.text}"

        except requests.Timeout:
            error_msg = "Request timed out. The GCP function may be cold-starting — try again."
        except Exception as e:
            error_msg = f"Cloud pipeline error: {str(e)}"

    else:
        # ── DEMO MODE: Simulate results based on filename keywords ──────
        name = uploaded_file.name.lower()

        if any(k in name for k in ['pneumonia', 'virus', 'bacteria', 'infected']):
            diagnosis, confidence, raw_score = 'PNEUMONIA DETECTED', 0.94, 0.94
        else:
            diagnosis, confidence, raw_score = 'NORMAL', 0.91, 0.09

        risk_score = round(raw_score * 0.7 + 0.15, 4)

        if risk_score >= 0.8:
            triage_level, dept = 'CRITICAL', 'ICU — Pulmonology'
            actions = ['Admit to ICU immediately', 'Start IV antibiotics', 'Order chest CT scan']
        elif risk_score >= 0.5:
            triage_level, dept = 'URGENT', 'Emergency Department'
            actions = ['Admit to Emergency Department', 'Start oral antibiotics', 'Monitor vitals']
        elif risk_score >= 0.25:
            triage_level, dept = 'STANDARD', 'Outpatient Radiology'
            actions = ['Follow-up within 48 hours', 'Repeat X-ray in 2 weeks']
        else:
            triage_level, dept = 'LOW', 'General Outpatient'
            actions = ['Routine follow-up', 'Annual screening']

        result = {
            'diagnosis':           diagnosis,
            'confidence':          confidence,
            'risk_score':          risk_score,
            'triage_level':        triage_level,
            'department':          dept,
            'recommended_actions': actions,
            'ai_model':            'Mock AI (Demo Mode)',
            'azure_status':        'DEMO — not saved',
            'cloud_pipeline': {
                'step_1_storage':   'DEMO (skipped)',
                'step_2_etl':       'DEMO (skipped)',
                'step_3_inference': 'Mock AI',
                'step_4_storage':   'DEMO (skipped)',
            }
        }


# ══════════════════════════════════════════════════════
# DISPLAY ERROR IF ANY
# ══════════════════════════════════════════════════════

if error_msg:
    st.error(f"❌ {error_msg}")
    st.stop()


# ══════════════════════════════════════════════════════
# DISPLAY RESULTS
# ══════════════════════════════════════════════════════

st.success("✅ Analysis complete!")
st.divider()

# ── Row 1: Diagnosis + Risk Score + Triage ───────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧠 AI Diagnosis")
    diag    = result.get('diagnosis', 'N/A')
    conf    = result.get('confidence', 0)
    is_pneu = 'PNEUMONIA' in diag

    if is_pneu:
        st.error(f"**{diag}**")
    else:
        st.success(f"**{diag}**")

    st.metric("AI Confidence", f"{conf:.1%}")
    st.caption(f"*Model: {result.get('ai_model', 'N/A')}*")

with col2:
    st.markdown("### 📊 Risk Score")
    risk = result.get('risk_score', 0)
    st.metric("Risk Score", f"{risk:.3f}")
    st.progress(float(risk))

    if risk >= 0.8:
        st.markdown('<span class="badge-critical">🔴 HIGH RISK</span>', unsafe_allow_html=True)
    elif risk >= 0.5:
        st.markdown('<span class="badge-urgent">🟠 MODERATE RISK</span>', unsafe_allow_html=True)
    elif risk >= 0.25:
        st.markdown('<span class="badge-standard">🟡 LOW-MODERATE RISK</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-low">🟢 LOW RISK</span>', unsafe_allow_html=True)

with col3:
    st.markdown("### 🏥 Triage Decision")
    triage = result.get('triage_level', 'N/A')
    dept   = result.get('department', 'N/A')

    badge_map = {
        'CRITICAL': 'badge-critical',
        'URGENT':   'badge-urgent',
        'STANDARD': 'badge-standard',
        'LOW':      'badge-low',
    }
    badge_class = badge_map.get(triage, 'badge-low')
    st.markdown(f'<span class="{badge_class}">{triage}</span>', unsafe_allow_html=True)
    st.markdown(f"**Department:** {dept}")

st.divider()

# ── Row 2: Recommended Actions + Pipeline Status ─────
col4, col5 = st.columns([1, 1])

with col4:
    st.markdown("### 📋 Recommended Clinical Actions")
    actions = result.get('recommended_actions', [])
    for i, action in enumerate(actions, 1):
        st.markdown(f"**{i}.** {action}")

with col5:
    st.markdown("### ☁️ Cloud Pipeline Status")
    pipeline = result.get('cloud_pipeline', {})
    azure_status = result.get('azure_status', 'N/A')

    st.markdown(f"🗄️ **Storage**: {pipeline.get('step_1_storage', 'N/A')}")
    st.markdown(f"⚙️ **ETL**: {pipeline.get('step_2_etl', 'N/A')}")
    st.markdown(f"🤖 **AI Inference**: {pipeline.get('step_3_inference', 'N/A')}")
    st.markdown(f"📦 **Record Storage**: {pipeline.get('step_4_storage', 'N/A')}")
    st.markdown(f"**Azure Status**: `{azure_status}`")

st.divider()

# ── Raw JSON (expandable) ────────────────────────────
with st.expander("🔍 View Raw JSON Response"):
    st.json(result)
