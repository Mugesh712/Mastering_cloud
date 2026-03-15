"""
GCP Cloud Function — AI Inference Engine
==========================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

WHAT THIS DOES:
  Step 2 of the 3-cloud pipeline.
  Receives a chest X-ray (as base64 text) from AWS Lambda,
  runs the DenseNet-121 AI model to detect pneumonia,
  decides the patient triage level, then forwards the
  complete result to the Azure Function for storage.

  If the real model file is not present (e.g. in serverless cloud),
  it automatically falls back to a MOCK AI that uses the filename
  to simulate realistic results — perfect for demos.

HOW TO DEPLOY:
  cd gcp/
  gcloud functions deploy pneumonia-analyzer \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --region europe-west1 \
    --entry-point predict_pneumonia \
    --memory 512Mi \
    --timeout 60s
"""

import functions_framework   # GCP's library for Cloud Functions
import json
import urllib.request
import base64
import io
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# Azure Function URL — Step 3 of the pipeline.
# After AI analysis, we send the full result here.
# ─────────────────────────────────────────────────────
AZURE_URL = "https://pneumonia-receiver-mugesh1-dvahfjd8cca9gmcm.centralus-01.azurewebsites.net/api/savediagnosis"

IMG_SIZE = 224   # DenseNet-121 expects 224x224 images

# ─────────────────────────────────────────────────────
# Try to load the trained DenseNet-121 model.
# In serverless GCP environments TensorFlow is often too
# large to include, so we gracefully fall back to mock AI.
# ─────────────────────────────────────────────────────
MODEL = None
try:
    import tensorflow as tf
    MODEL = tf.keras.models.load_model('pneumonia_model.h5')
    print("[GCP] ✅ Real DenseNet-121 model loaded")
except Exception:
    print("[GCP] ⚠️  Model not found — using Mock AI mode")


# ──────────────────────────────────────────────────────
# HELPER: Preprocess the incoming image
# ──────────────────────────────────────────────────────

def preprocess_image(image_base64: str):
    """
    Decode a base64 image string into a numpy array ready for the model.

    Steps:
      1. base64 decode → raw bytes
      2. Load as PIL Image → convert to RGB
      3. Resize to 224x224
      4. Normalise pixel values from [0, 255] → [0.0, 1.0]
      5. Add batch dimension: shape (1, 224, 224, 3)

    Returns:
        img_input  — model-ready numpy array, shape (1, 224, 224, 3)
        img_array  — raw numpy array, shape (224, 224, 3) — used for Grad-CAM
    """
    image_bytes = base64.b64decode(image_base64)
    img         = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img         = img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img, dtype=np.float32) / 255.0   # normalise
    img_input   = np.expand_dims(img_array, axis=0)          # batch dim
    return img_input, img_array


# ──────────────────────────────────────────────────────
# HELPER: Real AI inference using the loaded model
# ──────────────────────────────────────────────────────

def predict_real(img_input) -> tuple[str, float, float]:
    """
    Run the DenseNet-121 model and return a diagnosis.

    The model outputs a single float:
      - Close to 1.0 = PNEUMONIA
      - Close to 0.0 = NORMAL

    Returns:
        diagnosis:   'PNEUMONIA DETECTED' or 'NORMAL'
        confidence:  0.0–1.0 (how sure the model is)
        raw_score:   raw model output (used for risk calculation)
    """
    raw_score  = float(MODEL.predict(img_input, verbose=0)[0][0])
    diagnosis  = 'PNEUMONIA DETECTED' if raw_score > 0.5 else 'NORMAL'
    confidence = raw_score if raw_score > 0.5 else (1.0 - raw_score)
    return diagnosis, round(confidence, 4), raw_score


# ──────────────────────────────────────────────────────
# HELPER: Mock AI (no model needed — uses filename)
# ──────────────────────────────────────────────────────

def predict_mock(filename: str) -> tuple[str, float, float]:
    """
    Simulate AI results based on the filename.
    Used when TensorFlow or the model file is not available.

    Filename keywords:
      'pneumonia' or 'virus'   → high-confidence pneumonia
      'bacteria'               → pneumonia with slightly lower confidence
      anything else            → NORMAL
    """
    name = filename.lower()
    if 'pneumonia' in name or 'virus' in name:
        return 'PNEUMONIA DETECTED', 0.94, 0.94
    elif 'bacteria' in name:
        return 'PNEUMONIA DETECTED', 0.87, 0.87
    else:
        return 'NORMAL', 0.92, 0.08


# ──────────────────────────────────────────────────────
# HELPER: Compute risk score + triage decision
# ──────────────────────────────────────────────────────

def compute_triage(raw_score: float, patient_id: str) -> dict:
    """
    Map the model's raw score to a clinical triage level.

    Risk score formula:
      risk = raw_score × 0.7 + 0.15
      (This ensures even 0% confidence gives a baseline risk of 0.15)

    Thresholds:
      ≥ 0.80 → CRITICAL  (ICU)
      ≥ 0.50 → URGENT    (Emergency)
      ≥ 0.25 → STANDARD  (Outpatient)
      < 0.25 → LOW       (Routine)
    """
    risk_score = round(raw_score * 0.7 + 0.15, 4)

    if risk_score >= 0.80:
        return {
            'risk_score':          risk_score,
            'triage_level':        'CRITICAL',
            'department':          'ICU — Pulmonology',
            'recommended_actions': [
                'Admit to ICU immediately',
                'Start IV antibiotics (Ceftriaxone + Azithromycin)',
                'Order chest CT scan for confirmation',
                'Notify attending pulmonologist',
            ]
        }
    elif risk_score >= 0.50:
        return {
            'risk_score':          risk_score,
            'triage_level':        'URGENT',
            'department':          'Emergency Department',
            'recommended_actions': [
                'Admit to Emergency Department',
                'Start oral antibiotics',
                'Order blood work (CBC, CRP)',
                'Monitor vitals every 2 hours',
            ]
        }
    elif risk_score >= 0.25:
        return {
            'risk_score':          risk_score,
            'triage_level':        'STANDARD',
            'department':          'Outpatient Radiology',
            'recommended_actions': [
                'Schedule follow-up within 48 hours',
                'Prescribe oral antibiotics if bacterial suspected',
                'Repeat X-ray in 2 weeks',
            ]
        }
    else:
        return {
            'risk_score':          risk_score,
            'triage_level':        'LOW',
            'department':          'General Outpatient',
            'recommended_actions': [
                'No immediate intervention required',
                'Routine follow-up at next visit',
                'Annual screening recommended',
            ]
        }


# ──────────────────────────────────────────────────────
# HELPER: Generate Clinical Summary via Servam AI
# ──────────────────────────────────────────────────────

def generate_clinical_summary(patient_id, diagnosis, confidence, risk_score, triage_level, recommended_actions):
    """
    Call the Servam AI API to write a professional clinical summary
    based strictly on the DenseNet model's outputs.
    """
    import os
    import urllib.request
    import json
    
    api_key = os.environ.get('SERVAM_API_KEY')
    api_url = os.environ.get('SERVAM_API_URL', 'https://api.servamai.com/v1/chat/completions')
    
    if not api_key:
        print("[GCP] Servam AI key not found, skipping natural language summary.")
        return "Clinical summary currently unavailable (Servam AI API key not configured)."
        
    print(f"[GCP] Generating clinical summary via Servam AI...")
    
    prompt = f"""
    You are a professional medical assistant. Write a short, single-paragraph clinical summary for an X-ray report.
    Use ONLY the data provided below. Do not add outside diagnoses. Be extremely concise.
    
    Patient ID: {patient_id}
    AI Diagnosis: {diagnosis}
    AI Confidence: {confidence:.0%}
    Calculated Risk Score: {risk_score:.2f}/1.00
    Triage Level: {triage_level}
    Recommended Actions: {', '.join(recommended_actions)}
    """
    
    payload = {
        "model": "servam-clinical-fast", # or whichever model name they use
        "messages": [
            {"role": "system", "content": "You are a professional medical scribe writing clinical notes based strictly on provided data. Do not hallucinate."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 150
    }
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {api_key}')
        
        response = urllib.request.urlopen(req, json.dumps(payload).encode('utf-8'), timeout=15)
        response_data = json.loads(response.read().decode('utf-8'))
        
        # Standard OpenAI response format
        summary = response_data['choices'][0]['message']['content'].strip()
        print("[GCP] ✅ Servam AI Summary generated successfully.")
        return summary
        
    except Exception as e:
        print(f"[GCP] ❌ Servam AI API failed: {str(e)}")
        return f"Clinical summary unavailable (AI generation failed: {str(e)})."


# ──────────────────────────────────────────────────────
# MAIN ENTRY POINT — GCP calls this on every HTTP request
# ──────────────────────────────────────────────────────

@functions_framework.http
def predict_pneumonia(request):
    """
    Main Cloud Function entry point.

    Accepts an HTTP POST with JSON body:
        filename      — name of the uploaded X-ray file
        patient_id    — patient identifier
        image_base64  — base64 encoded image (optional)
        source        — who sent the request (e.g. 'AWS_S3_Lambda')
    """
    try:
        # ── Parse incoming JSON ──────────────────────────────────────────
        body         = request.get_json(silent=True) or {}
        filename     = body.get('filename', 'unknown.jpg')
        patient_id   = body.get('patient_id', 'P-UNKNOWN')
        image_base64 = body.get('image_base64')
        source       = body.get('source', 'direct')

        print(f"[GCP] Processing: {filename} | Patient: {patient_id} | Source: {source}")

        # ── Step 1: Run AI inference ─────────────────────────────────────
        if MODEL is not None and image_base64:
            # Real AI: preprocess image then run DenseNet-121
            img_input, _            = preprocess_image(image_base64)
            diagnosis, confidence, raw_score = predict_real(img_input)
            ai_mode = 'DenseNet-121 (Real Model)'
        else:
            # Mock AI: simulate output based on filename keywords
            diagnosis, confidence, raw_score = predict_mock(filename)
            ai_mode = 'Mock AI (Demo Mode)'

        print(f"[GCP] Diagnosis: {diagnosis} | Confidence: {confidence:.2%}")

        # ── Step 2: Compute triage ───────────────────────────────────────
        triage = compute_triage(raw_score, patient_id)
        
        # ── Step 3: Generate Servam AI Summary ───────────────────────────
        ai_summary = generate_clinical_summary(
            patient_id, 
            diagnosis, 
            confidence, 
            triage['risk_score'], 
            triage['triage_level'], 
            triage['recommended_actions']
        )

        # ── Step 4: Assemble the full result ─────────────────────────────
        result = {
            'status':              'Pipeline Complete',
            'patient_id':          patient_id,
            'filename':            filename,
            'diagnosis':           diagnosis,
            'confidence':          confidence,
            'risk_score':          triage['risk_score'],
            'triage_level':        triage['triage_level'],
            'department':          triage['department'],
            'recommended_actions': triage['recommended_actions'],
            'ai_summary':          ai_summary,
            'ai_model':            ai_mode,
            'cloud_pipeline': {
                'step_1_storage':   f'AWS S3 [{source}]',
                'step_2_etl':       'AWS Lambda',
                'step_3_inference': 'GCP Cloud Functions',
                'step_4_storage':   'Azure Functions',
            }
        }

        # ── Step 5: Forward the result to Azure for storage ──────────────
        azure_payload = {
            'patient_file': filename,
            'patient_id':   patient_id,
            'diagnosis':    diagnosis,
            'conf_score':   confidence,
            'risk_score':   triage['risk_score'],
            'triage_level': triage['triage_level'],
            'department':   triage['department'],
            'ai_summary':   ai_summary,
            'actions':      json.dumps(triage['recommended_actions'])
        }

        try:
            req = urllib.request.Request(AZURE_URL)
            req.add_header('Content-Type', 'application/json')
            azure_body     = json.dumps(azure_payload).encode('utf-8')
            azure_response = urllib.request.urlopen(req, azure_body, timeout=30)
            azure_result   = azure_response.read().decode('utf-8')
            result['azure_status'] = 'SAVED'
            print(f"[GCP] Azure confirmed: {azure_result[:100]}")
        except Exception as azure_error:
            result['azure_status'] = f'FAILED: {str(azure_error)}'
            print(f"[GCP] Azure call failed: {azure_error}")

        # ── Step 5: Return the full diagnosis result to the caller ────────
        return json.dumps(result), 200, {'Content-Type': 'application/json'}

    except Exception as error:
        print(f"[GCP] ERROR: {str(error)}")
        return json.dumps({'error': str(error)}), 500, {'Content-Type': 'application/json'}
