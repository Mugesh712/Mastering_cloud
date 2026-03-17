"""
GCP Cloud Function — AI Inference Engine (TFLite)
==================================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

Uses a TFLite-converted DenseNet-121 model for real pneumonia detection.
The model file (pneumonia_model.tflite) is bundled with the deployment.
"""

import functions_framework
import json
import urllib.request
import base64
import io
import os
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# Azure Function URL
# ─────────────────────────────────────────────────────
AZURE_URL = "https://pneumonia-receiver-mugesh1.azurewebsites.net/api/savediagnosis"

IMG_SIZE = 224

# ─────────────────────────────────────────────────────
# Load TFLite Model at startup (only runs once per container)
# ─────────────────────────────────────────────────────
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None

try:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_model.tflite')
    if not os.path.exists(model_path):
        model_path = 'pneumonia_model.tflite'
    INTERPRETER = Interpreter(model_path=model_path)
    INTERPRETER.allocate_tensors()
    INPUT_DETAILS = INTERPRETER.get_input_details()
    OUTPUT_DETAILS = INTERPRETER.get_output_details()
    print(f"[GCP] ✅ TFLite DenseNet-121 model loaded ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
except Exception as e:
    print(f"[GCP] ⚠️ TFLite model not found: {e} — using Mock AI mode")


def preprocess_image(image_base64: str):
    """Decode base64 image and prepare for model input."""
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input


def predict_real(img_input) -> tuple:
    """Run real TFLite inference."""
    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], img_input)
    INTERPRETER.invoke()
    raw_score = float(INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0][0])
    diagnosis = 'PNEUMONIA DETECTED' if raw_score > 0.5 else 'NORMAL'
    confidence = raw_score if raw_score > 0.5 else (1.0 - raw_score)
    return diagnosis, round(confidence, 4), raw_score


def predict_mock(filename: str) -> tuple:
    """Fallback mock AI based on filename keywords."""
    name = filename.lower()
    if 'pneumonia' in name or 'virus' in name:
        return 'PNEUMONIA DETECTED', 0.94, 0.94
    elif 'bacteria' in name:
        return 'PNEUMONIA DETECTED', 0.87, 0.87
    else:
        return 'NORMAL', 0.92, 0.08


def compute_triage(raw_score: float) -> dict:
    """Map raw AI score to clinical triage level."""
    risk_score = round(raw_score * 0.7 + 0.15, 4)

    if risk_score >= 0.80:
        return {
            'risk_score': risk_score, 'triage_level': 'CRITICAL',
            'department': 'ICU — Pulmonology',
            'recommended_actions': [
                'Admit to ICU immediately',
                'Start IV antibiotics (Ceftriaxone + Azithromycin)',
                'Order chest CT scan for confirmation',
                'Notify attending pulmonologist',
            ]
        }
    elif risk_score >= 0.50:
        return {
            'risk_score': risk_score, 'triage_level': 'URGENT',
            'department': 'Emergency Department',
            'recommended_actions': [
                'Admit to Emergency Department',
                'Start oral antibiotics',
                'Order blood work (CBC, CRP)',
                'Monitor vitals every 2 hours',
            ]
        }
    elif risk_score >= 0.25:
        return {
            'risk_score': risk_score, 'triage_level': 'STANDARD',
            'department': 'Outpatient Radiology',
            'recommended_actions': [
                'Schedule follow-up within 48 hours',
                'Prescribe oral antibiotics if bacterial suspected',
                'Repeat X-ray in 2 weeks',
            ]
        }
    else:
        return {
            'risk_score': risk_score, 'triage_level': 'LOW',
            'department': 'General Outpatient',
            'recommended_actions': [
                'No immediate intervention required',
                'Routine follow-up at next visit',
                'Annual screening recommended',
            ]
        }


def generate_clinical_summary(patient_id, diagnosis, confidence, risk_score, triage_level, recommended_actions):
    """Generate clinical summary using Google Gemini API."""
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        return "Clinical summary unavailable (Gemini API key not configured)."

    prompt = f"""You are a professional medical assistant. Write a short, single-paragraph clinical summary for a chest X-ray report. Be extremely concise (3-4 sentences max). Use ONLY the data provided:

Patient ID: {patient_id}
AI Diagnosis: {diagnosis}
AI Confidence: {confidence:.0%}
Risk Score: {risk_score:.2f}/1.00
Triage Level: {triage_level}
Recommended Actions: {', '.join(recommended_actions)}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200}
    }

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        response = urllib.request.urlopen(req, json.dumps(payload).encode('utf-8'), timeout=5)
        data = json.loads(response.read().decode('utf-8'))
        summary = data['candidates'][0]['content']['parts'][0]['text'].strip()
        print("[GCP] ✅ Gemini AI clinical summary generated.")
        return summary
    except Exception as e:
        print(f"[GCP] ❌ Gemini API failed: {str(e)}")
        return f"Clinical summary unavailable ({str(e)})."


@functions_framework.http
def predict_pneumonia(request):
    """Main Cloud Function entry point."""
    try:
        body = request.get_json(silent=True) or {}
        filename = body.get('filename', 'unknown.jpg')
        patient_id = body.get('patient_id', 'P-UNKNOWN')
        image_base64 = body.get('image_base64')
        source = body.get('source', 'direct')

        print(f"[GCP] Processing: {filename} | Patient: {patient_id} | Source: {source}")

        # ── AI Inference ─────────────────────────────────────────
        if INTERPRETER is not None and image_base64:
            img_input = preprocess_image(image_base64)
            diagnosis, confidence, raw_score = predict_real(img_input)
            ai_mode = 'DenseNet-121 TFLite (Real Model)'
        else:
            diagnosis, confidence, raw_score = predict_mock(filename)
            ai_mode = 'Mock AI (Demo Mode)'
            if INTERPRETER is not None and not image_base64:
                ai_mode = 'Mock AI (no image sent)'

        print(f"[GCP] Diagnosis: {diagnosis} | Confidence: {confidence:.2%} | Model: {ai_mode}")

        # ── Triage ───────────────────────────────────────────────
        triage = compute_triage(raw_score)

        # ── Servam AI Summary ────────────────────────────────────
        ai_summary = generate_clinical_summary(
            patient_id, diagnosis, confidence,
            triage['risk_score'], triage['triage_level'],
            triage['recommended_actions']
        )

        # ── Build Result ─────────────────────────────────────────
        result = {
            'status': 'Pipeline Complete',
            'patient_id': patient_id,
            'filename': filename,
            'diagnosis': diagnosis,
            'confidence': confidence,
            'risk_score': triage['risk_score'],
            'triage_level': triage['triage_level'],
            'department': triage['department'],
            'recommended_actions': triage['recommended_actions'],
            'ai_summary': ai_summary,
            'ai_model': ai_mode,
            'cloud_pipeline': {
                'step_1_storage': f'AWS S3 [{source}]',
                'step_2_etl': 'AWS Lambda',
                'step_3_inference': 'GCP Cloud Functions',
                'step_4_storage': 'Azure Functions',
            }
        }

        # ── Forward to Azure (10 sec timeout) ────────────────────
        azure_payload = {
            'patient_file': filename, 'patient_id': patient_id,
            'diagnosis': diagnosis, 'conf_score': confidence,
            'risk_score': triage['risk_score'], 'triage_level': triage['triage_level'],
            'department': triage['department'], 'ai_summary': ai_summary,
            'actions': json.dumps(triage['recommended_actions'])
        }

        try:
            req = urllib.request.Request(AZURE_URL)
            req.add_header('Content-Type', 'application/json')
            azure_response = urllib.request.urlopen(req, json.dumps(azure_payload).encode('utf-8'), timeout=10)
            result['azure_status'] = 'SAVED'
        except Exception as azure_error:
            result['azure_status'] = f'FAILED: {str(azure_error)}'

        return json.dumps(result), 200, {'Content-Type': 'application/json'}

    except Exception as error:
        print(f"[GCP] ERROR: {str(error)}")
        return json.dumps({'error': str(error)}), 500, {'Content-Type': 'application/json'}
