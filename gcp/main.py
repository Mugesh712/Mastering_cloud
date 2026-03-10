"""
GCP Cloud Function — AI Inference Engine
==========================================
Multi-Cloud Pneumonia Detection & Triage System

This Cloud Function:
1. Receives preprocessed X-ray data from AWS Lambda
2. Runs DenseNet-121 CNN for pneumonia detection
3. Generates Grad-CAM heatmap
4. Computes risk score and triage decision
5. Sends complete triage record to Azure Function

DEPLOYMENT:
    GCP Console → Cloud Functions → Create
    Name: pneumonia-analyzer
    Runtime: Python 3.10
    Entry point: predict_pneumonia
    Memory: 512MB (for model loading)
    Timeout: 120 seconds
"""

import functions_framework
import json
import urllib.request
import base64
import io
import numpy as np
from PIL import Image

# ──────────────────────────────────────
# PASTE YOUR AZURE FUNCTION URL HERE
# ──────────────────────────────────────
AZURE_URL = "https://pneumonia-receiver-mugesh1-dvahfjd8cca9gmcm.centralus-01.azurewebsites.net/api/savediagnosis"

# Image preprocessing config
IMG_SIZE = 224

# Try to load the real model, fall back to mock if not available
MODEL = None
try:
    import tensorflow as tf
    MODEL = tf.keras.models.load_model('pneumonia_model.h5')
    print("[GCP] Loaded real DenseNet-121 model")
except Exception:
    print("[GCP] Real model not found — using mock AI logic")


def preprocess_image(image_base64):
    """Decode and preprocess base64 image for model input."""
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0), img_array


def predict_with_model(img_input):
    """Run inference using the trained CNN model."""
    prediction = MODEL.predict(img_input, verbose=0)[0][0]
    diagnosis = 'PNEUMONIA DETECTED' if prediction > 0.5 else 'NORMAL'
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    return diagnosis, confidence, float(prediction)


def predict_mock(filename):
    """Mock AI logic for demo/testing without model file."""
    if 'virus' in filename.lower() or 'pneumonia' in filename.lower():
        return 'PNEUMONIA DETECTED', 0.94, 0.94
    elif 'bacteria' in filename.lower():
        return 'PNEUMONIA DETECTED', 0.88, 0.88
    else:
        return 'NORMAL', 0.92, 0.08


def compute_triage(confidence, raw_score, patient_id):
    """Compute risk score and triage decision."""
    risk_score = raw_score * 0.7 + 0.15  # Simplified risk calculation

    if risk_score >= 0.8:
        triage_level = 'CRITICAL'
        actions = [
            'Admit to ICU immediately',
            'Start IV antibiotics (Ceftriaxone + Azithromycin)',
            'Order chest CT scan for confirmation',
            'Notify attending pulmonologist'
        ]
        department = 'ICU - Pulmonology'
    elif risk_score >= 0.5:
        triage_level = 'URGENT'
        actions = [
            'Admit to Emergency Department',
            'Start oral antibiotics',
            'Order blood work (CBC, CRP)',
            'Monitor vitals every 2 hours'
        ]
        department = 'Emergency Department'
    elif risk_score >= 0.25:
        triage_level = 'STANDARD'
        actions = [
            'Schedule outpatient follow-up within 48 hours',
            'Prescribe oral antibiotics if bacterial suspected',
            'Repeat X-ray in 2 weeks'
        ]
        department = 'Outpatient Radiology'
    else:
        triage_level = 'LOW'
        actions = [
            'No immediate intervention required',
            'Routine follow-up at next visit',
            'Recommend annual screening'
        ]
        department = 'General Outpatient'

    return {
        'risk_score': round(risk_score, 4),
        'triage_level': triage_level,
        'recommended_actions': actions,
        'department': department
    }


@functions_framework.http
def predict_pneumonia(request):
    """
    Main Cloud Function entry point.

    Accepts POST with JSON body:
        - filename: str
        - patient_id: str (optional)
        - image_base64: str (optional — base64 encoded image)
        - source: str (optional — 'AWS_S3_Lambda')
    """
    try:
        request_json = request.get_json(silent=True) or {}
        filename = request_json.get('filename', 'unknown')
        patient_id = request_json.get('patient_id', 'N/A')
        image_base64 = request_json.get('image_base64', None)
        source = request_json.get('source', 'direct')

        print(f"[GCP] Processing: {filename} (patient: {patient_id}, source: {source})")

        # ── AI Inference ──
        if MODEL is not None and image_base64:
            # REAL AI: Use trained model
            img_input, img_array = preprocess_image(image_base64)
            diagnosis, confidence, raw_score = predict_with_model(img_input)
            ai_mode = 'DenseNet-121 (Real Model)'
        else:
            # MOCK AI: Filename-based logic
            diagnosis, confidence, raw_score = predict_mock(filename)
            ai_mode = 'Mock AI (Demo Mode)'

        print(f"[GCP] Diagnosis: {diagnosis} (confidence: {confidence:.4f})")

        # ── Triage Decision ──
        triage = compute_triage(confidence, raw_score, patient_id)

        # ── Build complete result ──
        result = {
            'status': 'Workflow Complete',
            'patient_id': patient_id,
            'filename': filename,
            'diagnosis': diagnosis,
            'confidence': round(confidence, 4),
            'risk_score': triage['risk_score'],
            'triage_level': triage['triage_level'],
            'department': triage['department'],
            'recommended_actions': triage['recommended_actions'],
            'ai_model': ai_mode,
            'cloud_pipeline': {
                'storage': f'AWS S3 ({source})',
                'etl': 'AWS Lambda',
                'ai_inference': 'GCP Cloud Functions',
                'record_storage': 'Azure SQL/Table Storage'
            }
        }

        # ── Send to Azure ──
        azure_payload = {
            'patient_file': filename,
            'patient_id': patient_id,
            'diagnosis': diagnosis,
            'conf_score': round(confidence, 4),
            'risk_score': triage['risk_score'],
            'triage_level': triage['triage_level'],
            'department': triage['department'],
            'actions': json.dumps(triage['recommended_actions'])
        }

        try:
            req = urllib.request.Request(AZURE_URL)
            req.add_header('Content-Type', 'application/json')
            azure_data = json.dumps(azure_payload).encode('utf-8')
            azure_response = urllib.request.urlopen(req, azure_data, timeout=30)
            azure_result = azure_response.read().decode('utf-8')
            result['azure_status'] = 'SAVED'
            print(f"[GCP] Azure response: {azure_result}")
        except Exception as e:
            result['azure_status'] = f'FAILED: {str(e)}'
            print(f"[GCP] Azure call failed: {e}")

        return json.dumps(result), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"[GCP] Error: {str(e)}")
        error_response = json.dumps({'error': str(e)})
        return error_response, 500, {'Content-Type': 'application/json'}
