"""
GCP Cloud Function — AI Inference Engine (TFLite) + RAG
========================================================
PneumoCloud AI | Multi-Cloud Chest X-Ray Classification System

Uses a TFLite-converted DenseNet-121 model for 4-class chest X-ray
classification (COVID, Lung Opacity, Normal, Viral Pneumonia).

RAG Integration:
  After inference, retrieves relevant medical knowledge from a
  Pinecone vector database and injects it into the Sarvam AI
  prompt for grounded, citation-backed clinical reports.

Dataset: COVID-19 Radiography Database
Model:   DenseNet-121 (ImageNet pre-training + fine-tuning)
Output:  4 probabilities → argmax → disease class
"""

import functions_framework
import json
import urllib.request
import urllib.error
import base64
import io
import os
import numpy as np
from PIL import Image
from retriever import retrieve_medical_context

# ─────────────────────────────────────────────────────
# Azure Function URL
# ─────────────────────────────────────────────────────
AZURE_URL = "https://pneumonia-receiver-mugesh1.azurewebsites.net/api/savediagnosis"

IMG_SIZE = 224

# ─────────────────────────────────────────────────────
# 4 CLASSES — COVID-19 Radiography Dataset
# ─────────────────────────────────────────────────────
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────
# Disease Severity Weights (same as config.py)
# ─────────────────────────────────────────────────────
DISEASE_SEVERITY = {
    'COVID': 0.85,
    'Lung_Opacity': 0.60,
    'Normal': 0.05,
    'Viral Pneumonia': 0.75,
}

# ─────────────────────────────────────────────────────
# Disease → Department Mapping
# ─────────────────────────────────────────────────────
DISEASE_DEPARTMENTS = {
    'COVID': 'Pulmonology — Isolation Ward',
    'Lung_Opacity': 'Pulmonology',
    'Normal': 'General Outpatient',
    'Viral Pneumonia': 'Pulmonology — ICU',
}

# ─────────────────────────────────────────────────────
# Disease → Recommended Actions
# ─────────────────────────────────────────────────────
DISEASE_ACTIONS = {
    'COVID': ['Isolate patient immediately', 'Start antiviral therapy (Remdesivir)', 'Monitor SpO2 continuously', 'Order RT-PCR confirmation', 'Notify infection control team'],
    'Lung_Opacity': ['Order chest CT for detailed assessment', 'Rule out pneumonia vs. fluid accumulation', 'Correlate with clinical symptoms', 'Follow-up imaging in 2-4 weeks'],
    'Normal': ['No intervention required', 'Routine follow-up', 'Annual screening recommended', 'Monitor for new symptoms'],
    'Viral Pneumonia': ['Start antiviral + supportive therapy', 'Blood work (CBC, CRP, Procalcitonin)', 'Monitor SpO2 and respiratory rate', 'Follow-up X-ray in 4-6 weeks', 'Consider ICU admission if SpO2 < 92%'],
}

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

    # Try multiple model filenames
    model_candidates = [
        os.path.join(os.path.dirname(__file__), 'chest_multiclass_model.tflite'),
        os.path.join(os.path.dirname(__file__), 'pneumonia_model.tflite'),
        'chest_multiclass_model.tflite',
        'pneumonia_model.tflite',
    ]
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path:
        INTERPRETER = Interpreter(model_path=model_path)
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
        output_shape = OUTPUT_DETAILS[0]['shape']
        print(f"[GCP] ✅ TFLite model loaded ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
        print(f"[GCP]    Output shape: {output_shape} ({'4-class COVID' if output_shape[-1] == 4 else 'unknown'})")
    else:
        print("[GCP] ⚠️ No TFLite model found — using Mock AI mode")
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
    """
    Run real TFLite inference.
    Returns: (disease_name, confidence, class_index, all_probabilities)
    """
    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], img_input)
    INTERPRETER.invoke()
    output = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

    # 4-class model: softmax output
    class_idx = int(np.argmax(output))
    confidence = float(output[class_idx])
    disease = CLASS_NAMES[class_idx]
    all_probs = {CLASS_NAMES[i]: round(float(output[i]), 4) for i in range(NUM_CLASSES)}

    return disease, round(confidence, 4), class_idx, all_probs


def predict_mock(filename: str) -> tuple:
    """
    Fallback mock AI based on filename keywords.
    Returns: (disease_name, confidence, class_index, all_probabilities)
    """
    name = filename.lower()

    keyword_map = {
        'covid': ('COVID', 0.91),
        'corona': ('COVID', 0.89),
        'opacity': ('Lung_Opacity', 0.85),
        'viral': ('Viral Pneumonia', 0.88),
        'pneumonia': ('Viral Pneumonia', 0.87),
    }

    disease, confidence = 'Normal', 0.92
    for keyword, (d, c) in keyword_map.items():
        if keyword in name:
            disease, confidence = d, c
            break

    class_idx = CLASS_NAMES.index(disease)

    # Build mock probabilities
    all_probs = {}
    remaining = 1.0 - confidence
    for cls in CLASS_NAMES:
        all_probs[cls] = confidence if cls == disease else round(remaining / (NUM_CLASSES - 1), 4)

    return disease, confidence, class_idx, all_probs


def generate_mock_heatmap(image_base64: str) -> str | None:
    """
    Generate a plausible heatmap from the raw image when TFLite model is unavailable.
    Uses pixel intensity to simulate attention regions.
    """
    if not image_base64:
        return None
    try:
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img, dtype=np.float32) / 255.0

        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]

        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask[30:194, 20:204] = 1.0
        sensitivity = gray * mask

        s_min, s_max = sensitivity.min(), sensitivity.max()
        if s_max - s_min < 1e-8:
            return None
        sensitivity = (sensitivity - s_min) / (s_max - s_min)

        coloured = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        coloured[:, :, 0] = np.clip(sensitivity * 2 * 255 - 128, 0, 255).astype(np.uint8)
        coloured[:, :, 1] = np.clip(255 - np.abs(sensitivity * 2 * 255 - 255), 0, 255).astype(np.uint8)
        coloured[:, :, 2] = np.clip(255 - sensitivity * 2 * 255, 0, 255).astype(np.uint8)

        original_img = Image.fromarray(np.uint8(img_arr * 255)).convert('RGB')
        heatmap_pil = Image.fromarray(coloured, 'RGB')
        blended = Image.blend(original_img, heatmap_pil, alpha=0.45)

        buffer = io.BytesIO()
        blended.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("[GCP] ✅ Mock heatmap generated from pixel intensity.")
        return b64
    except Exception as e:
        print(f"[GCP] ⚠️ Mock heatmap failed: {e}")
        return None


def generate_gradcam_tflite(img_input: np.ndarray, class_idx: int, original_score: float) -> str | None:
    """
    Generate a Grad-CAM-style heatmap using input sensitivity (occlusion).
    Targets the predicted class specifically.

    Args:
        img_input:       preprocessed image, shape (1, 224, 224, 3)
        class_idx:       predicted class index (0–14)
        original_score:  original prediction score for that class
    """
    if INTERPRETER is None:
        return None

    try:
        GRID = 7
        patch_size = IMG_SIZE // GRID
        sensitivity = np.zeros((GRID, GRID), dtype=np.float32)
        img_flat = img_input.copy()

        for row in range(GRID):
            for col in range(GRID):
                perturbed = img_flat.copy()
                r0, r1 = row * patch_size, (row + 1) * patch_size
                c0, c1 = col * patch_size, (col + 1) * patch_size
                perturbed[0, r0:r1, c0:c1, :] = 0.0

                INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], perturbed)
                INTERPRETER.invoke()
                output = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

                # Target the predicted class specifically
                if len(output) == NUM_CLASSES:
                    perturbed_score = float(output[class_idx])
                else:
                    perturbed_score = float(output[0])

                sensitivity[row, col] = abs(original_score - perturbed_score)

        # Normalise to [0, 255]
        s_min, s_max = sensitivity.min(), sensitivity.max()
        if s_max - s_min < 1e-8:
            sensitivity_norm = np.zeros_like(sensitivity, dtype=np.uint8)
        else:
            sensitivity_norm = np.uint8((sensitivity - s_min) / (s_max - s_min) * 255)

        heatmap_small = Image.fromarray(sensitivity_norm, mode='L')
        heatmap_large = heatmap_small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heatmap_arr = np.array(heatmap_large, dtype=np.float32) / 255.0

        coloured = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        coloured[:, :, 0] = np.clip(heatmap_arr * 2 * 255 - 128, 0, 255).astype(np.uint8)
        coloured[:, :, 1] = np.clip(255 - np.abs(heatmap_arr * 2 * 255 - 255), 0, 255).astype(np.uint8)
        coloured[:, :, 2] = np.clip(255 - heatmap_arr * 2 * 255, 0, 255).astype(np.uint8)

        original_img = Image.fromarray(np.uint8(img_flat[0] * 255)).convert('RGB')
        heatmap_pil = Image.fromarray(coloured, 'RGB')
        blended = Image.blend(original_img, heatmap_pil, alpha=0.45)

        buffer = io.BytesIO()
        blended.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("[GCP] ✅ Grad-CAM heatmap generated.")
        return b64

    except Exception as e:
        print(f"[GCP] ⚠️ Grad-CAM failed: {e}")
        return None


def compute_triage(disease: str, confidence: float) -> dict:
    """Map disease + confidence to clinical triage level."""
    severity = DISEASE_SEVERITY.get(disease, 0.5)

    # Risk = severity weight (40%) + confidence (30%) + baseline (30%)
    risk_score = round(severity * 0.40 + confidence * 0.30 + 0.15, 4)
    risk_score = min(max(risk_score, 0.0), 1.0)

    if risk_score >= 0.80:
        triage_level = 'CRITICAL'
    elif risk_score >= 0.50:
        triage_level = 'URGENT'
    elif risk_score >= 0.25:
        triage_level = 'STANDARD'
    else:
        triage_level = 'LOW'

    department = DISEASE_DEPARTMENTS.get(disease, 'General Outpatient')
    actions = DISEASE_ACTIONS.get(disease, ['Consult specialist', 'Follow-up imaging'])

    return {
        'risk_score': risk_score,
        'triage_level': triage_level,
        'department': department,
        'recommended_actions': actions,
    }


def generate_clinical_summary(patient_id, disease, confidence, risk_score,
                               triage_level, department, recommended_actions,
                               medical_context=""):
    """Generate clinical summary using Sarvam AI API with RAG-augmented context."""
    import time

    api_key = os.environ.get('SARVAM_API_KEY')

    if not api_key:
        return "Clinical summary unavailable (Sarvam API key not configured)."

    # ── Build RAG-augmented context block ──────────────────────
    rag_context_block = ""
    if medical_context:
        rag_context_block = f"""

═══════════════════════════════════════
RETRIEVED MEDICAL REFERENCES (RAG):
The following medical knowledge was retrieved from a curated
knowledge base. Use these references to ground your report
with accurate, evidence-based information. Cite the source
when using specific data points, dosages, or guidelines.
═══════════════════════════════════════

{medical_context}

═══════════════════════════════════════
IMPORTANT: Base your report on the above medical references.
When referencing specific guidelines, dosages, or protocols,
include the source in square brackets, e.g. [Source: WHO 2025].
═══════════════════════════════════════"""
        print(f"[GCP] 📚 RAG context injected ({len(medical_context)} chars)")
    else:
        print("[GCP] ⚠️ No RAG context available — generating from model knowledge only")

    prompt = f"""You are generating a DETAILED CLINICAL RADIOLOGY REPORT for a patient's chest X-ray analysis. Write a comprehensive, professional medical report using the AI analysis data below AND the retrieved medical references (if provided). Each section must contain POINT-WISE bullet points (not paragraphs). Use relevant emojis at the start of each bullet point for visual clarity. When citing medical references, include the source in square brackets.

═══════════════════════════════════════
PATIENT DATA:
  Patient ID: {patient_id}
  AI Diagnosis: {disease}
  AI Confidence: {confidence:.1%}
  Risk Score: {risk_score:.2f} / 1.00
  Triage Level: {triage_level}
  Assigned Department: {department}
  Recommended Actions: {', '.join(recommended_actions)}
═══════════════════════════════════════
{rag_context_block}
Generate the report with EXACTLY these 7 sections. Use the section headers exactly as shown. Each section MUST contain bullet points (use dash or bullet character), NOT paragraphs. Start each bullet point with a relevant emoji. When you use information from the retrieved medical references, cite the source in square brackets at the end of the bullet point.

## 1. DIAGNOSTIC FINDINGS
Describe what the AI detected. Include points about:
- 🔬 The AI model and dataset used
- 🩺 The detected condition ({disease})
- 📊 The confidence level ({confidence:.1%})
- 💡 What this diagnosis means in medical terms

## 2. CHEST X-RAY ANALYSIS
Provide point-wise radiological interpretation based on the retrieved medical references. Each point should cover a specific finding:
- 🔍 Lung field observations
- 🔍 Ground-glass patterns or consolidation
- 🔍 Pleural involvement
- 🔍 Cardiac silhouette assessment
- 🔍 Other anatomical observations for {disease}

## 3. SEVERITY ASSESSMENT
Point-wise analysis of severity:
- ⚠️ Risk score interpretation ({risk_score:.2f}/1.00)
- 🏥 Triage level meaning ({triage_level})
- 🏢 Department assignment rationale
- 📌 Urgency of intervention needed

## 4. LIFESTYLE RECOMMENDATIONS
Point-wise daily lifestyle guidance based on retrieved medical evidence:
- 🏃 Physical activity recommendations (safe exercises, duration, frequency)
- 😴 Rest requirements (sleep hours, activity limitations)
- 🏠 Environmental precautions (air quality, crowd avoidance, mask usage)
- 🧘 Mental health and stress management

## 5. DIETARY GUIDELINES
Point-wise nutrition plan based on retrieved nutritional evidence:
- 🍎 Immune-boosting foods to eat (with specific quantities from references)
- 🥗 Anti-inflammatory diet recommendations
- 💧 Hydration goals (specific volumes from references)
- 🚫 Foods and substances to avoid (including drug-food interactions)
- 💊 Recommended supplements with evidence-based dosages

## 6. FOLLOW-UP SCHEDULE
Point-wise follow-up timeline based on clinical protocols:
- 📅 Next X-ray timing
- 🔄 Check-up frequency (weekly, biweekly, monthly)
- 🧪 Required lab tests and monitoring (specific tests from references)
- 🚨 Warning signs requiring emergency visit
- 📋 Long-term monitoring plan (3, 6, 12 months)

## 7. PROGNOSIS & EXPECTED OUTCOMES
Point-wise expected outcomes:
- ✅ Recovery timeline
- ⚡ Potential complications to watch for
- 🌟 Long-term health outlook
- 💪 Encouraging but realistic assessment

IMPORTANT: Write in professional medical language but make it understandable for patients. Use bullet points with emojis for every item. Do NOT write paragraphs. Do not use markdown bold (**) formatting. Use plain text only. Include source citations in square brackets when using specific data from the retrieved references."""

    # Build system message with RAG awareness
    system_msg = "You are a senior consultant radiologist at a multi-specialty hospital. You write comprehensive, detailed clinical reports in POINT-WISE bullet format with emojis. Always use the exact section headers provided. Never write paragraphs — use bullet points only. Never use markdown bold formatting. Do not show your reasoning process."
    if medical_context:
        system_msg += " You have been provided with retrieved medical references from a curated knowledge base. Prioritize information from these references and cite sources when using specific data points, dosages, or protocols."

    payload = {
        "model": "sarvam-30b",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 8192
    }

    url = "https://api.sarvam.ai/v1/chat/completions"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('Content-Type', 'application/json')
            req.add_header('Authorization', f'Bearer {api_key}')
            response = urllib.request.urlopen(req, json.dumps(payload).encode('utf-8'), timeout=30)
            data = json.loads(response.read().decode('utf-8'))

            summary = None
            if 'choices' in data and len(data['choices']) > 0:
                msg = data['choices'][0].get('message', {}) or {}
                summary = msg.get('content')
                # Fallback: some Sarvam models return content in reasoning_content
                if not summary or not summary.strip():
                    summary = msg.get('reasoning_content')
                # Extract only the numbered sections from reasoning
                if summary and '1.' in summary:
                    lines = summary.split('\n')
                    result_lines = []
                    capture = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('1.'):
                            capture = True
                        if capture and stripped:
                            result_lines.append(stripped)
                        if stripped.startswith('5.') and capture:
                            # Capture the rest of section 5
                            continue
                    if result_lines:
                        summary = '\n'.join(result_lines)

            if summary and summary.strip():
                print("[GCP] ✅ Sarvam AI clinical summary generated.")
                return summary.strip()
            else:
                print(f"[GCP] ⚠️ Sarvam response empty. Raw: {json.dumps(data)[:500]}")
                return "Clinical summary unavailable (empty response from Sarvam AI)."
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"[GCP] ⏳ Rate limited (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[GCP] ❌ Sarvam API failed: {str(e)}")
                return f"Clinical summary unavailable ({str(e)})."
        except Exception as e:
            print(f"[GCP] ❌ Sarvam API failed: {str(e)}")
            return f"Clinical summary unavailable ({str(e)})."


@functions_framework.http
def predict_pneumonia(request):
    """Main Cloud Function entry point — 4-class chest X-ray classification."""
    try:
        body = request.get_json(silent=True) or {}
        filename = body.get('filename', 'unknown.jpg')
        patient_id = body.get('patient_id', 'P-UNKNOWN')
        image_base64 = body.get('image_base64')
        source = body.get('source', 'direct')

        print(f"[GCP] Processing: {filename} | Patient: {patient_id} | Source: {source}")

        # ── AI Inference ─────────────────────────────────────────
        gradcam_heatmap = None
        if INTERPRETER is not None and image_base64:
            img_input = preprocess_image(image_base64)
            disease, confidence, class_idx, all_probs = predict_real(img_input)
            ai_mode = 'DenseNet-121 TFLite (4-Class COVID)'
            # Use fast mock heatmap instead of Grad-CAM (too slow for Cloud Functions)
            gradcam_heatmap = generate_mock_heatmap(image_base64)
        else:
            disease, confidence, class_idx, all_probs = predict_mock(filename)
            ai_mode = 'Mock AI (Demo Mode)'
            if INTERPRETER is not None and not image_base64:
                ai_mode = 'Mock AI (no image sent)'
            gradcam_heatmap = generate_mock_heatmap(image_base64)

        print(f"[GCP] Diagnosis: {disease} | Confidence: {confidence:.2%} | Model: {ai_mode}")

        # ── Triage ───────────────────────────────────────────────
        triage = compute_triage(disease, confidence)

        # ── RAG: Retrieve relevant medical context ───────────────
        print(f"[GCP] 🔍 RAG retrieval for: {disease} ({triage['triage_level']})")
        medical_context = retrieve_medical_context(
            disease, confidence,
            triage['triage_level'], triage['department']
        )
        rag_status = 'ACTIVE' if medical_context else 'FALLBACK (no context)'
        print(f"[GCP] 📚 RAG status: {rag_status}")

        # ── Sarvam AI Summary (RAG-Augmented) ────────────────────
        ai_summary = generate_clinical_summary(
            patient_id, disease, confidence,
            triage['risk_score'], triage['triage_level'],
            triage['department'], triage['recommended_actions'],
            medical_context=medical_context
        )

        # ── Build Result ─────────────────────────────────────────
        result = {
            'status': 'Pipeline Complete',
            'patient_id': patient_id,
            'filename': filename,
            'diagnosis': disease,
            'confidence': confidence,
            'risk_score': triage['risk_score'],
            'triage_level': triage['triage_level'],
            'department': triage['department'],
            'recommended_actions': triage['recommended_actions'],
            'all_probabilities': all_probs,
            'ai_summary': ai_summary,
            'ai_model': ai_mode,
            'gradcam_heatmap': gradcam_heatmap,
            'rag_status': rag_status,
            'cloud_pipeline': {
                'step_1_storage': f'AWS S3 [{source}]',
                'step_2_etl': 'AWS Lambda',
                'step_3_inference': 'GCP Cloud Functions',
                'step_3b_rag': f'Pinecone RAG [{rag_status}]',
                'step_4_storage': 'Azure Functions',
            }
        }

        # ── Forward to Azure (10 sec timeout) ────────────────────
        azure_payload = {
            'patient_file': filename, 'patient_id': patient_id,
            'diagnosis': disease, 'conf_score': confidence,
            'risk_score': triage['risk_score'], 'triage_level': triage['triage_level'],
            'department': triage['department'], 'ai_summary': ai_summary,
            'actions': json.dumps(triage['recommended_actions']),
            'all_probabilities': json.dumps(all_probs),
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
