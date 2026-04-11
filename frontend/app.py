"""
Flask Backend — PneumoCloud AI
================================
Serves the HTML/CSS/JS frontend and handles API requests.
"""

from flask import Flask, render_template, request, jsonify
import requests as http_requests
import base64
import io
import os
import json
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
GCP_URL = "https://pneumonia-analyzer-782668642236.europe-west1.run.app"
AZURE_RECORDS_URL = "https://pneumonia-receiver-mugesh1.azurewebsites.net/api/getrecords"

# SMTP Email Config
SMTP_HOST = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_EMAIL = 'mugeshg712@gmail.com'
SMTP_PASSWORD = 'jucqeaoaquaiwjwh'

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'chest_multiclass_model.tflite')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_mapping.json')

CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

DISEASE_DEPARTMENTS = {
    'COVID': 'Pulmonology — Isolation Ward',
    'Lung_Opacity': 'Pulmonology',
    'Normal': 'General Outpatient',
    'Viral Pneumonia': 'Pulmonology — ICU',
}

DISEASE_SEVERITY = {
    'COVID': 0.85,
    'Lung_Opacity': 0.60,
    'Normal': 0.05,
    'Viral Pneumonia': 0.75,
}

DISEASE_ACTIONS = {
    'COVID': [
        'Isolate patient immediately',
        'Start antiviral therapy (Remdesivir)',
        'Monitor SpO2 continuously',
        'Order RT-PCR confirmation',
        'Notify infection control team',
    ],
    'Lung_Opacity': [
        'Order chest CT for detailed assessment',
        'Rule out pneumonia vs. fluid accumulation',
        'Correlate with clinical symptoms',
        'Follow-up imaging in 2-4 weeks',
    ],
    'Normal': [
        'No intervention required',
        'Routine follow-up',
        'Annual screening recommended',
        'Monitor for new symptoms',
    ],
    'Viral Pneumonia': [
        'Start antiviral + supportive therapy',
        'Blood work (CBC, CRP, Procalcitonin)',
        'Monitor SpO2 and respiratory rate',
        'Follow-up X-ray in 4-6 weeks',
        'Consider ICU admission if SpO2 < 92%',
    ],
}


# ─────────────────────────────────────────────────────
# LOAD TFLITE MODEL
# ─────────────────────────────────────────────────────
interpreter = None
class_index_map = {}

def load_model():
    global interpreter, class_index_map
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    except ImportError:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        except ImportError:
            interpreter = None

    if interpreter:
        interpreter.allocate_tensors()
        print(f"✅ TFLite model loaded: {MODEL_PATH}")
    else:
        print("⚠️  No TFLite runtime found. Will use fallback mode.")

    # Load class mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'r') as f:
            class_index_map = json.load(f)
        print(f"✅ Class mapping loaded: {class_index_map}")

load_model()

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_index_map.items()} if class_index_map else {i: c for i, c in enumerate(CLASS_NAMES)}


def run_tflite_inference(img):
    """Run the trained TFLite model on a PIL image."""
    global interpreter
    if interpreter is None:
        return None

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions


@app.route('/')
def index():
    return render_template('index.html')

# ── Local Report Generation ──────────────────────────────────
def generate_local_report(disease, confidence, risk_score, triage_level, dept, actions, patient_id):
    """Generate a detailed clinical report locally for demo mode."""
    from datetime import datetime
    date = datetime.now().strftime('%d %b %Y, %I:%M %p')
    actions_str = ', '.join(actions)

    reports = {
        'Normal': {
            'xray': [
                'Clear bilateral lung fields with no focal opacities or consolidation',
                'Cardiac silhouette within normal limits',
                'Costophrenic angles are sharp bilaterally',
                'No pleural effusion or pneumothorax detected',
                'Trachea is midline, mediastinal structures unremarkable'
            ],
            'lifestyle': [
                'Moderate aerobic exercise (brisk walking, swimming, cycling) — 30 mins, 5 days/week',
                '7-8 hours of quality sleep per night',
                'Practice deep breathing exercises daily',
                'Avoid smoking and secondhand smoke exposure',
                'Maintain good hand hygiene practices'
            ],
            'diet': [
                'Balanced diet rich in fruits, vegetables, leafy greens, and berries',
                'Include omega-3 fatty acids — fish, walnuts, flaxseed',
                'Stay hydrated — 2-3 liters of water daily',
                'Vitamin C rich foods — oranges, bell peppers, guava',
                'Vitamin D sources — fortified milk, eggs, sunlight',
                'Limit processed foods and refined sugars'
            ],
            'followup': [
                'Annual routine chest X-ray for baseline monitoring',
                'Annual general health check-up with primary care physician',
                'Return immediately if persistent cough, breathlessness, or chest pain develops'
            ],
            'prognosis': [
                'Excellent health outlook — lungs functioning normally',
                'No signs of active disease detected',
                'Continue current healthy habits for long-term wellness',
                'Very favorable long-term respiratory prognosis'
            ]
        },
        'COVID': {
            'xray': [
                'Bilateral ground-glass opacities in peripheral and posterior lung fields',
                'Pattern highly consistent with COVID-19 pneumonia',
                'Possible emerging consolidation in lower lobes',
                'Cardiac silhouette appears normal',
                'Findings characteristic of SARS-CoV-2 infection'
            ],
            'lifestyle': [
                'Complete bed rest for 7-10 days',
                'Breathing exercises (incentive spirometry) 4-6 times daily',
                'Strict isolation — no visitors, N95 mask required',
                'Gentle bed mobility exercises to prevent blood clots',
                'Sleep in prone/lateral position for better oxygenation',
                'No strenuous activity for 4-6 weeks post-recovery',
                'Monitor mental health — seek counseling if needed'
            ],
            'diet': [
                'High-protein diet — eggs, lean chicken, lentils, paneer',
                'Anti-inflammatory foods — turmeric milk, ginger, garlic, green tea',
                'Stay hydrated — 3+ liters warm fluids daily (soups, herbal teas)',
                'Vitamin C supplements — 500-1000mg daily',
                'Vitamin D3 — 60,000 IU weekly for 8 weeks',
                'Zinc supplements — 50mg daily for 2 weeks',
                'Avoid cold beverages, fried foods, excessive sugar',
                'Small frequent meals if appetite is poor'
            ],
            'followup': [
                'Daily SpO2 monitoring and temperature checks',
                'Follow-up chest X-ray at 2 weeks and 6 weeks',
                'RT-PCR test at Day 10 for clearance',
                'Weekly blood tests (CRP, D-Dimer) during acute phase',
                'Pulmonary function test at 3 months',
                'EMERGENCY: fever >101F, SpO2 <93%, difficulty breathing, chest pain',
                'Long COVID screening at 3 and 6 months'
            ],
            'prognosis': [
                'Majority recover fully within 2-4 weeks with proper management',
                'Mild to moderate cases — excellent recovery rates (>95%)',
                'Monitor for post-COVID fatigue, brain fog, reduced exercise tolerance',
                'Full lung recovery typically within 3-6 months'
            ]
        },
        'Lung_Opacity': {
            'xray': [
                'Focal or diffuse opacities in lung parenchyma',
                'Possible consolidation, atelectasis, or pleural abnormality',
                'May indicate fluid accumulation or inflammation',
                'Cardiac silhouette needs further evaluation',
                'Clinical correlation recommended for definitive diagnosis'
            ],
            'lifestyle': [
                'Light physical activity only — avoid strenuous exercise',
                'Rest adequately — 8-9 hours of sleep',
                'Use a humidifier for moist air',
                'Avoid dusty environments, smoke, and pollution',
                'Practice cough hygiene and deep breathing exercises',
                'Elevate head during sleep if breathless'
            ],
            'diet': [
                'Anti-inflammatory diet — leafy greens, berries, fatty fish',
                'Warm fluids — soups, herbal teas, honey-lemon water',
                'High-protein foods for tissue repair',
                'Vitamin A rich foods — carrots, sweet potato, spinach',
                'Avoid dairy if excess mucus production',
                'Avoid alcohol — worsens inflammation',
                'Stay hydrated — 2.5-3 liters daily'
            ],
            'followup': [
                'Follow-up chest X-ray in 4-6 weeks',
                'HRCT chest if opacity persists',
                'Pulmonary function tests for lung capacity assessment',
                'Pulmonologist visit within 1 week',
                'EMERGENCY: worsening breathlessness, coughing blood, high fever',
                '3-month follow-up to confirm complete resolution'
            ],
            'prognosis': [
                'Most causes resolve completely within 4-8 weeks with treatment',
                'Full recovery expected with appropriate antibiotic therapy',
                'Early detection prevents complications like abscess formation',
                'Regular monitoring ensures optimal outcome'
            ]
        },
        'Viral Pneumonia': {
            'xray': [
                'Bilateral interstitial infiltrates and patchy ground-glass opacities',
                'Predominantly in lower lung zones',
                'Peribronchial thickening may be present',
                'Pattern consistent with viral etiology',
                'Diffuse bilateral involvement typical of respiratory virus'
            ],
            'lifestyle': [
                'Complete bed rest for 5-7 days',
                'Practice respiratory hygiene — cover coughs, wash hands frequently',
                'Use a humidifier to ease breathing',
                'Light walking after acute symptoms resolve',
                'Avoid contact with others to prevent spread',
                'Gradually resume activities over 2-3 weeks',
                'Breathing exercises during recovery phase'
            ],
            'diet': [
                'Warm, easily digestible meals — soups, porridge, steamed vegetables',
                'Immune-boosting foods — garlic, ginger, turmeric, citrus fruits',
                'Probiotics — yogurt, kefir for gut-immune support',
                'Vitamin C (500mg daily), Zinc (20mg daily)',
                'Plenty of warm fluids — aim for 3 liters daily',
                'Avoid cold beverages and ice cream',
                'Small, frequent meals if appetite is reduced',
                'Protein-rich foods — eggs, dal, fish'
            ],
            'followup': [
                'Follow-up chest X-ray at 2-3 weeks',
                'Temperature and SpO2 monitoring twice daily',
                'Blood tests (CBC, CRP) at 1 week',
                'Pulmonology consultation if symptoms persist beyond 2 weeks',
                'EMERGENCY: SpO2 <92%, severe chest pain, blue lips',
                '6-week follow-up for full recovery assessment',
                'Influenza and pneumococcal vaccination after recovery'
            ],
            'prognosis': [
                'Typically resolves within 1-3 weeks with supportive care',
                'Most patients make a complete recovery',
                'Excellent outcomes for young, healthy patients',
                'Full lung function recovery expected within 4-8 weeks',
                'Post-illness fatigue may last 2-4 weeks — this is normal'
            ]
        }
    }

    r = reports.get(disease, reports['Normal'])

    def fmt_list(items, emoji):
        return '\n'.join([f'{emoji} {item}' for item in items])

    return f"""## 1. DIAGNOSTIC FINDINGS
🔬 AI Model: DenseNet-121 (COVID-19 Radiography Dataset, 4-class)
🩺 Detected Condition: {disease}
📊 Confidence Level: {confidence:.1%}
💡 Interpretation: {'No significant abnormalities detected in the chest radiograph' if disease == 'Normal' else f'Radiological evidence consistent with {disease}'}

## 2. CHEST X-RAY ANALYSIS
{fmt_list(r['xray'], '🔍')}

## 3. SEVERITY ASSESSMENT
⚠️ Risk Score: {risk_score:.2f} / 1.00
🏥 Triage Level: {triage_level}
🏢 Department: {dept}
📌 {'No urgent intervention required' if disease == 'Normal' else 'Immediate medical attention recommended' if triage_level in ['CRITICAL', 'URGENT'] else 'Routine follow-up is recommended'}

## 4. LIFESTYLE RECOMMENDATIONS
{fmt_list(r['lifestyle'], '🏃')}

## 5. DIETARY GUIDELINES
{fmt_list(r['diet'], '🍎')}

## 6. FOLLOW-UP SCHEDULE
{fmt_list(r['followup'], '📅')}

## 7. PROGNOSIS & EXPECTED OUTCOMES
{fmt_list(r['prognosis'], '✅')}"""


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Handle image upload and return analysis results."""
    try:
        file = request.files.get('image')
        mode = request.form.get('mode', 'demo')
        patient_id = request.form.get('patient_id', 'P-001')
        print(f"[ANALYZE] Received request: mode={mode}, patient={patient_id}, file={file.filename if file else 'None'}", flush=True)

        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Read image
        img = Image.open(file).convert('RGB')
        print(f"[ANALYZE] Image opened: {img.size}", flush=True)

        if mode == 'cloud':
            # ── CLOUD MODE: Forward to GCP ──
            print("[ANALYZE] Cloud mode — preparing payload...", flush=True)
            img_resized = img.resize((224, 224))
            buf = io.BytesIO()
            img_resized.save(buf, format='JPEG', quality=70)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            print(f"[ANALYZE] Base64 size: {len(image_base64)} chars", flush=True)

            payload = {
                'filename': file.filename,
                'patient_id': patient_id,
                'image_base64': image_base64,
                'source': 'Web_Dashboard'
            }
            print(f"[ANALYZE] Sending to GCP: {GCP_URL}", flush=True)
            response = http_requests.post(GCP_URL, json=payload, timeout=30)
            print(f"[ANALYZE] GCP responded: HTTP {response.status_code}", flush=True)
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({'error': f'GCP returned HTTP {response.status_code}'}), 500
        else:
            # ── DEMO / LOCAL AI MODE ──
            predictions = run_tflite_inference(img)

            if predictions is not None:
                # Real model inference
                pred_index = int(np.argmax(predictions))
                disease = index_to_class.get(pred_index, CLASS_NAMES[pred_index])
                confidence = float(predictions[pred_index])

                all_probs = {}
                for i, prob in enumerate(predictions):
                    cls_name = index_to_class.get(i, CLASS_NAMES[i])
                    all_probs[cls_name] = float(prob)

                ai_model = 'DenseNet-121 TFLite (Local)'
            else:
                # Fallback: keyword-based mock
                name = file.filename.lower()
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

                all_probs = {}
                for cls in CLASS_NAMES:
                    all_probs[cls] = confidence if cls == disease else round(float((1.0 - confidence) / 3), 4)

                ai_model = 'Mock AI (Fallback — TFLite not available)'

            # ── Compute triage ──
            severity = DISEASE_SEVERITY.get(disease, 0.5)
            risk_score = float(severity * 0.40 + confidence * 0.30 + 0.15)
            risk_score = min(max(risk_score, 0.0), 1.0)

            if risk_score >= 0.80:
                triage_level = 'CRITICAL'
            elif risk_score >= 0.50:
                triage_level = 'URGENT'
            elif risk_score >= 0.25:
                triage_level = 'STANDARD'
            else:
                triage_level = 'LOW'

            dept = DISEASE_DEPARTMENTS.get(disease, 'General Outpatient')
            actions = DISEASE_ACTIONS.get(disease, ['Consult specialist'])

            summary = generate_local_report(disease, confidence, risk_score, triage_level, dept, actions, patient_id)

            return jsonify({
                'patient_id': patient_id,
                'diagnosis': disease,
                'confidence': round(confidence, 4),
                'risk_score': round(risk_score, 4),
                'triage_level': triage_level,
                'department': dept,
                'recommended_actions': actions,
                'all_probabilities': all_probs,
                'ai_summary': summary,
                'ai_model': ai_model,
                'azure_status': 'DEMO — not saved',
                'cloud_pipeline': {
                    'step_1_storage': 'Local upload',
                    'step_2_etl': 'Local preprocessing',
                    'step_3_inference': ai_model,
                    'step_4_storage': 'DEMO (not saved)',
                }
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/records')
def get_records():
    """Fetch records from Azure."""
    try:
        resp = http_requests.get(AZURE_RECORDS_URL, params={'limit': 100}, timeout=15)
        if resp.status_code == 200:
            return jsonify(resp.json())
        return jsonify({'records': [], 'error': f'HTTP {resp.status_code}'}), 200
    except Exception as e:
        return jsonify({'records': [], 'error': str(e)}), 200


AZURE_DELETE_URL = "https://pneumonia-receiver-mugesh1.azurewebsites.net/api/deleterecord"

@app.route('/api/records/delete', methods=['POST'])
def delete_record():
    """Delete a record from Azure Cosmos DB."""
    try:
        data = request.get_json()
        record_id = data.get('record_id', '').strip()

        if not record_id:
            return jsonify({'success': False, 'error': 'record_id is required'}), 400

        resp = http_requests.post(AZURE_DELETE_URL, json={'record_id': record_id}, timeout=15)

        if resp.status_code == 200:
            print(f"[DELETE] ✅ Record {record_id} deleted from Azure")
            return jsonify(resp.json())
        else:
            error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {'error': f'HTTP {resp.status_code}'}
            return jsonify(error_data), resp.status_code

    except Exception as e:
        print(f"[DELETE] ❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send-report', methods=['POST'])
def send_report_email():
    """Send the clinical report via email."""
    try:
        data = request.get_json()
        recipient_email = data.get('email', '').strip()
        result = data.get('result', {})
        summary = data.get('summary', '')

        if not recipient_email:
            return jsonify({'success': False, 'error': 'Email address is required'}), 400

        if not SMTP_PASSWORD:
            return jsonify({'success': False, 'error': 'SMTP not configured. Set SMTP_PASSWORD environment variable.'}), 500

        # Build HTML email
        disease = result.get('diagnosis', 'N/A')
        confidence = result.get('confidence', 0)
        risk_score = result.get('risk_score', 0)
        triage_level = result.get('triage_level', 'N/A')
        department = result.get('department', 'N/A')
        patient_id = result.get('patient_id', 'N/A')
        ai_model = result.get('ai_model', 'DenseNet-121')
        is_normal = disease == 'Normal'
        date = datetime.now().strftime('%d %b %Y, %I:%M %p')

        # Color based on severity
        diag_color = '#10b981' if is_normal else '#ef4444'
        diag_bg = '#e6fff5' if is_normal else '#ffebeb'
        triage_colors = {
            'CRITICAL': '#dc2626', 'URGENT': '#f97316',
            'STANDARD': '#f59e0b', 'LOW': '#10b981'
        }
        triage_color = triage_colors.get(triage_level, '#3b82f6')

        # Process summary: remove treatment plan section
        import re
        clean_summary = re.sub(
            r'(?:##\s*)?\d+\.\s*TREATMENT\s+PLAN\s*[&]\s*CLINICAL\s+ACTIONS[\s\S]*?(?=(?:##\s*)?\d+\.\s+[A-Z]|$)',
            '', summary, flags=re.IGNORECASE
        ).strip()

        # Convert summary to HTML
        summary_html = ''
        sections = re.split(r'(?:##\s*)?(\d+)\.\s+([A-Z][A-Z\s&]+)', clean_summary)
        section_colors = {
            'DIAGNOSTIC': '#3b82f6', 'CHEST': '#8b5cf6', 'SEVERITY': '#f59e0b',
            'LIFESTYLE': '#10b981', 'DIETARY': '#ec4899', 'FOLLOW': '#f97316', 'PROGNOSIS': '#06b6d4'
        }

        i = 1
        section_num = 0
        while i < len(sections) - 1:
            section_num += 1
            title = sections[i + 1].strip()
            content = sections[i + 2].strip() if i + 2 < len(sections) else ''
            i += 3

            sec_color = '#3b82f6'
            for key, color in section_colors.items():
                if key in title.upper():
                    sec_color = color
                    break

            # Convert content to bullet points
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            bullet_lines = [l for l in lines if l.startswith(('-', '•')) or any(l.startswith(c) for c in '🔬🩺📊💡🔍⚠️🏥🏢📌🏃😴🏠🧘🍎🥗💧🚫💊📅🔄🧪🚨📋✅⚡🌟💪')]

            if len(bullet_lines) >= 2:
                points = [l.lstrip('-•').strip() for l in bullet_lines]
            else:
                full = ' '.join(lines)
                points = [s.strip() for s in re.split(r'(?<=\.)\s+(?=[A-Z])', full) if len(s.strip()) > 15]
                if not points:
                    points = [full]

            points_html = ''.join([
                f'<tr><td style="padding:4px 8px;vertical-align:top;color:{sec_color};font-size:18px;">•</td>'
                f'<td style="padding:4px 8px;color:#444;font-size:14px;line-height:1.6;">{p}</td></tr>'
                for p in points
            ])

            summary_html += f'''
            <div style="margin-bottom:20px;">
                <div style="background:{sec_color};color:white;padding:8px 16px;border-radius:6px;font-size:14px;font-weight:700;letter-spacing:0.5px;">
                    {section_num}. {title}
                </div>
                <table style="width:100%;margin-top:8px;border-collapse:collapse;">
                    {points_html}
                </table>
            </div>'''

        # Probability table
        prob_html = ''
        all_probs = result.get('all_probabilities', {})
        if all_probs:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            prob_rows = ''
            prob_colors = {'COVID': '#ef4444', 'Viral Pneumonia': '#f97316', 'Lung_Opacity': '#f59e0b', 'Normal': '#10b981'}
            for name, prob in sorted_probs:
                pc = prob_colors.get(name, '#3b82f6')
                bar_width = int(prob * 200)
                prob_rows += f'''
                <tr>
                    <td style="padding:6px 12px;font-size:13px;color:#444;font-weight:500;">{name}</td>
                    <td style="padding:6px 12px;font-size:13px;color:#444;font-weight:700;">{prob*100:.1f}%</td>
                    <td style="padding:6px 12px;"><div style="width:{bar_width}px;height:12px;background:{pc};border-radius:3px;"></div></td>
                </tr>'''
            prob_html = f'''
            <div style="margin:20px 0;">
                <div style="font-size:15px;font-weight:700;color:#0096dc;margin-bottom:10px;">DISEASE PROBABILITY DISTRIBUTION</div>
                <table style="width:100%;border-collapse:collapse;background:#f8fafc;border-radius:8px;">{prob_rows}</table>
            </div>'''

        html_body = f'''
        <div style="max-width:650px;margin:0 auto;font-family:Arial,Helvetica,sans-serif;">
            <!-- Header -->
            <div style="background:linear-gradient(135deg,#0a1937,#1a2f5a);padding:30px;border-radius:12px 12px 0 0;text-align:center;">
                <div style="font-size:28px;margin-bottom:6px;">🫁</div>
                <h1 style="color:white;font-size:24px;margin:0;">PneumoCloud AI</h1>
                <p style="color:#8aa4c8;font-size:13px;margin:4px 0 0 0;">Multi-Cloud Chest X-Ray Classification System</p>
                <div style="width:60px;height:3px;background:#00d2ff;margin:12px auto 0;border-radius:2px;"></div>
            </div>

            <!-- Body -->
            <div style="background:white;padding:30px;border:1px solid #e8ecf1;">
                <h2 style="color:#1a2f5a;font-size:18px;margin:0 0 20px 0;border-bottom:2px solid #00d2ff;padding-bottom:10px;">📋 Clinical Analysis Report</h2>

                <!-- Patient Info -->
                <table style="width:100%;background:#f0f5ff;border:1px solid #d0e0ff;border-radius:8px;border-collapse:collapse;margin-bottom:20px;">
                    <tr>
                        <td style="padding:12px 16px;font-size:13px;color:#1a2f5a;"><strong>Patient ID:</strong> {patient_id}</td>
                        <td style="padding:12px 16px;font-size:13px;color:#1a2f5a;"><strong>Date:</strong> {date}</td>
                    </tr>
                    <tr>
                        <td style="padding:12px 16px;font-size:13px;color:#1a2f5a;"><strong>AI Model:</strong> {ai_model}</td>
                        <td style="padding:12px 16px;font-size:13px;color:#1a2f5a;"><strong>Pipeline:</strong> AWS → GCP → Azure</td>
                    </tr>
                </table>

                <!-- Diagnosis Box -->
                <div style="background:{diag_bg};border-left:4px solid {diag_color};padding:16px 20px;border-radius:0 8px 8px 0;margin-bottom:20px;">
                    <div style="font-size:22px;font-weight:800;color:{diag_color};">{disease}</div>
                    <div style="font-size:13px;color:#666;margin-top:6px;">
                        Confidence: <strong>{confidence*100:.1f}%</strong> &nbsp;|&nbsp;
                        Risk: <strong>{risk_score:.3f}</strong> &nbsp;|&nbsp;
                        Triage: <span style="background:{triage_color};color:white;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:700;">{triage_level}</span> &nbsp;|&nbsp;
                        Dept: <strong>{department}</strong>
                    </div>
                </div>

                {prob_html}

                <!-- Report Sections -->
                {summary_html}
            </div>

            <!-- Footer -->
            <div style="background:#f8fafc;padding:20px 30px;border:1px solid #e8ecf1;border-top:none;border-radius:0 0 12px 12px;text-align:center;">
                <p style="color:#8aa4c8;font-size:11px;margin:0;">This report was generated by PneumoCloud AI using DenseNet-121 deep learning model.</p>
                <p style="color:#8aa4c8;font-size:11px;margin:4px 0 0 0;">For clinical decisions, always consult a qualified healthcare professional.</p>
                <p style="color:#aab8cc;font-size:10px;margin:8px 0 0 0;">PneumoCloud AI | Multi-Cloud Pipeline: AWS • GCP • Azure</p>
            </div>
        </div>
        '''

        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'PneumoCloud AI Report — {disease} ({patient_id})'
        msg['From'] = f'PneumoCloud AI <{SMTP_EMAIL}>'
        msg['To'] = recipient_email

        # Plain text fallback
        plain_text = f"""PneumoCloud AI - Clinical Analysis Report
{'='*50}
Patient: {patient_id} | Date: {date}
Diagnosis: {disease} | Confidence: {confidence*100:.1f}%
Risk: {risk_score:.3f} | Triage: {triage_level} | Dept: {department}
{'='*50}

{clean_summary}

{'='*50}
Generated by PneumoCloud AI (DenseNet-121)
"""

        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"[EMAIL] ✅ Report sent to {recipient_email}")
        return jsonify({'success': True, 'message': f'Report sent to {recipient_email}'})

    except smtplib.SMTPAuthenticationError:
        print(f"[EMAIL] ❌ SMTP authentication failed")
        return jsonify({'success': False, 'error': 'Email authentication failed. Check SMTP credentials.'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[EMAIL] ❌ Failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0', threaded=True)
