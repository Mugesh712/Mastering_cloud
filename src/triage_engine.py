"""
triage_engine.py — Clinical Triage Decision Engine
=====================================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

Takes the AI model's output and patient metadata,
produces a risk score and triage decision.

TRIAGE LEVELS:
  🔴 CRITICAL  → ICU immediately
  🟠 URGENT    → Emergency Department
  🟡 STANDARD  → Outpatient follow-up
  🟢 LOW       → Routine care
"""

from src.config import TRIAGE_CRITICAL, TRIAGE_URGENT, TRIAGE_STANDARD


def compute_risk_score(cnn_confidence: float, age: int = 50, has_comorbidity: bool = False) -> float:
    """
    Compute a risk score from 0.0 to 1.0 using:
      - CNN confidence (main driver, 70% weight)
      - Patient age     (older = higher risk, 20% weight)
      - Comorbidities   (diabetes, heart disease, etc., 10% weight)

    Args:
        cnn_confidence:  AI model confidence that patient HAS pneumonia (0.0 – 1.0)
        age:             Patient age in years (default 50)
        has_comorbidity: True if patient has known conditions that worsen outcome

    Returns:
        risk_score: Float between 0.0 and 1.0
    """
    # Normalise age to a 0–1 scale (clamp at 90)
    age_factor = min(age / 90.0, 1.0)

    # Comorbidity factor: 1.0 if has comorbidity, else 0.0
    comorbidity_factor = 1.0 if has_comorbidity else 0.0

    # Weighted sum
    risk_score = (
        cnn_confidence    * 0.70 +
        age_factor        * 0.20 +
        comorbidity_factor * 0.10
    )

    # Clamp to [0.0, 1.0] just in case
    return round(min(max(risk_score, 0.0), 1.0), 4)


def get_triage_level(risk_score: float) -> dict:
    """
    Map a risk score to a triage level, department, and recommended clinical actions.

    Args:
        risk_score: Float between 0.0 and 1.0

    Returns:
        dict with keys: triage_level, department, recommended_actions, emoji
    """
    if risk_score >= TRIAGE_CRITICAL:
        return {
            'triage_level': 'CRITICAL',
            'emoji': '🔴',
            'department': 'ICU — Pulmonology',
            'recommended_actions': [
                'Admit to ICU immediately',
                'Start IV antibiotics (Ceftriaxone + Azithromycin)',
                'Order chest CT scan for confirmation',
                'Notify attending pulmonologist',
                'Continuous vital sign monitoring',
            ]
        }
    elif risk_score >= TRIAGE_URGENT:
        return {
            'triage_level': 'URGENT',
            'emoji': '🟠',
            'department': 'Emergency Department',
            'recommended_actions': [
                'Admit to Emergency Department',
                'Start oral antibiotics',
                'Order blood work (CBC, CRP, blood cultures)',
                'Monitor vitals every 2 hours',
                'Chest X-ray in 24 hours',
            ]
        }
    elif risk_score >= TRIAGE_STANDARD:
        return {
            'triage_level': 'STANDARD',
            'emoji': '🟡',
            'department': 'Outpatient Radiology',
            'recommended_actions': [
                'Schedule outpatient follow-up within 48 hours',
                'Prescribe oral antibiotics if bacterial suspected',
                'Repeat chest X-ray in 2 weeks',
                'Advise rest and increased fluid intake',
            ]
        }
    else:
        return {
            'triage_level': 'LOW',
            'emoji': '🟢',
            'department': 'General Outpatient',
            'recommended_actions': [
                'No immediate intervention required',
                'Routine follow-up at next scheduled visit',
                'Annual screening recommended',
                'Monitor for symptom changes',
            ]
        }


def run_triage(cnn_confidence: float, diagnosis: str,
               age: int = 50, has_comorbidity: bool = False) -> dict:
    """
    Full triage pipeline: risk score + triage decision.

    Args:
        cnn_confidence:  Model confidence (0.0–1.0)
        diagnosis:       'PNEUMONIA DETECTED' or 'NORMAL'
        age:             Patient age
        has_comorbidity: Comorbidity flag

    Returns:
        Complete triage result dict
    """
    # If the AI says NORMAL with high confidence, risk is low
    if diagnosis == 'NORMAL':
        raw_confidence = 1.0 - cnn_confidence   # invert: confidence of being normal
    else:
        raw_confidence = cnn_confidence

    risk_score = compute_risk_score(raw_confidence, age, has_comorbidity)
    triage = get_triage_level(risk_score)

    return {
        'diagnosis': diagnosis,
        'confidence': round(cnn_confidence, 4),
        'risk_score': risk_score,
        'triage_level': triage['triage_level'],
        'emoji': triage['emoji'],
        'department': triage['department'],
        'recommended_actions': triage['recommended_actions'],
    }
