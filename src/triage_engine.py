"""
Clinical Triage Decision Engine
================================
Multi-Cloud Intelligent Chest X-ray Triage System

This module takes the CNN prediction output and patient metadata,
then generates a clinical triage decision including:
- Risk score (0-1)
- Triage level (CRITICAL / URGENT / STANDARD / LOW)
- Recommended clinical actions
- Department routing

This is the "decision-making" component that makes the project unique.
"""


# ──────────────────────────────────────
# Triage Level Definitions
# ──────────────────────────────────────
TRIAGE_LEVELS = {
    'CRITICAL': {
        'range': (0.8, 1.0),
        'color': 'red',
        'department': 'ICU - Pulmonology',
        'response_time': 'Immediate (< 5 minutes)'
    },
    'URGENT': {
        'range': (0.5, 0.8),
        'color': 'orange',
        'department': 'Emergency Department',
        'response_time': 'Within 30 minutes'
    },
    'STANDARD': {
        'range': (0.2, 0.5),
        'color': 'yellow',
        'department': 'Outpatient Radiology',
        'response_time': 'Within 2 hours'
    },
    'LOW': {
        'range': (0.0, 0.2),
        'color': 'green',
        'department': 'General Outpatient',
        'response_time': 'Routine appointment'
    }
}


# ──────────────────────────────────────
# Clinical Action Templates
# ──────────────────────────────────────
ACTIONS = {
    'CRITICAL': [
        'Admit to ICU immediately',
        'Start IV antibiotics (Ceftriaxone + Azithromycin)',
        'Order chest CT scan for confirmation',
        'Continuous pulse oximetry monitoring',
        'Notify attending pulmonologist',
        'Prepare for possible intubation if SpO2 < 90%'
    ],
    'URGENT': [
        'Admit to Emergency Department',
        'Start oral antibiotics (Amoxicillin-Clavulanate)',
        'Order blood work (CBC, CRP, Procalcitonin)',
        'Chest X-ray PA and lateral views',
        'Monitor vitals every 2 hours',
        'Consult pulmonology if no improvement in 24h'
    ],
    'STANDARD': [
        'Schedule outpatient follow-up within 48 hours',
        'Prescribe oral antibiotics if bacterial suspected',
        'Recommend rest and hydration',
        'Repeat X-ray in 2 weeks',
        'Educate patient on warning signs'
    ],
    'LOW': [
        'No immediate intervention required',
        'Routine follow-up at next scheduled visit',
        'Provide preventive health counseling',
        'Recommend annual chest X-ray screening'
    ]
}


def compute_risk_score(prediction_confidence, patient_age=None, patient_gender=None):
    """
    Compute a clinical risk score (0-1) based on:
    - CNN prediction confidence (primary factor)
    - Patient age (older = higher risk)
    - Patient gender (adjustment factor)

    Args:
        prediction_confidence: float (0-1) — CNN output for pneumonia
        patient_age: int or None — patient age in years
        patient_gender: str or None — 'M' or 'F'

    Returns:
        float: risk score between 0 and 1
    """
    # Base risk = CNN confidence (70% weight)
    risk = prediction_confidence * 0.7

    # Age factor (20% weight)
    if patient_age is not None:
        if patient_age < 5:
            age_factor = 0.8        # Very young = high risk
        elif patient_age < 18:
            age_factor = 0.3        # Children = moderate
        elif patient_age < 50:
            age_factor = 0.2        # Adults = lower risk
        elif patient_age < 70:
            age_factor = 0.6        # Older adults = higher risk
        else:
            age_factor = 0.9        # Elderly = highest risk
        risk += age_factor * 0.2
    else:
        risk += 0.5 * 0.2          # Unknown age = middle risk

    # Gender factor (10% weight) — males slightly higher risk for pneumonia
    if patient_gender == 'M':
        risk += 0.6 * 0.1
    elif patient_gender == 'F':
        risk += 0.4 * 0.1
    else:
        risk += 0.5 * 0.1          # Unknown = middle

    return min(max(risk, 0.0), 1.0)


def get_triage_level(risk_score):
    """
    Determine triage level from risk score.

    Returns:
        str: 'CRITICAL', 'URGENT', 'STANDARD', or 'LOW'
    """
    for level, info in TRIAGE_LEVELS.items():
        low, high = info['range']
        if low <= risk_score <= high:
            return level
    return 'STANDARD'


def generate_triage_decision(prediction_confidence, diagnosis,
                              patient_age=None, patient_gender=None,
                              patient_id='N/A'):
    """
    Generate a complete triage decision.

    Args:
        prediction_confidence: float (0-1) — CNN confidence
        diagnosis: str — 'PNEUMONIA' or 'NORMAL'
        patient_age: int or None
        patient_gender: str or None — 'M' or 'F'
        patient_id: str — patient identifier

    Returns:
        dict: Complete triage decision
    """
    # If diagnosis is NORMAL with low confidence, still compute risk
    if diagnosis == 'NORMAL':
        risk_score = compute_risk_score(1 - prediction_confidence,
                                        patient_age, patient_gender)
    else:
        risk_score = compute_risk_score(prediction_confidence,
                                        patient_age, patient_gender)

    triage_level = get_triage_level(risk_score)
    triage_info = TRIAGE_LEVELS[triage_level]
    actions = ACTIONS[triage_level]

    return {
        'patient_id': patient_id,
        'diagnosis': diagnosis,
        'prediction_confidence': round(prediction_confidence, 4),
        'risk_score': round(risk_score, 4),
        'triage_level': triage_level,
        'triage_color': triage_info['color'],
        'department': triage_info['department'],
        'response_time': triage_info['response_time'],
        'recommended_actions': actions,
    }


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("TRIAGE ENGINE — Test Cases")
    print("=" * 60)

    # Test Case 1: High confidence pneumonia in elderly
    result = generate_triage_decision(
        prediction_confidence=0.94,
        diagnosis='PNEUMONIA',
        patient_age=72,
        patient_gender='M',
        patient_id='P-1001'
    )
    print(f"\nTest 1 — Elderly male, high confidence pneumonia:")
    print(f"  Risk Score  : {result['risk_score']}")
    print(f"  Triage      : {result['triage_level']}")
    print(f"  Department  : {result['department']}")
    print(f"  Actions     : {result['recommended_actions'][0]}")

    # Test Case 2: Low confidence in young adult
    result = generate_triage_decision(
        prediction_confidence=0.35,
        diagnosis='PNEUMONIA',
        patient_age=28,
        patient_gender='F',
        patient_id='P-1002'
    )
    print(f"\nTest 2 — Young female, low confidence:")
    print(f"  Risk Score  : {result['risk_score']}")
    print(f"  Triage      : {result['triage_level']}")
    print(f"  Department  : {result['department']}")
    print(f"  Actions     : {result['recommended_actions'][0]}")

    # Test Case 3: Normal result
    result = generate_triage_decision(
        prediction_confidence=0.08,
        diagnosis='NORMAL',
        patient_age=45,
        patient_gender='M',
        patient_id='P-1003'
    )
    print(f"\nTest 3 — Normal result:")
    print(f"  Risk Score  : {result['risk_score']}")
    print(f"  Triage      : {result['triage_level']}")
    print(f"  Department  : {result['department']}")
    print(f"  Actions     : {result['recommended_actions'][0]}")
