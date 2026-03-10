"""
XGBoost Risk Scoring Model
============================
Multi-Cloud Intelligent Chest X-ray Triage System

This script trains an XGBoost model that takes:
- CNN prediction confidence
- Patient age
- Patient gender
And outputs a risk score for clinical triage.

For this project, we generate synthetic patient metadata
since the Kaggle X-ray dataset doesn't include demographics.
In production, this data would come from hospital EHR systems.

Usage (Google Colab):
    >>> from train_risk_model import train_risk_model, predict_risk
    >>> model = train_risk_model()
    >>> risk = predict_risk(model, cnn_confidence=0.92, age=65, gender='M')
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os


# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
MODEL_SAVE_PATH = 'models/risk_model.pkl'
RANDOM_STATE = 42
NUM_SAMPLES = 5000         # Synthetic training samples


def generate_synthetic_data(n_samples=NUM_SAMPLES):
    """
    Generate synthetic patient data for risk model training.

    Features:
        - cnn_confidence: CNN prediction confidence (0-1)
        - age: Patient age (1-95)
        - gender: 0=Female, 1=Male
        - has_comorbidity: Whether patient has existing conditions (0/1)

    Target:
        - triage_level: 0=LOW, 1=STANDARD, 2=URGENT, 3=CRITICAL
    """
    np.random.seed(RANDOM_STATE)

    # Generate features
    cnn_confidence = np.random.beta(2, 5, n_samples)    # Skewed toward lower values
    age = np.random.randint(1, 96, n_samples)
    gender = np.random.randint(0, 2, n_samples)          # 0=F, 1=M
    has_comorbidity = np.random.randint(0, 2, n_samples)

    # Generate target based on clinical logic
    triage = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        score = cnn_confidence[i] * 0.5

        # Age risk
        if age[i] < 5 or age[i] > 70:
            score += 0.25
        elif age[i] > 50:
            score += 0.1

        # Gender minor factor
        if gender[i] == 1:
            score += 0.05

        # Comorbidity
        if has_comorbidity[i] == 1:
            score += 0.15

        # Add noise
        score += np.random.normal(0, 0.05)
        score = np.clip(score, 0, 1)

        # Assign triage level
        if score >= 0.7:
            triage[i] = 3    # CRITICAL
        elif score >= 0.45:
            triage[i] = 2    # URGENT
        elif score >= 0.25:
            triage[i] = 1    # STANDARD
        else:
            triage[i] = 0    # LOW

    df = pd.DataFrame({
        'cnn_confidence': cnn_confidence,
        'age': age,
        'gender': gender,
        'has_comorbidity': has_comorbidity,
        'triage_level': triage
    })

    return df


def train_risk_model(save_path=MODEL_SAVE_PATH):
    """
    Train XGBoost risk scoring model.

    Returns:
        Trained XGBoost model
    """
    print("=" * 50)
    print("RISK MODEL TRAINING")
    print("=" * 50)

    # Generate data
    print("\n[1/4] Generating synthetic patient data...")
    df = generate_synthetic_data()
    print(f"  Samples: {len(df)}")
    print(f"  Triage distribution:")
    labels = ['LOW', 'STANDARD', 'URGENT', 'CRITICAL']
    for level, count in df['triage_level'].value_counts().sort_index().items():
        print(f"    {labels[level]}: {count}")

    # Split
    print("\n[2/4] Splitting data...")
    X = df[['cnn_confidence', 'age', 'gender', 'has_comorbidity']]
    y = df['triage_level']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train
    print("\n[3/4] Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=4,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    print("\n[4/4] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"  Model saved to: {save_path}")

    return model


def predict_risk(model, cnn_confidence, age=50, gender='M', has_comorbidity=0):
    """
    Predict risk/triage level for a single patient.

    Args:
        model: Trained XGBoost model
        cnn_confidence: float (0-1)
        age: int
        gender: 'M' or 'F'
        has_comorbidity: 0 or 1

    Returns:
        dict with triage_level and probabilities
    """
    gender_encoded = 1 if gender == 'M' else 0

    features = np.array([[cnn_confidence, age, gender_encoded, has_comorbidity]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    labels = ['LOW', 'STANDARD', 'URGENT', 'CRITICAL']

    return {
        'triage_level': labels[int(prediction)],
        'triage_probabilities': {
            labels[i]: round(float(p), 4) for i, p in enumerate(probabilities)
        }
    }


if __name__ == '__main__':
    model = train_risk_model()

    # Test predictions
    print("\n" + "=" * 50)
    print("TEST PREDICTIONS")
    print("=" * 50)

    test_cases = [
        {'cnn_confidence': 0.95, 'age': 75, 'gender': 'M', 'has_comorbidity': 1},
        {'cnn_confidence': 0.60, 'age': 45, 'gender': 'F', 'has_comorbidity': 0},
        {'cnn_confidence': 0.15, 'age': 30, 'gender': 'M', 'has_comorbidity': 0},
        {'cnn_confidence': 0.40, 'age': 3, 'gender': 'F', 'has_comorbidity': 1},
    ]

    for i, tc in enumerate(test_cases):
        result = predict_risk(model, **tc)
        print(f"\n  Case {i+1}: CNN={tc['cnn_confidence']}, Age={tc['age']}, "
              f"Gender={tc['gender']} → {result['triage_level']}")
