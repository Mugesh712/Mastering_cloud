"""
config.py — Centralized Configuration
=======================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

All project-wide settings in ONE place.
Change a value here and it updates everywhere.
"""

import os

# ─────────────────────────────────────────
# IMAGE SETTINGS
# DenseNet-121 expects 224 x 224 RGB input
# ─────────────────────────────────────────
IMG_SIZE   = 224   # pixels (width and height)
CHANNELS   = 3     # RGB — 3 colour channels

# ─────────────────────────────────────────
# MODEL FILE PATHS
# Place trained model files in the models/ folder
# ─────────────────────────────────────────
MODEL_DIR       = os.environ.get('MODEL_DIR', 'models')
CNN_MODEL_PATH  = os.path.join(MODEL_DIR, 'pneumonia_model.h5')   # DenseNet-121
RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_model.pkl')       # XGBoost

# ─────────────────────────────────────────
# CLASSES
# 0 = Normal chest X-ray
# 1 = Pneumonia detected
# ─────────────────────────────────────────
CLASSES      = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES  = len(CLASSES)

# ─────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# Used by train_cnn.py on Google Colab
# ─────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS           = 20
LEARNING_RATE    = 1e-4    # initial learning rate
FINE_TUNE_LR     = 1e-5    # lower rate for fine-tuning
FINE_TUNE_LAYERS = 30      # number of DenseNet layers to unfreeze

# ─────────────────────────────────────────
# DATA AUGMENTATION
# Used to solve Class Imbalance during training
# (4273 Pneumonia images vs 1583 Normal images)
# ─────────────────────────────────────────
DATA_AUGMENTATION_CONFIG = {
    'rotation_range': 15,       # Rotate slightly up to 15 degrees
    'zoom_range': 0.1,          # Zoom in/out by 10%
    'width_shift_range': 0.1,   # Shift horizontally by 10%
    'height_shift_range': 0.1,  # Shift vertically by 10%
    'horizontal_flip': True,    # Allow flipping left-to-right
    'fill_mode': 'nearest'      # How to fill missing pixels
}

# ─────────────────────────────────────────
# TRIAGE THRESHOLDS
# Risk score → triage category
#   ≥ 0.8  → CRITICAL (ICU)
#   ≥ 0.5  → URGENT   (Emergency)
#   ≥ 0.25 → STANDARD (Outpatient)
#   < 0.25 → LOW      (Routine)
# ─────────────────────────────────────────
TRIAGE_CRITICAL  = 0.80
TRIAGE_URGENT    = 0.50
TRIAGE_STANDARD  = 0.25

# ─────────────────────────────────────────
# CLOUD SERVICE URLs
# Override via environment variables in production
# These default URLs point to the deployed functions
# ─────────────────────────────────────────
GCP_FUNCTION_URL = os.environ.get(
    'GCP_FUNCTION_URL',
    'https://pneumonia-analyzer-782668642236.europe-west1.run.app'
)

AZURE_FUNCTION_URL = os.environ.get(
    'AZURE_FUNCTION_URL',
    'https://pneumonia-receiver-mugesh1.azurewebsites.net/api/savediagnosis'
)

# ─────────────────────────────────────────
# AWS SETTINGS
# ─────────────────────────────────────────
AWS_BUCKET_NAME = 'xray-upload-mugesh'
AWS_REGION      = 'us-east-1'
