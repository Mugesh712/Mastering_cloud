# 🫁 PneumoCloud AI — Multi-Cloud Pneumonia Detection & Triage System

A **multi-cloud intelligent medical imaging system** that detects pneumonia from chest X-rays using deep learning, provides clinical triage recommendations, and demonstrates a real-world **AWS → GCP → Azure** serverless pipeline.

Built as part of the **Mastering Cloud** course (2026).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [AI/ML Pipeline](#aiml-pipeline)
- [Cloud Pipeline](#cloud-pipeline)
- [Setup & Installation](#setup--installation)
- [Deployment Guide](#deployment-guide)
- [Running the Dashboard](#running-the-dashboard)
- [Results & Evaluation](#results--evaluation)
- [Screenshots](#screenshots)
- [License](#license)

---

## Overview

**PneumoCloud AI** is an end-to-end medical imaging system that:

1. **Detects pneumonia** from chest X-ray images using a DenseNet-121 CNN (transfer learning)
2. **Explains decisions** with Grad-CAM heatmap visualizations
3. **Scores patient risk** using an XGBoost ensemble model
4. **Triages patients** into CRITICAL / URGENT / STANDARD / LOW categories
5. **Routes across 3 clouds** — AWS (storage & ETL), GCP (AI inference), Azure (record storage)
6. **Visualises results** in a premium Streamlit dashboard

### Key Highlights

| Feature | Detail |
|---|---|
| **AI Model** | DenseNet-121 with transfer learning (ImageNet weights) |
| **Dataset** | Kaggle Chest X-ray dataset — 5,856 images (Normal vs Pneumonia) |
| **Accuracy** | ~94% on test set |
| **Explainability** | Grad-CAM heatmaps showing AI attention regions |
| **Risk Scoring** | XGBoost model combining CNN confidence + patient metadata |
| **Triage** | Rule-based engine → ICU / ED / Outpatient routing |
| **Cloud Services** | AWS S3, AWS Lambda, GCP Cloud Functions, Azure Functions |
| **Frontend** | Streamlit with glassmorphism UI, animated pipeline tracker |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  MULTI-CLOUD PIPELINE ARCHITECTURE               │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌──────────┐ │
│  │  AWS S3   │────▶│AWS Lambda│────▶│ GCP Cloud│───▶│  Azure   │ │
│  │  Bucket   │     │   ETL    │     │ Function │    │ Function │ │
│  │ (Storage) │     │(Process) │     │ (AI/ML)  │    │ (Record) │ │
│  └──────────┘     └──────────┘     └──────────┘    └──────────┘ │
│       ▲                                                  │       │
│       │              ┌──────────────┐                    │       │
│       └──────────────│  Streamlit   │◀───────────────────┘       │
│                      │  Dashboard   │                            │
│                      └──────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **AWS S3** — Chest X-ray image is uploaded to an S3 bucket
2. **AWS Lambda** — S3 event trigger invokes Lambda; reads image, encodes to base64, sends to GCP
3. **GCP Cloud Functions** — Receives image, runs DenseNet-121 AI inference (mock mode in serverless), computes risk score & triage level, forwards results to Azure
4. **Azure Functions** — Receives complete diagnosis record, stores it with timestamps and compliance metadata
5. **Streamlit Dashboard** — Can invoke the pipeline directly or operate in local/demo mode; displays diagnosis, Grad-CAM, risk score, triage, and full pipeline status

---

## Tech Stack

### AI / ML
| Component | Technology |
|---|---|
| CNN Model | DenseNet-121 (TensorFlow/Keras, ImageNet transfer learning) |
| Risk Model | XGBoost classifier |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Preprocessing | OpenCV, Pillow, NumPy |
| Training | Google Colab (T4 GPU) |

### Cloud Services
| Provider | Service | Role |
|---|---|---|
| **AWS** | S3 | X-ray image storage |
| **AWS** | Lambda | ETL — image preprocessing & forwarding |
| **GCP** | Cloud Run Functions | AI inference engine |
| **Azure** | Functions | Medical record storage |

### Frontend
| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Styling | Custom CSS (glassmorphism, gradients, animations) |
| Visualisation | Matplotlib, Pillow (Grad-CAM overlays) |

---

## Project Structure

```
Mastering_cloud/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Root GCP function (initial version)
│
├── aws/
│   └── lambda_function.py       # AWS Lambda — S3 trigger, ETL, forward to GCP
│
├── gcp/
│   ├── main.py                  # GCP Cloud Function — AI inference + triage
│   └── requirements.txt         # GCP function dependencies
│
├── azure/
│   └── function_app.py          # Azure Function — store diagnosis records
│
├── frontend/
│   └── app.py                   # Streamlit dashboard (premium UI)
│
├── src/
│   ├── preprocess.py            # Image preprocessing & augmentation pipeline
│   ├── train_cnn.py             # DenseNet-121 model builder & trainer
│   ├── train_risk_model.py      # XGBoost risk scoring model
│   ├── triage_engine.py         # Clinical triage decision engine
│   ├── evaluate.py              # Model evaluation & metrics visualisation
│   └── gradcam.py               # Grad-CAM heatmap generator
│
├── notebooks/
│   └── train_model_colab.py     # Google Colab training script
│
└── models/
    ├── pneumonia_model.h5       # Trained DenseNet-121 weights (31 MB)
    ├── risk_model.pkl           # Trained XGBoost risk model
    ├── evaluation_metrics.png   # Training/validation curves
    ├── gradcam_results.png      # Grad-CAM sample outputs
    ├── risk_feature_importance.png  # XGBoost feature importance
    └── sample_predictions.png   # Sample prediction grid
```

---

## AI/ML Pipeline

### 1. Data Preprocessing (`src/preprocess.py`)
- Loads chest X-ray images from Kaggle dataset directory structure
- Resizes to 224×224 RGB
- Normalises pixel values to [0, 1]
- Data augmentation: rotation, shift, zoom, horizontal flip
- Train/Validation/Test split: 70% / 15% / 15%
- Class balancing via oversampling

### 2. CNN Training (`src/train_cnn.py`, `notebooks/train_model_colab.py`)
- **Base model**: DenseNet-121 pre-trained on ImageNet
- **Custom head**: GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- **Phase 1**: Frozen base, train head only (10 epochs, lr=1e-3)
- **Phase 2**: Unfreeze last 30 layers, fine-tune (20 epochs, lr=1e-5)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Output**: `models/pneumonia_model.h5`

### 3. Risk Scoring (`src/train_risk_model.py`)
- XGBoost gradient boosting classifier
- Features: CNN confidence, patient age, gender, comorbidity flags
- Generates synthetic training data for demonstration
- **Output**: `models/risk_model.pkl`

### 4. Explainability (`src/gradcam.py`)
- Grad-CAM: extracts gradients from last convolutional layer
- Generates heatmap overlay showing which lung regions drove the AI decision
- Supports both local generation and base64 encoding for API transfer

### 5. Triage Engine (`src/triage_engine.py`)
- Combines CNN prediction + risk score
- Rule-based classification:

| Risk Score | Triage Level | Department | Actions |
|---|---|---|---|
| ≥ 0.8 | 🔴 CRITICAL | ICU - Pulmonology | Admit immediately, IV antibiotics, CT scan |
| ≥ 0.5 | 🟠 URGENT | Emergency Dept | Admit to ED, oral antibiotics, monitor vitals |
| ≥ 0.25 | 🟡 STANDARD | Outpatient Radiology | Follow-up in 48h, repeat X-ray |
| < 0.25 | 🟢 LOW | General Outpatient | Routine follow-up |

### 6. Evaluation (`src/evaluate.py`)
- Confusion matrix, ROC curve, Precision-Recall curve
- Classification report (precision, recall, F1-score)
- Sample prediction visualisation grid

---

## Cloud Pipeline

### AWS — Storage & ETL

**S3 Bucket**: `xray-upload-mugesh`
- Stores uploaded chest X-ray images
- Triggers Lambda on `s3:ObjectCreated:*`

**Lambda Function**: `xray-etl-processor`
- Runtime: Python 3.12
- Trigger: S3 event
- Reads image from S3, validates format (JPEG/PNG)
- Base64 encodes and forwards to GCP Cloud Function
- IAM: LabRole (AWS Academy)

### GCP — AI Inference

**Cloud Run Function**: `pneumonia-analyzer`
- Runtime: Python 3.11
- Region: europe-west1
- Receives base64-encoded X-ray
- Runs AI inference (mock mode in serverless — TensorFlow too large)
- Computes risk score and triage level
- Forwards complete diagnosis to Azure
- URL: `https://pneumonia-analyzer-782668642236.europe-west1.run.app`

### Azure — Record Storage

**Function App**: `pneumonia-receiver-mugesh1`
- Runtime: Python 3.11
- Region: Central US
- HTTP trigger receives diagnosis JSON
- Stores complete medical record with timestamps
- Returns confirmation with record ID
- URL: `https://pneumonia-receiver-mugesh1-dvahfjd8cca9gmcm.centralus-01.azurewebsites.net/api/savediagnosis`

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip
- AWS CLI (for AWS deployment)
- Google Cloud SDK (for GCP deployment)
- Azure Functions Core Tools v4 (for Azure deployment)
- GitHub CLI (optional)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Mugesh712/Mastering_cloud.git
cd Mastering_cloud

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Deployment Guide

### 1. AWS Lambda

```bash
# Create S3 bucket
aws s3 mb s3://xray-upload-mugesh

# Create Lambda function via AWS Console:
# - Name: xray-etl-processor
# - Runtime: Python 3.12
# - Role: LabRole
# - Paste code from aws/lambda_function.py
# - Add S3 trigger: bucket=xray-upload-mugesh, event=s3:ObjectCreated:*
# - Set timeout to 30 seconds
```

### 2. GCP Cloud Function

```bash
cd gcp/

# Deploy
gcloud functions deploy pneumonia-analyzer \
  --gen2 \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region europe-west1 \
  --entry-point predict_pneumonia \
  --memory 512Mi \
  --timeout 60s
```

### 3. Azure Function

```bash
# Create resource group & function app
az group create --name pneumonia-rg --location centralus
az storage account create --name pneurecvmug --location centralus --resource-group pneumonia-rg --sku Standard_LRS
az functionapp create --resource-group pneumonia-rg --consumption-plan-location centralus \
  --runtime python --runtime-version 3.11 --functions-version 4 \
  --name pneumonia-receiver-mugesh1 --storage-account pneurecvmug --os-type Linux

# Deploy (from /tmp/azure-func or azure/ directory)
func azure functionapp publish pneumonia-receiver-mugesh1 --python --build local
```

---

## Running the Dashboard

```bash
cd frontend
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** and supports 3 modes:

| Mode | Description |
|---|---|
| ☁️ Cloud Pipeline | Sends X-ray through real AWS→GCP→Azure pipeline |
| 💻 Local Model | Uses locally loaded DenseNet-121 model (requires TensorFlow) |
| 🎮 Demo Mode | Simulated AI responses for demonstration without cloud access |

### Dashboard Features
- Upload chest X-ray image (JPG/PNG)
- AI diagnosis with confidence score
- Grad-CAM explainability heatmap
- Risk score with progress bar
- Triage level badge with department routing
- Clinical action recommendations
- Multi-cloud pipeline status tracker
- Diagnosis history table

---

## Results & Evaluation

### Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~94% |
| Precision (Pneumonia) | ~93% |
| Recall (Pneumonia) | ~97% |
| F1-Score (Pneumonia) | ~95% |
| AUC-ROC | ~0.97 |

### Sample Output

```json
{
  "diagnosis": "PNEUMONIA DETECTED",
  "confidence": 0.94,
  "risk_score": 0.808,
  "triage_level": "CRITICAL",
  "department": "ICU - Pulmonology",
  "recommended_actions": [
    "Admit to ICU immediately",
    "Start IV antibiotics",
    "Order chest CT scan"
  ],
  "azure_status": "SAVED"
}
```

---

## Training the Model

The model was trained on **Google Colab** using a T4 GPU:

1. Upload `notebooks/train_model_colab.py` to Google Colab
2. Run all cells — downloads dataset, trains model, generates evaluation plots
3. Download outputs: `pneumonia_model.h5`, `risk_model.pkl`, and PNG plots
4. Place files in the `models/` directory

**Dataset**: [Kaggle Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 images (1,583 Normal + 4,273 Pneumonia)

---

## License

This project was built for educational purposes as part of the **Mastering Cloud** university course (2026).

---

<p align="center">
  <b>🫁 PneumoCloud AI</b> — Multi-Cloud Pneumonia Detection & Triage System<br>
  AWS • GCP • Azure • DenseNet-121 • Grad-CAM • XGBoost • Streamlit
</p>
