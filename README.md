# 🫁 PneumoCloud AI — Multi-Cloud Pneumonia Detection & Triage System

A **multi-cloud intelligent medical imaging system** that detects pneumonia from chest X-rays using deep learning, provides clinical triage recommendations, and demonstrates a real-world **AWS → GCP → Azure** serverless pipeline.

Built as part of the **Mastering Cloud** course (2026).

---

## 🏗️ Architecture at a Glance

```
Doctor uploads X-ray
        │
        ▼
   ┌─────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
   │  AWS S3 │──────▶│ AWS Lambda   │──────▶│ GCP Cloud    │──────▶│ Azure        │
   │ Bucket  │       │ (ETL)        │       │ Function     │       │ Function     │
   │ Storage │       │ Validate &   │       │ DenseNet-121 │       │ Save record  │
   └─────────┘       │ Forward      │       │ AI Inference │       │ with ID &    │
                     └──────────────┘       │ + Triage     │       │ timestamp    │
                                            └──────────────┘       └──────────────┘
                                                                           │
                                            ┌──────────────────────────────┘
                                            ▼
                                    ┌───────────────┐
                                    │   Streamlit   │
                                    │   Dashboard   │
                                    │  (localhost)  │
                                    └───────────────┘
```

---

## 📁 Project Structure

```
Mastering_cloud/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies for the dashboard
│
├── aws/
│   └── lambda_function.py      ← AWS Lambda: S3 trigger, validate, forward to GCP
│
├── gcp/
│   ├── main.py                 ← GCP Cloud Function: AI inference + triage
│   └── requirements.txt        ← GCP-specific dependencies
│
├── azure/
│   └── function_app.py         ← Azure Function: receive and store diagnosis record
│
├── src/
│   ├── config.py               ← Centralized settings (URLs, paths, thresholds)
│   ├── triage_engine.py        ← Clinical triage logic (risk scoring + routing)
│   └── gradcam.py              ← Grad-CAM heatmap generator
│
├── frontend/
│   └── app.py                  ← Streamlit dashboard (Cloud + Demo modes)
│
└── models/
    ├── pneumonia_model.h5      ← Trained DenseNet-121 (place here after training)
    └── risk_model.pkl          ← Trained XGBoost risk model
```

---

## ☁️ Cloud Pipeline (Step by Step)

### Step 1 — AWS S3 (Storage)
- Doctor uploads chest X-ray to `s3://xray-upload-mugesh`
- S3 fires an event: "new file created!"

### Step 2 — AWS Lambda (ETL)
- Lambda wakes up automatically when S3 fires the event
- Reads the image → validates it's a JPEG/PNG → base64 encodes it
- POSTs the encoded image to GCP Cloud Function

### Step 3 — GCP Cloud Function (AI Inference)
- Receives the base64 image from Lambda
- Runs DenseNet-121 CNN → gets a pneumonia probability (0.0–1.0)
- If model not found → uses Mock AI (based on filename keywords)
- Computes risk score and triage decision
- Forwards the full result to Azure

### Step 4 — Azure Function (Record Storage)
- Receives the complete diagnosis JSON
- Assigns a unique Record ID (e.g. `REC-20260315094250`)
- Logs the record with HIPAA-Ready compliance metadata
- Returns confirmation to GCP

### Step 5 — Streamlit Dashboard (Display)
- Can invoke the pipeline directly (skipping S3/Lambda)
- Displays: diagnosis, confidence, risk score, triage level, department, actions

---

## 🚀 Deployment Guide

### Prerequisites
- Python 3.11+
- AWS CLI (`aws configure`)
- Google Cloud SDK (`gcloud auth login`)
- Azure Functions Core Tools v4 (`func --version`)

### 1️⃣ AWS

```bash
# Create the S3 bucket
aws s3 mb s3://xray-upload-mugesh

# Deploy Lambda via AWS Console:
# 1. Go to AWS Console → Lambda → Create Function
# 2. Name: xray-etl-processor | Runtime: Python 3.12 | Role: LabRole
# 3. Paste code from aws/lambda_function.py
# 4. Add S3 Trigger → bucket: xray-upload-mugesh → event: s3:ObjectCreated:*
# 5. Set Timeout to 30 seconds → Save
```

### 2️⃣ GCP

```bash
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
```

### 3️⃣ Azure

```bash
# Create resource group and storage account
az group create --name pneumonia-rg --location centralus
az storage account create --name pneurecvmug --location centralus \
  --resource-group pneumonia-rg --sku Standard_LRS

# Create the function app
az functionapp create \
  --resource-group pneumonia-rg \
  --consumption-plan-location centralus \
  --runtime python --runtime-version 3.11 \
  --functions-version 4 \
  --name pneumonia-receiver-mugesh1 \
  --storage-account pneurecvmug \
  --os-type Linux

# Deploy the code
cd azure/
func azure functionapp publish pneumonia-receiver-mugesh1 --python --build local
```

---

## 💻 Running the Dashboard Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
cd frontend
streamlit run app.py
```

Open: **http://localhost:8501**

| Mode | Description |
|---|---|
| ☁️ Cloud Pipeline | Sends X-ray through real GCP→Azure pipeline |
| 🎮 Demo Mode | Simulated AI responses — no cloud needed |

---

## 🧠 AI Model Details

The Kaggle Chest X-ray dataset suffers from Severe Class Imbalance (4,273 Pneumonia images vs 1,583 Normal images). To prevent the model from becoming biased towards diagnosing Pneumonia, we utilized **Data Augmentation** (rotation, zoom, shifting, and flipping) during training to artificially expand the Normal images and force the DenseNet-121 model to learn robust, generalized features.

| Metric | Value |
|---|---|
| Architecture | DenseNet-121 (ImageNet pre-training) |
| Dataset | Kaggle Chest X-ray — 5,856 images |
| Class Imbalance | 73% Pneumonia / 27% Normal |
| Solution | **Data Augmentation** used during Colab training |
| Test Accuracy | ~94% |
| AUC-ROC | ~0.97 |
| Explainability | Grad-CAM heatmaps |

---

## 🏥 Triage Levels

| Risk Score | Level | Department |
|---|---|---|
| ≥ 0.80 | 🔴 CRITICAL | ICU — Pulmonology |
| ≥ 0.50 | 🟠 URGENT | Emergency Department |
| ≥ 0.25 | 🟡 STANDARD | Outpatient Radiology |
| < 0.25 | 🟢 LOW | General Outpatient |

---

*Built for the Mastering Cloud university course (2026) — AWS • GCP • Azure • DenseNet-121 • Streamlit*
