# Multi-Cloud Architecture — PneumoCloud AI

## System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │              USER / FRONTEND                │
                    │         Streamlit Dashboard (:8501)          │
                    │   Upload X-ray → View Diagnosis + Triage    │
                    └────────────┬──────────────┬─────────────────┘
                                 │              ▲
                         Upload  │              │  Results
                                 ▼              │
┌────────────────────────────────────────────────────────────────────┐
│                        CLOUD PIPELINE                              │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐    │
│  │   AWS S3      │   │  AWS Lambda   │   │  GCP Cloud Function │    │
│  │              │──▶│              │──▶│                     │    │
│  │ xray-upload- │   │ xray-etl-    │   │ pneumonia-analyzer  │    │
│  │ mugesh       │   │ processor    │   │                     │    │
│  │              │   │              │   │ • AI Inference       │    │
│  │ • Store      │   │ • Read image │   │ • Risk Scoring       │    │
│  │   X-ray      │   │ • Validate   │   │ • Triage Decision    │    │
│  │   images     │   │ • Base64     │   │ • Forward to Azure   │    │
│  │              │   │ • Forward    │   │                     │    │
│  └──────────────┘   └──────────────┘   └─────────┬───────────┘    │
│                                                   │                │
│                                                   ▼                │
│                                        ┌─────────────────────┐    │
│                                        │  Azure Functions     │    │
│                                        │                     │    │
│                                        │ pneumonia-receiver-  │    │
│                                        │ mugesh1              │    │
│                                        │                     │    │
│                                        │ • Store diagnosis   │    │
│                                        │ • Medical records   │    │
│                                        │ • HIPAA compliance  │    │
│                                        └─────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

## Data Flow (Step by Step)

### Step 1: Image Upload → AWS S3
- User uploads chest X-ray to S3 bucket `xray-upload-mugesh`
- S3 event notification triggers Lambda function
- **Region**: us-east-1

### Step 2: ETL Processing → AWS Lambda
- Lambda reads the image from S3
- Validates file format (JPEG/PNG)
- Encodes image to base64
- Sends HTTP POST to GCP Cloud Function
- **Runtime**: Python 3.12, Timeout: 30s

### Step 3: AI Inference → GCP Cloud Functions
- Receives base64-encoded image
- Runs DenseNet-121 CNN inference
- Computes risk score and triage level
- Generates clinical recommendations
- Forwards complete diagnosis to Azure
- **Runtime**: Python 3.11, Region: europe-west1

### Step 4: Record Storage → Azure Functions
- Receives complete diagnosis JSON
- Adds timestamps and record IDs
- Stores medical record
- Returns confirmation
- **Runtime**: Python 3.11, Region: Central US

### Step 5: Dashboard Display → Streamlit
- Displays diagnosis, confidence, Grad-CAM heatmap
- Shows risk score, triage level, department routing
- Tracks multi-cloud pipeline status
- Maintains diagnosis history

## AI/ML Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       AI/ML PIPELINE                            │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Chest X-ray    │───▶│  Preprocessing  │                    │
│  │  Input Image    │    │  224×224 RGB     │                    │
│  │                 │    │  Normalize [0,1] │                    │
│  └─────────────────┘    └────────┬────────┘                    │
│                                  │                              │
│                                  ▼                              │
│                     ┌─────────────────────┐                    │
│                     │   DenseNet-121 CNN   │                    │
│                     │   (Transfer Learn)   │                    │
│                     │                     │                    │
│                     │  ImageNet weights   │                    │
│                     │  + Custom head      │                    │
│                     │  + Fine-tuned       │                    │
│                     └────────┬────────────┘                    │
│                              │                                  │
│                    ┌─────────┼─────────┐                       │
│                    ▼         ▼         ▼                       │
│              ┌──────────┐ ┌────────┐ ┌──────────────┐          │
│              │ Diagnosis│ │Grad-CAM│ │  XGBoost     │          │
│              │ Normal/  │ │Heatmap │ │  Risk Score  │          │
│              │ Pneumonia│ │ (XAI)  │ │              │          │
│              └────┬─────┘ └───┬────┘ └──────┬───────┘          │
│                   │           │              │                  │
│                   └───────────┼──────────────┘                  │
│                               ▼                                 │
│                    ┌─────────────────────┐                      │
│                    │   Triage Engine     │                      │
│                    │   CRITICAL/URGENT/  │                      │
│                    │   STANDARD/LOW      │                      │
│                    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Service Endpoints

| Service | URL |
|---|---|
| GCP Function | `https://pneumonia-analyzer-782668642236.europe-west1.run.app` |
| Azure Function | `https://pneumonia-receiver-mugesh1-dvahfjd8cca9gmcm.centralus-01.azurewebsites.net/api/savediagnosis` |
| Streamlit | `http://localhost:8501` |
| AWS S3 | `s3://xray-upload-mugesh` |

## Security Considerations

- AWS Lambda uses IAM LabRole with least-privilege S3 read access
- GCP Cloud Function allows unauthenticated access (for demo — restrict in production)
- Azure CORS configured for cross-origin requests
- No PHI/PII stored — demo uses synthetic patient IDs
- HIPAA compliance metadata fields included for production readiness
