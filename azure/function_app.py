"""
Azure Function — Medical Record Storage
=========================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

WHAT THIS DOES:
  Step 3 (final step) of the 3-cloud pipeline.
  Receives the complete diagnosis result from GCP,
  wraps it with a unique Record ID and timestamp,
  logs it, and returns a confirmation.

  In a real hospital system, this is where you'd write
  to a database (Azure SQL, Cosmos DB, Table Storage).
  For this project we log the record so it appears in
  Azure's Application Insights / Function logs.

HOW TO DEPLOY:
  Option A — Azure CLI:
    az functionapp create ...
    func azure functionapp publish pneumonia-receiver-mugesh1 --python

  Option B — Azure Portal:
    Go to your Function App → Functions → + Add
    Paste this code into function_app.py

NOTE: This uses Azure Functions Python V2 model (function_app.py style).
"""

import logging
import json
import azure.functions as func
from datetime import datetime

# Create the Azure Function App object
# (Azure uses this to discover and register your functions)
app = func.FunctionApp()


@app.route(route="savediagnosis", auth_level=func.AuthLevel.ANONYMOUS)
def savediagnosis(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered Azure Function — endpoint: /api/savediagnosis

    Receives a POST request from the GCP Cloud Function with JSON:
        patient_file  — filename of the uploaded X-ray
        patient_id    — unique patient identifier
        diagnosis     — 'PNEUMONIA DETECTED' or 'NORMAL'
        conf_score    — AI confidence (0.0–1.0)
        risk_score    — computed risk score (0.0–1.0)
        triage_level  — 'CRITICAL' / 'URGENT' / 'STANDARD' / 'LOW'
        department    — clinical department to route patient to
        actions       — JSON array string of recommended clinical actions

    Returns:
        JSON confirmation containing the saved record ID and timestamp
    """
    logging.info('[Azure] Incoming diagnosis record from GCP pipeline')

    try:
        # ── Step 1: Parse the incoming JSON body ──────────────────────────
        body = req.get_json()

        # ── Step 2: Build a complete medical record ───────────────────────
        # Generate a unique Record ID using the current UTC timestamp
        # e.g. "REC-20260315094250"
        record_id = f"REC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        timestamp = datetime.utcnow().isoformat() + 'Z'   # ISO 8601 format

        record = {
            'record_id':           record_id,
            'timestamp':           timestamp,
            'patient_file':        body.get('patient_file', 'unknown'),
            'patient_id':          body.get('patient_id', 'N/A'),
            'diagnosis':           body.get('diagnosis', 'UNKNOWN'),
            'confidence':          body.get('conf_score', 0.0),
            'risk_score':          body.get('risk_score', 0.0),
            'triage_level':        body.get('triage_level', 'UNKNOWN'),
            'department':          body.get('department', 'N/A'),
            'recommended_actions': body.get('actions', '[]'),
            'stored_by':           'Microsoft Azure Functions',
            'compliance':          'HIPAA-Ready',   # metadata for audit purposes
        }

        # ── Step 3: Log the record (replaces DB write in this demo) ───────
        logging.info('─── MEDICAL RECORD SAVED ───────────────────────')
        logging.info(f"  Record ID  : {record['record_id']}")
        logging.info(f"  Timestamp  : {record['timestamp']}")
        logging.info(f"  Patient ID : {record['patient_id']}")
        logging.info(f"  File       : {record['patient_file']}")
        logging.info(f"  Diagnosis  : {record['diagnosis']}")
        logging.info(f"  Confidence : {record['confidence']}")
        logging.info(f"  Risk Score : {record['risk_score']}")
        logging.info(f"  Triage     : {record['triage_level']}")
        logging.info(f"  Department : {record['department']}")
        logging.info('────────────────────────────────────────────────')

        # ── Step 4: Return confirmation to GCP ────────────────────────────
        response = {
            'status':       'SAVED TO AZURE',
            'record_id':    record['record_id'],
            'patient_id':   record['patient_id'],
            'diagnosis':    record['diagnosis'],
            'triage_level': record['triage_level'],
            'timestamp':    record['timestamp'],
            'compliance':   'HIPAA-Ready',
        }

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype='application/json'
        )

    except ValueError:
        # Malformed JSON in the request body
        logging.error('[Azure] Invalid JSON received')
        return func.HttpResponse(
            body=json.dumps({'error': 'Invalid JSON body — check Content-Type header'}),
            status_code=400,
            mimetype='application/json'
        )

    except Exception as error:
        logging.error(f'[Azure] Unexpected error: {str(error)}')
        return func.HttpResponse(
            body=json.dumps({'error': str(error)}),
            status_code=500,
            mimetype='application/json'
        )
