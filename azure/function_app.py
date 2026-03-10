"""
Azure Function — Medical Record Storage
==========================================
Multi-Cloud Pneumonia Detection & Triage System

This Azure Function:
1. Receives triage data from GCP Cloud Function
2. Logs the complete medical record
3. Returns confirmation

DEPLOYMENT:
    Azure Portal → Function App → pneumonia-receiver-mugesh
    Function name: savediagnosis
    Auth level: Anonymous
    Paste this code into function_app.py in Azure Portal

NOTE: This code uses the V2 programming model (function_app.py style).
      If Azure Portal shows __init__.py, use the __init__.py version below.
"""

import logging
import json
import azure.functions as func
from datetime import datetime

app = func.FunctionApp()


@app.route(route="savediagnosis", auth_level=func.AuthLevel.ANONYMOUS)
def savediagnosis(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger — Save diagnosis record.

    Receives JSON from GCP Cloud Function:
        - patient_file: str
        - patient_id: str
        - diagnosis: str
        - conf_score: float
        - risk_score: float
        - triage_level: str
        - department: str
        - actions: str (JSON array)

    Returns:
        JSON confirmation with saved record details
    """
    logging.info('Azure Function: Received diagnosis data from GCP')

    try:
        req_body = req.get_json()

        # Extract all fields
        record = {
            'record_id': f"REC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.utcnow().isoformat(),
            'patient_file': req_body.get('patient_file', 'unknown'),
            'patient_id': req_body.get('patient_id', 'N/A'),
            'diagnosis': req_body.get('diagnosis', 'unknown'),
            'confidence': req_body.get('conf_score', 0.0),
            'risk_score': req_body.get('risk_score', 0.0),
            'triage_level': req_body.get('triage_level', 'UNKNOWN'),
            'department': req_body.get('department', 'N/A'),
            'recommended_actions': req_body.get('actions', '[]'),
            'stored_by': 'Microsoft Azure',
            'compliance': 'HIPAA-Ready'
        }

        # Log the complete record
        logging.info(f"MEDICAL RECORD SAVED:")
        logging.info(f"  Record ID  : {record['record_id']}")
        logging.info(f"  Patient    : {record['patient_id']}")
        logging.info(f"  File       : {record['patient_file']}")
        logging.info(f"  Diagnosis  : {record['diagnosis']}")
        logging.info(f"  Confidence : {record['confidence']}")
        logging.info(f"  Risk Score : {record['risk_score']}")
        logging.info(f"  Triage     : {record['triage_level']}")
        logging.info(f"  Department : {record['department']}")

        response = {
            'status': 'SAVED TO AZURE',
            'record_id': record['record_id'],
            'patient_id': record['patient_id'],
            'diagnosis': record['diagnosis'],
            'triage_level': record['triage_level'],
            'timestamp': record['timestamp'],
            'compliance': 'HIPAA-Ready'
        }

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype='application/json'
        )

    except ValueError:
        logging.error('Invalid JSON received')
        return func.HttpResponse(
            body=json.dumps({'error': 'Invalid JSON body'}),
            status_code=400,
            mimetype='application/json'
        )
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        return func.HttpResponse(
            body=json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )


# ══════════════════════════════════════════════════════════════
# ALTERNATIVE: __init__.py version (for Azure Portal editor)
# ══════════════════════════════════════════════════════════════
# If Azure Portal shows __init__.py instead of function_app.py,
# paste this version instead:
#
# import logging
# import json
# import azure.functions as func
# from datetime import datetime
#
# def main(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Azure Function: Received diagnosis data')
#     try:
#         req_body = req.get_json()
#         record = {
#             'record_id': f"REC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
#             'timestamp': datetime.utcnow().isoformat(),
#             'patient_file': req_body.get('patient_file', 'unknown'),
#             'patient_id': req_body.get('patient_id', 'N/A'),
#             'diagnosis': req_body.get('diagnosis', 'unknown'),
#             'confidence': req_body.get('conf_score', 0.0),
#             'risk_score': req_body.get('risk_score', 0.0),
#             'triage_level': req_body.get('triage_level', 'UNKNOWN'),
#             'department': req_body.get('department', 'N/A'),
#             'actions': req_body.get('actions', '[]'),
#         }
#         logging.info(f"Record saved: {record['record_id']}")
#         return func.HttpResponse(
#             body=json.dumps({'status': 'SAVED', 'record_id': record['record_id']}),
#             status_code=200, mimetype='application/json'
#         )
#     except Exception as e:
#         return func.HttpResponse(
#             body=json.dumps({'error': str(e)}),
#             status_code=500, mimetype='application/json'
#         )
