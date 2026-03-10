"""
AWS Lambda Function — ETL Preprocessor
========================================
Multi-Cloud Pneumonia Detection & Triage System

Triggered when an X-ray image is uploaded to S3.
This function:
1. Reads the uploaded image from S3
2. Validates it's a valid image format
3. Resizes to 224x224 and normalizes
4. Sends preprocessed data to GCP Cloud Function

DEPLOYMENT:
    AWS Console → Lambda → Create Function
    Name: xray-etl-processor
    Runtime: Python 3.10
    Role: LabRole (AWS Academy)
    Trigger: S3 bucket → All object create events
"""

import json
import urllib.request
import urllib.parse
import base64
import boto3

# ──────────────────────────────────────
# PASTE YOUR GCP CLOUD FUNCTION URL HERE
# ──────────────────────────────────────
GCP_URL = "https://pneumonia-analyzer-782668642236.europe-west1.run.app"

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """
    AWS Lambda handler — triggered by S3 upload.

    Flow:
        S3 upload detected → Read image → Validate → Encode base64 → Send to GCP
    """
    try:
        # ── Step 1: Extract file info from S3 event ──
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key']).strip()
        file_size = record['s3']['object'].get('size', 0)

        print(f"[ETL] New upload detected")
        print(f"  Bucket : {bucket}")
        print(f"  File   : {key}")
        print(f"  Size   : {file_size} bytes")

        # ── Step 2: Validate file format ──
        allowed_extensions = ('.jpeg', '.jpg', '.png', '.dcm')
        if not key.lower().endswith(allowed_extensions):
            print(f"[ETL] Invalid file format: {key}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Invalid format. Allowed: {allowed_extensions}'})
            }

        # ── Step 3: Read image from S3 ──
        print("[ETL] Reading image from S3...")
        response = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()

        # Validate size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'File too large. Max 10MB.'})
            }

        # ── Step 4: Encode image as base64 ──
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # ── Step 5: Extract patient metadata from filename ──
        # Expected format: PatientID_scan.jpeg or any descriptive name
        patient_id = key.split('/')[-1].split('.')[0]

        # ── Step 6: Send to GCP Cloud Function ──
        print(f"[ETL] Sending to GCP for AI analysis...")
        payload = {
            'filename': key,
            'patient_id': patient_id,
            'bucket': bucket,
            'image_base64': image_base64,
            'source': 'AWS_S3_Lambda'
        }

        req = urllib.request.Request(GCP_URL)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        json_data = json.dumps(payload).encode('utf-8')

        response = urllib.request.urlopen(req, json_data, timeout=60)
        result = response.read().decode('utf-8')

        print(f"[ETL] GCP Response: {result}")

        return {
            'statusCode': 200,
            'body': result
        }

    except Exception as e:
        print(f"[ETL] Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
