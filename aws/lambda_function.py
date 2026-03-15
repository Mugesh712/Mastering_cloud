"""
AWS Lambda Function — ETL Preprocessor
========================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

WHAT THIS DOES:
  Step 1 of the 3-cloud pipeline.
  When a doctor uploads a chest X-ray to the S3 bucket,
  this Lambda function automatically wakes up and:
    1. Reads the image from S3
    2. Validates it is a real image file (jpg/png)
    3. Converts it to base64 text (so it can be sent over HTTP)
    4. Posts it to the GCP Cloud Function for AI analysis

HOW TO DEPLOY:
  AWS Console → Lambda → Create Function
    Name:    xray-etl-processor
    Runtime: Python 3.12
    Role:    LabRole (AWS Academy provides this)
    Timeout: 30 seconds

  Then add a Trigger:
    Type:   S3
    Bucket: xray-upload-mugesh
    Event:  s3:ObjectCreated:* (fires on every new upload)
"""

import json
import urllib.request
import urllib.parse
import base64
import boto3

# ─────────────────────────────────────────────────
# The URL of our GCP Cloud Function (Step 2).
# Lambda will forward the image here for AI analysis.
# ─────────────────────────────────────────────────
GCP_URL = "https://pneumonia-analyzer-782668642236.europe-west1.run.app"

# boto3 is the AWS Python SDK — it lets us read from S3
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Entry point — AWS calls this function automatically when an S3 upload happens.

    'event' contains details about the upload (bucket name, file name, file size).
    'context' contains Lambda runtime info (we don't use it directly).
    """
    try:
        # ── Step 1: Extract the file details from the S3 event ──────────────
        # The event is a JSON object; we pull out the first upload record
        record    = event['Records'][0]
        bucket    = record['s3']['bucket']['name']         # e.g. "xray-upload-mugesh"
        key       = urllib.parse.unquote_plus(record['s3']['object']['key']).strip()
        file_size = record['s3']['object'].get('size', 0)  # bytes

        print(f"[Lambda] New upload detected!")
        print(f"  Bucket : {bucket}")
        print(f"  File   : {key}")
        print(f"  Size   : {file_size} bytes")

        # ── Step 2: Validate the file format ────────────────────────────────
        # We only want to process medical image formats
        allowed = ('.jpg', '.jpeg', '.png', '.dcm')
        if not key.lower().endswith(allowed):
            print(f"[Lambda] Rejected: not a valid image format ({key})")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Invalid file type. Allowed: {allowed}'})
            }

        # ── Step 3: Read the image bytes from S3 ────────────────────────────
        print("[Lambda] Downloading image from S3...")
        s3_response  = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes  = s3_response['Body'].read()

        # Safety check: reject very large files (> 10 MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'File too large. Maximum size is 10 MB.'})
            }

        # ── Step 4: Convert image bytes → base64 string ─────────────────────
        # HTTP JSON can only carry text, not raw binary data.
        # base64 encoding turns bytes into a safe text string.
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # ── Step 5: Extract a patient ID from the filename ──────────────────
        # We use the filename (without extension) as the patient ID.
        # e.g.  "P001_scan.jpg"  →  patient_id = "P001_scan"
        patient_id = key.split('/')[-1].rsplit('.', 1)[0]

        # ── Step 6: Build the payload and POST to GCP ───────────────────────
        print(f"[Lambda] Forwarding to GCP for AI analysis...")

        payload = {
            'filename':     key,
            'patient_id':   patient_id,
            'bucket':       bucket,
            'image_base64': image_base64,   # the actual image, base64 encoded
            'source':       'AWS_S3_Lambda' # tells GCP where this came from
        }

        # Create an HTTP POST request
        request = urllib.request.Request(GCP_URL)
        request.add_header('Content-Type', 'application/json; charset=utf-8')
        json_bytes = json.dumps(payload).encode('utf-8')

        # Send the request and wait up to 60 seconds for a response
        gcp_response = urllib.request.urlopen(request, json_bytes, timeout=60)
        result       = gcp_response.read().decode('utf-8')

        print(f"[Lambda] GCP responded: {result[:200]}...")  # log first 200 chars

        # ── Step 7: Return success ───────────────────────────────────────────
        return {
            'statusCode': 200,
            'body': result
        }

    except Exception as error:
        error_msg = str(error)
        print(f"[Lambda] ERROR: {error_msg}")
        
        # ── Step 8: Send SNS Email Alert on Failure ──────────────────────
        try:
            import os
            sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
            
            if sns_topic_arn:
                sns = boto3.client('sns')
                message = (
                    f"🚨 PneumoCloud Pipeline Alert 🚨\n\n"
                    f"The AWS Lambda ETL processor failed to process an image.\n\n"
                    f"Error Details:\n{error_msg}\n\n"
                    f"Please check the AWS CloudWatch logs for more information."
                )
                
                sns.publish(
                    TopicArn=sns_topic_arn,
                    Subject="PneumoCloud AI - Pipeline Failure Alert",
                    Message=message
                )
                print("[Lambda] Alert email sent successfully via SNS")
            else:
                print("[Lambda] SNS_TOPIC_ARN not set. Skipping email alert.")
                
        except Exception as sns_error:
            print(f"[Lambda] Failed to send SNS alert: {str(sns_error)}")

        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
