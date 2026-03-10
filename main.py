import functions_framework
import json

@functions_framework.http
def predict_pneumonia(request):
    # In a real scenario, you would load a TensorFlow model here.
    # For this infrastructure demo, we simulate the AI.

    request_json = request.get_json(silent=True)
    filename = request_json.get("filename", "unknown")

    # Mock Logic: If filename has "virus", it's Pneumonia
    if "virus" in filename.lower() or "pneumonia" in filename.lower():
        result = "PNEUMONIA DETECTED"
        confidence = 0.98
    else:
        result = "NORMAL"
        confidence = 0.10

    return json.dumps({
        "diagnosis": result,
        "confidence": confidence,
        "processed_by": "Google Cloud AI"
    })
