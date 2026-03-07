import joblib, json, os
import numpy as np

MODEL_PATH  = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "model.pkl")
SCALER_PATH = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "scaler.pkl")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Support cả short name lẫn full name, full name ưu tiên
FIELD_ALIASES = {
    'temperature':    ['temperature', 'T'],
    'humidity':       ['humidity',    'H'],
    'pressure':       ['pressure',    'P'],
    'gas_resistance': ['gas_resistance', 'G'],
}

def _parse(body):
    out = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in body:
                out[field] = float(body[alias])
                break
        else:
            raise ValueError(f"Missing: {aliases[0]} or {aliases[1]}")
    return out['temperature'], out['humidity'], out['pressure'], out['gas_resistance']

def handler(event, context):
    body       = json.loads(event["body"])
    T, H, P, G = _parse(body)
    scaled     = scaler.transform([[T, H, P, G]])
    prediction = model.predict(scaled)[0]
    score      = float(model.decision_function(scaled)[0])
    return {
        "statusCode": 200,
        "body": json.dumps({
            "is_anomaly":     bool(prediction == -1),
            "decision_score": round(score, 4),
            "label":          "ANOMALY" if prediction == -1 else "NORMAL"
        })
    }
