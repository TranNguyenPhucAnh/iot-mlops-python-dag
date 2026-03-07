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

def _extract_body(event):
    """
    Handle 3 cases:
      1. API Gateway wrapped:  event["body"] là JSON string
      2. Direct JSON:          event chính là payload (Lambda test console)
      3. Pre-parsed:           event["body"] đã là dict (một số proxy config)
    """
    raw = event.get("body", event)  # fallback sang event nếu không có "body"
    if isinstance(raw, str):
        return json.loads(raw)
    return raw  # đã là dict

def handler(event, context):
    try:
        body       = _extract_body(event)
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
    except ValueError as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
