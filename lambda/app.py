import joblib, json, os
import numpy as np

MODEL_PATH  = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "model.pkl")
SCALER_PATH = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def handler(event, context):
    body = json.loads(event["body"])
    raw = [[
        float(body["T"]),  # temperature
        float(body["H"]),  # humidity
        float(body["P"]),  # pressure
        float(body["G"]),  # gas_resistance
    ]]
    scaled     = scaler.transform(raw)
    prediction = model.predict(scaled)[0]      # -1 or 1
    score      = float(model.decision_function(scaled)[0])  # <0 = anomaly

    return {
        "statusCode": 200,
        "body": json.dumps({
            "is_anomaly":     bool(prediction == -1),
            "decision_score": round(score, 4),
            "label":          "ANOMALY" if prediction == -1 else "NORMAL"
        })
    }
