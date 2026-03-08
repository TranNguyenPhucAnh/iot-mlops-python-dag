import joblib
import json
import os
import numpy as np

MODEL_PATH  = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "model.pkl")
SCALER_PATH = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "scaler.pkl")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Field aliases — support cả short (T/H/P/G) và full name ──────────────────
FIELD_ALIASES = {
    'temperature':    ['temperature', 'T'],
    'humidity':       ['humidity',    'H'],
    'pressure':       ['pressure',    'P'],
    'gas_resistance': ['gas_resistance', 'G'],
}

# ── Ngưỡng tuyệt đối — BME680 BSEC2 spec + điều kiện nhiệt đới ───────────────
THRESHOLDS = {
    'temp_critical':   35.0,   # °C — nguy hiểm, có thể cháy/nguồn nhiệt lớn
    'temp_high':       33.0,   # °C — nhiệt độ cao bất thường
    'temp_low':        20.0,   # °C — lạnh bất thường (VN)
    'humidity_high':   75.0,   # % — ẩm mốc, nấm
    'humidity_low':    30.0,   # % — khô bất thường
    'gas_very_low':    20000,  # Ω — không khí rất kém
    'gas_low':         50000,  # Ω — không khí kém
    'gas_high':        150000, # Ω — không khí tốt
}

# ── IAQ Score — công thức Bosch simplified ────────────────────────────────────
def calculate_iaq(gas_resistance: float, humidity: float) -> int:
    """
    IAQ 0-500 dựa trên gas_resistance và humidity compensation.
    Bosch BSEC2 simplified formula — không cần SDK.
    0-50: Excellent, 51-100: Good, 101-150: Lightly Polluted,
    151-200: Moderately Polluted, 201-300: Heavily Polluted, 301-500: Severely Polluted
    """
    # Humidity compensation: optimal là 40%, deviation làm giảm score
    hum_offset    = humidity - 40.0
    hum_score     = 0.25 if hum_offset > 0 else 0.75
    hum_component = hum_score / 100.0 * hum_offset * -1

    # Gas component: normalize gas_resistance về 0-1
    gas_ceil   = 250000.0  # Ω — upper bound clean air
    gas_floor  = 10000.0   # Ω — lower bound very polluted
    gas_norm   = (min(max(gas_resistance, gas_floor), gas_ceil) - gas_floor) / (gas_ceil - gas_floor)
    gas_component = gas_norm * 0.75

    # Kết hợp và scale về 0-500
    raw_score = (hum_component + gas_component) * 100
    iaq       = int(max(0, min(500, (1 - raw_score) * 500 / 100)))
    return iaq


def get_air_quality_label(iaq: int) -> str:
    if iaq <= 50:   return "Excellent"
    if iaq <= 100:  return "Good"
    if iaq <= 150:  return "Lightly Polluted"
    if iaq <= 200:  return "Moderately Polluted"
    if iaq <= 300:  return "Heavily Polluted"
    return "Severely Polluted"


# ── Decision score interpretation ────────────────────────────────────────────
def interpret_decision_score(score: float) -> str:
    """
    IsolationForest decision_function:
    > 0       : nằm sâu trong vùng normal, rất bình thường
    0 ~ -0.05 : gần boundary, hơi bất thường
    -0.05 ~ -0.1 : bất thường rõ ràng
    -0.1 ~ -0.2  : bất thường nghiêm trọng
    < -0.2    : outlier cực đoan
    """
    if score > 0:       return "Deeply normal — nằm sâu trong vùng bình thường"
    if score > -0.05:   return "Near boundary — hơi bất thường, cần theo dõi"
    if score > -0.10:   return "Anomalous — bất thường rõ ràng"
    if score > -0.20:   return "Severely anomalous — bất thường nghiêm trọng"
    return "Extreme outlier — cực kỳ bất thường"


# ── Correlation rules — domain knowledge encoding ────────────────────────────
def analyze_correlation(T, H, P, G):
    """
    Trả về (anomaly_type, severity, contributing_factors)
    dựa trên tương quan giữa 4 chỉ số.
    """
    th = THRESHOLDS
    factors = []

    # Collect individual factor flags
    temp_critical = T >= th['temp_critical']
    temp_high     = T >= th['temp_high']
    temp_low      = T <= th['temp_low']
    hum_high      = H >= th['humidity_high']
    hum_low       = H <= th['humidity_low']
    gas_very_low  = G <= th['gas_very_low']
    gas_low       = G <= th['gas_low']
    gas_high      = G >= th['gas_high']

    if temp_critical: factors.append("temperature_critical")
    elif temp_high:   factors.append("temperature_high")
    if temp_low:      factors.append("temperature_low")
    if hum_high:      factors.append("humidity_high")
    if hum_low:       factors.append("humidity_low")
    if gas_very_low:  factors.append("gas_very_low")
    elif gas_low:     factors.append("gas_low")
    if gas_high:      factors.append("gas_clean")

    # ── Correlation patterns ──────────────────────────────────────────────────
    # T tăng cao + G giảm mạnh → nguồn nhiệt/cháy
    if temp_critical and gas_very_low:
        return "FIRE_RISK", "CRITICAL", factors

    # T tăng nhẹ + G giảm → nấu ăn/hơi nước/thiết bị nhiệt
    if temp_high and gas_low:
        return "HEAT_SOURCE", "HIGH", factors

    # G giảm mạnh + T ổn định → hóa chất/VOC/khí gas rò rỉ
    if gas_very_low and not temp_high:
        return "CHEMICAL_VOC", "CRITICAL", factors

    # G giảm + T ổn định → ô nhiễm không khí nhẹ
    if gas_low and not temp_high:
        return "AIR_POLLUTION", "MEDIUM", factors

    # H tăng cao + G giảm → ẩm mốc/nấm
    if hum_high and gas_low:
        return "MOLD_RISK", "HIGH", factors

    # H tăng cao + P giảm → thời tiết xấu/mưa bão
    if hum_high and P < 1005:
        return "WEATHER_CHANGE", "LOW", factors

    # T thấp + H thấp → điều hòa quá lạnh hoặc môi trường khô
    if temp_low and hum_low:
        return "DRY_COLD", "LOW", factors

    # Không khớp pattern cụ thể nhưng vẫn có factor bất thường
    if factors and not (len(factors) == 1 and factors[0] == "gas_clean"):
        return "GENERAL_ANOMALY", "MEDIUM", factors

    # Bình thường
    return "NORMAL", "NONE", factors


# ── Parse helpers ─────────────────────────────────────────────────────────────
def _parse(body: dict) -> tuple:
    out = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in body:
                out[field] = float(body[alias])
                break
        else:
            raise ValueError(f"Missing field: {aliases[0]} (or alias {aliases[1]})")
    return out['temperature'], out['humidity'], out['pressure'], out['gas_resistance']


def _extract_body(event: dict) -> dict:
    raw = event.get("body", event)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


# ── Lambda handler ────────────────────────────────────────────────────────────
def handler(event, context):
    try:
        body           = _extract_body(event)
        T, H, P, G     = _parse(body)
        scaled         = scaler.transform([[T, H, P, G]])
        prediction     = model.predict(scaled)[0]
        score          = float(model.decision_function(scaled)[0])
        is_anomaly     = bool(prediction == -1)

        iaq            = calculate_iaq(G, H)
        air_quality    = get_air_quality_label(iaq)
        score_meaning  = interpret_decision_score(score)

        # Correlation analysis — chạy kể cả khi model nói NORMAL
        # vì domain rules có thể catch patterns model bỏ sót
        anomaly_type, severity, factors = analyze_correlation(T, H, P, G)

        # Nếu model nói anomaly nhưng correlation không match pattern cụ thể
        if is_anomaly and anomaly_type == "NORMAL":
            anomaly_type = "STATISTICAL_ANOMALY"
            severity     = "MEDIUM"

        # Nếu correlation detect FIRE_RISK/CHEMICAL nhưng model nói normal
        # → trust domain knowledge, override
        if anomaly_type in ("FIRE_RISK", "CHEMICAL_VOC") and not is_anomaly:
            is_anomaly = True

        return {
            "statusCode": 200,
            "body": json.dumps({
                "is_anomaly":        is_anomaly,
                "anomaly_type":      anomaly_type,
                "severity":          severity,
                "iaq_score":         iaq,
                "air_quality":       air_quality,
                "contributing_factors": factors,
                "decision_score":    round(score, 4),
                "score_meaning":     score_meaning,
            })
        }

    except ValueError as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
