"""
Generate synthetic BME680 SQS messages để train ML model.

Normal distribution dựa trên sensor ranges thực tế:
  temperature: mean=31.0, std=0.3
  humidity:    mean=65.4, std=1.0
  pressure:    mean=1004.0, std=0.8
  gas_resistance: mean=55000, std=5000
  iaq_score tính từ gas + humidity

Anomaly scenarios (~5% của total):
  1. smoke_spike    — iaq cao, gas thấp
  2. extreme_heat   — nhiệt độ cao bất thường
  3. sensor_drift   — gas_resistance giảm dần bất thường
"""
# python message_generator.py --total 5000 --anomaly-rate 0.05 --days-spread 7 Gửi 5000 messages, anomaly 5%, trải trong 7 ngày
import boto3
import json
import uuid
import random
import math
import time
import argparse
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────────
SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
DEVICE_ID     = "pi-edge-01"

# Normal ranges (từ sensor thực tế)
NORMAL = {
    'temperature':    (31.0, 0.4),   # (mean, std)
    'humidity':       (65.4, 1.2),
    'pressure':       (1004.0, 0.8),
    'gas_resistance': (55000, 4000),
}

# IAQ thresholds (khớp với DOMAIN_THRESHOLDS trong training DAG)
IAQ_HUMIDITY_OPTIMAL_LOW  = 30
IAQ_HUMIDITY_OPTIMAL_HIGH = 60
IAQ_HUMIDITY_SCORE_FACTOR = 5
IAQ_MAX_SCORE             = 500


def calc_iaq_score(gas_resistance: float, humidity: float, gas_baseline: float = 55000) -> float:
    if gas_resistance >= gas_baseline:
        gas_score = 0.0
    else:
        gas_score = (1.0 - gas_resistance / gas_baseline) * 300.0

    if IAQ_HUMIDITY_OPTIMAL_LOW <= humidity <= IAQ_HUMIDITY_OPTIMAL_HIGH:
        hum_score = 0.0
    elif humidity < IAQ_HUMIDITY_OPTIMAL_LOW:
        hum_score = (IAQ_HUMIDITY_OPTIMAL_LOW - humidity) * IAQ_HUMIDITY_SCORE_FACTOR
    else:
        hum_score = (humidity - IAQ_HUMIDITY_OPTIMAL_HIGH) * IAQ_HUMIDITY_SCORE_FACTOR

    return round(min(IAQ_MAX_SCORE, gas_score + hum_score), 2)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def generate_normal_reading(ts: datetime) -> dict:
    """Generate một reading bình thường."""
    temperature    = round(random.gauss(*NORMAL['temperature']), 2)
    humidity       = round(random.gauss(*NORMAL['humidity']), 2)
    pressure       = round(random.gauss(*NORMAL['pressure']), 2)
    gas_resistance = round(random.gauss(*NORMAL['gas_resistance']), 0)

    # Clamp về giá trị hợp lý
    temperature    = clamp(temperature, 28.0, 33.0)
    humidity       = clamp(humidity, 60.0, 70.0)
    pressure       = clamp(pressure, 1000.0, 1010.0)
    gas_resistance = clamp(gas_resistance, 30000, 70000)

    iaq = calc_iaq_score(gas_resistance, humidity)

    return _build_message(ts, temperature, humidity, pressure, gas_resistance, iaq)


def generate_anomaly_reading(ts: datetime, scenario: str) -> dict:
    """
    Generate anomaly reading theo scenario.
    Đảm bảo vượt DOMAIN_THRESHOLDS để is_anomaly=1 trong training DAG.
    """
    # Base values bình thường trước
    temperature    = round(random.gauss(*NORMAL['temperature']), 2)
    humidity       = round(random.gauss(*NORMAL['humidity']), 2)
    pressure       = round(random.gauss(*NORMAL['pressure']), 2)
    gas_resistance = round(random.gauss(*NORMAL['gas_resistance']), 0)

    if scenario == 'smoke_spike':
        # Khói/nấu ăn: gas thấp → iaq_score > 150
        gas_resistance = random.uniform(3000, 15000)   # gas thấp → iaq cao
        humidity       = random.uniform(63, 67)         # humidity bình thường

    elif scenario == 'extreme_heat':
        # Thiết bị quá nhiệt hoặc mở cửa trời nắng
        temperature = random.uniform(34.0, 36.0)

    elif scenario == 'high_iaq':
        # Không khí kém: kết hợp gas thấp + humidity cao
        gas_resistance = random.uniform(5000, 20000)
        humidity       = random.uniform(64, 67)

    elif scenario == 'sensor_drift':
        # Sensor drift: gas giảm dần không tự nhiên
        gas_resistance = random.uniform(3000, 8000)
        temperature    = random.uniform(31.5, 32.8)

    iaq = calc_iaq_score(gas_resistance, humidity)

    return _build_message(ts, temperature, humidity, pressure, gas_resistance, iaq)


def _build_message(ts, temperature, humidity, pressure, gas_resistance, iaq) -> dict:
    return {
        "device_id": DEVICE_ID,
        "timestamp": ts.isoformat(),
        "version":   "3.2",
        "sensors": {
            "temperature":    round(temperature, 2),
            "humidity":       round(humidity, 2),
            "pressure":       round(pressure, 2),
            "gas_resistance": round(gas_resistance, 0),
            "iaq_score":      iaq,
        }
    }


def send_batch(sqs_client, messages: list[dict]) -> int:
    """Gửi batch tối đa 10 messages, return số thành công."""
    entries = [
        {
            'Id':          str(i),
            'MessageBody': json.dumps(msg),
        }
        for i, msg in enumerate(messages)
    ]

    response = sqs_client.send_message_batch(
        QueueUrl=SQS_QUEUE_URL,
        Entries=entries,
    )

    failed = response.get('Failed', [])
    if failed:
        print(f"  ⚠️  {len(failed)} messages failed: {[f['Message'] for f in failed]}")

    return len(messages) - len(failed)


def generate_messages(
    total: int = 5000,
    anomaly_rate: float = 0.05,
    days_spread: int = 7,
    batch_size: int = 10,
    dry_run: bool = False,
):
    """
    Generate và push messages lên SQS.

    total       : tổng số messages
    anomaly_rate: tỷ lệ anomaly (0.05 = 5%)
    days_spread : trải timestamp trong bao nhiêu ngày (để partition đúng)
    dry_run     : chỉ print, không push SQS
    """
    n_anomaly = int(total * anomaly_rate)
    n_normal  = total - n_anomaly

    print(f"📦 Generating {total} messages:")
    print(f"   Normal  : {n_normal} ({1-anomaly_rate:.0%})")
    print(f"   Anomaly : {n_anomaly} ({anomaly_rate:.0%})")
    print(f"   Spread  : {days_spread} days")
    print()

    # Tạo timestamps trải đều trong days_spread ngày
    now        = datetime.utcnow()
    start_time = now - timedelta(days=days_spread)
    interval   = (days_spread * 24 * 3600) / total  # giây giữa mỗi message

    timestamps = [
        start_time + timedelta(seconds=i * interval)
        for i in range(total)
    ]
    random.shuffle(timestamps)  # shuffle để anomaly không tập trung 1 chỗ

    # Build tất cả messages
    anomaly_scenarios = ['smoke_spike', 'extreme_heat', 'high_iaq', 'sensor_drift']
    all_messages = []

    # Normal messages
    for i in range(n_normal):
        all_messages.append(('normal', generate_normal_reading(timestamps[i])))

    # Anomaly messages
    for i in range(n_anomaly):
        scenario = random.choice(anomaly_scenarios)
        all_messages.append((scenario, generate_anomaly_reading(timestamps[n_normal + i], scenario)))

    # Shuffle lại để anomaly phân bổ đều theo thời gian
    random.shuffle(all_messages)

    # Stats preview
    iaq_scores = [m['sensors']['iaq_score'] for _, m in all_messages]
    temps      = [m['sensors']['temperature'] for _, m in all_messages]
    print(f"📊 Preview stats:")
    print(f"   iaq_score  : min={min(iaq_scores):.1f}, mean={sum(iaq_scores)/len(iaq_scores):.1f}, max={max(iaq_scores):.1f}")
    print(f"   temperature: min={min(temps):.1f}, mean={sum(temps)/len(temps):.1f}, max={max(temps):.1f}")

    # Verify anomaly labels sẽ được detect đúng
    DOMAIN_THRESHOLDS = {
        'iaq_score_max':   150.0,
        'temperature_max':  32.0,
        'temperature_min':  29.5,
        'humidity_max':     68.0,
        'humidity_min':     62.0,
    }
    detectable = sum(
        1 for _, m in all_messages
        if m['sensors']['iaq_score']   > DOMAIN_THRESHOLDS['iaq_score_max']
        or m['sensors']['temperature'] > DOMAIN_THRESHOLDS['temperature_max']
        or m['sensors']['temperature'] < DOMAIN_THRESHOLDS['temperature_min']
        or m['sensors']['humidity']    > DOMAIN_THRESHOLDS['humidity_max']
        or m['sensors']['humidity']    < DOMAIN_THRESHOLDS['humidity_min']
    )
    print(f"   Detectable anomalies (sẽ được label=1 trong training DAG): {detectable}/{total} = {detectable/total:.2%}")
    print()

    if dry_run:
        print("DRY RUN — không push SQS. Sample message:")
        print(json.dumps(all_messages[0][1], indent=2))
        return

    # Push lên SQS
    sqs_client   = boto3.client('sqs', region_name='ap-southeast-1')
    total_sent   = 0
    just_messages = [m for _, m in all_messages]

    for i in range(0, len(just_messages), batch_size):
        batch   = just_messages[i:i + batch_size]
        sent    = send_batch(sqs_client, batch)
        total_sent += sent

        if (i // batch_size + 1) % 50 == 0:
            print(f"  ✅ {total_sent}/{total} messages sent...")
        time.sleep(0.05)  # tránh throttle

    print(f"\n🎉 Done — {total_sent}/{total} messages pushed to SQS")
    print(f"   Queue: {SQS_QUEUE_URL}")
    print(f"\nNext steps:")
    print(f"  1. Trigger ingestion DAG với drain_mode=True, max_batches=100")
    print(f"  2. Trigger transform DAG với backfill_from_date='{start_time.strftime('%Y-%m-%d %H:%M')}'")
    print(f"  3. Trigger training DAG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic BME680 SQS messages')
    parser.add_argument('--total',        type=int,   default=5000,  help='Total messages (default: 5000)')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,  help='Anomaly rate 0-1 (default: 0.05 = 5%%)')
    parser.add_argument('--days-spread',  type=int,   default=7,     help='Spread over N days (default: 7)')
    parser.add_argument('--dry-run',      action='store_true',        help='Preview only, do not push to SQS')
    args = parser.parse_args()

    generate_messages(
        total        = args.total,
        anomaly_rate = args.anomaly_rate,
        days_spread  = args.days_spread,
        dry_run      = args.dry_run,
    )
