"""
IoT BME680 Data Pipeline v3.2
- Fixed: SQS batch delete limit (max 10)
- Fixed: Handle double-encoded JSON, SNS wrapping
- Fixed: Dynamic IAQ baseline thay vì hardcode 50000
- Fixed: Lưu gas_resistance_raw để feature engineering tự tính IAQ
"""

from airflow import DAG
from airflow.providers.amazon.aws.hooks.sqs import SqsHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import json
import pandas as pd
from datetime import datetime, timedelta
import io
import logging

logger = logging.getLogger(__name__)

SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
S3_BUCKET = "iot-bme680-data-lake-prod"

# IAQ config — không hardcode baseline ở đây nữa
IAQ_HUMIDITY_OPTIMAL_LOW  = 30
IAQ_HUMIDITY_OPTIMAL_HIGH = 60
IAQ_HUMIDITY_SCORE_FACTOR = 5
IAQ_MAX_SCORE             = 500


def parse_message_body(body_str):
    """
    Parse SQS message body với robust handling:
    - Double-encoded JSON
    - SNS notification wrapper
    - Nested Message field
    """
    data = json.loads(body_str)

    if isinstance(data, dict) and data.get('Type') == 'Notification':
        if 'Message' in data:
            data = json.loads(data['Message'])
            logger.debug("Unwrapped SNS notification")

    elif isinstance(data, dict) and 'Message' in data and isinstance(data['Message'], str):
        try:
            data = json.loads(data['Message'])
            logger.debug("Extracted nested Message field")
        except json.JSONDecodeError:
            pass

    elif isinstance(data, str):
        data = json.loads(data)
        logger.debug("Decoded double-encoded JSON")

    return data


def calc_iaq_score(gas_resistance: float, humidity: float, gas_baseline: float) -> float:
    """
    Tính IAQ score dựa trên dynamic baseline thay vì hardcode.

    gas_baseline nên là giá trị gas_resistance trong không khí sạch
    của chính sensor đó (tính từ data thực tế, ví dụ quantile 75).

    Trả về score trong [0, 500] — càng cao càng ô nhiễm.
    """
    # Gas score
    if gas_resistance >= gas_baseline:
        gas_score = 0.0
    else:
        gas_score = (1.0 - gas_resistance / gas_baseline) * 300.0

    # Humidity score
    if IAQ_HUMIDITY_OPTIMAL_LOW <= humidity <= IAQ_HUMIDITY_OPTIMAL_HIGH:
        hum_score = 0.0
    elif humidity < IAQ_HUMIDITY_OPTIMAL_LOW:
        hum_score = (IAQ_HUMIDITY_OPTIMAL_LOW - humidity) * IAQ_HUMIDITY_SCORE_FACTOR
    else:
        hum_score = (humidity - IAQ_HUMIDITY_OPTIMAL_HIGH) * IAQ_HUMIDITY_SCORE_FACTOR

    return round(min(IAQ_MAX_SCORE, gas_score + hum_score), 2)


def estimate_gas_baseline(records: list[dict]) -> float:
    """
    Ước tính baseline gas_resistance từ batch hiện tại.

    Dùng quantile 75 — giả định phần lớn thời gian không khí bình thường.
    Nếu batch quá ít record thì dùng fallback từ S3 history (nếu có),
    hoặc fallback cứng 80000 (thực tế BME680 trong phòng thường 80k-150k).
    """
    if len(records) >= 10:
        gas_values = [r['gas_resistance'] for r in records]
        baseline = float(pd.Series(gas_values).quantile(0.75))
        logger.info(f"📐 Dynamic gas baseline (q75 of batch): {baseline:.0f}")
        return baseline

    logger.warning(
        f"⚠️ Batch chỉ có {len(records)} records — không đủ để tính baseline. "
        "Dùng fallback 80000."
    )
    return 80_000.0


def delete_sqs_messages_batch(sqs_hook, receipt_handles):
    """Xóa SQS messages theo batch 10 (giới hạn AWS)"""
    if not receipt_handles:
        return

    deleted_count = 0
    for i in range(0, len(receipt_handles), 10):
        batch = receipt_handles[i:i + 10]
        try:
            sqs_hook.get_conn().delete_message_batch(
                QueueUrl=SQS_QUEUE_URL,
                Entries=[
                    {'Id': str(j), 'ReceiptHandle': rh}
                    for j, rh in enumerate(batch)
                ]
            )
            deleted_count += len(batch)
        except Exception as e:
            logger.error(f"Failed to delete batch {i // 10 + 1}: {e}")

    logger.info(f"✅ Deleted {deleted_count}/{len(receipt_handles)} messages from SQS")


def pull_and_process_sqs(**context):
    """Pull messages từ SQS, process, và save to S3"""
    sqs_hook = SqsHook(aws_conn_id='aws_default')

    # ── 1. Pull messages ──────────────────────────────────────────
    all_messages = []
    max_batches  = 10

    for batch_num in range(max_batches):
        response = sqs_hook.get_conn().receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,
            VisibilityTimeout=300
        )

        messages = response.get('Messages', [])
        if not messages:
            logger.info(f"No more messages after {batch_num} batches")
            break

        all_messages.extend(messages)
        logger.info(f"Batch {batch_num + 1}: pulled {len(messages)} messages")

        if len(all_messages) >= 50:
            break

    if not all_messages:
        logger.warning("No messages in SQS queue")
        raise AirflowSkipException("No messages to process")

    logger.info(f"Total messages pulled: {len(all_messages)}")

    # ── 2. Parse & validate (chưa tính IAQ) ──────────────────────
    raw_records    = []   # valid parsed records, chưa có IAQ
    invalid_count  = 0

    for idx, msg in enumerate(all_messages):
        try:
            data = parse_message_body(msg['Body'])

            if idx == 0:
                logger.info(f"Sample parsed data: {data}")
                logger.info(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

            if not isinstance(data, dict):
                invalid_count += 1
                logger.warning(f"Message {idx} is not a dict: {type(data)}")
                continue

            sensor_data = data.get('sensors', data)

            if not isinstance(sensor_data, dict):
                invalid_count += 1
                logger.warning(f"Message {idx}: 'sensors' field is not a dict")
                continue

            required = ['temperature', 'humidity', 'pressure', 'gas_resistance']
            missing  = [k for k in required if k not in sensor_data or sensor_data[k] is None]

            if missing:
                invalid_count += 1
                logger.warning(f"Message {idx} missing fields: {missing}")
                continue

            raw_records.append({
                'timestamp':       data.get('timestamp'),
                'device_id':       data.get('device_id'),
                'version':         data.get('version'),
                'temperature':     float(sensor_data['temperature']),
                'humidity':        float(sensor_data['humidity']),
                'pressure':        float(sensor_data['pressure']),
                'gas_resistance':  float(sensor_data['gas_resistance']),
                'processed_at':    datetime.utcnow().isoformat(),
                'partition_key':   datetime.utcnow().strftime('year=%Y/month=%m/day=%d/hour=%H')
            })

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            invalid_count += 1
            logger.error(f"Parse error on message {idx}: {e}")
            logger.debug(f"Raw body: {msg.get('Body', '')[:300]}")

    logger.info(f"Parsed: {len(raw_records)} valid, {invalid_count} invalid")

    if not raw_records:
        logger.warning("No valid records after parsing")
        delete_sqs_messages_batch(sqs_hook, [m['ReceiptHandle'] for m in all_messages])
        raise AirflowSkipException("No valid records")

    # ── 3. Tính IAQ với dynamic baseline ─────────────────────────
    gas_baseline = estimate_gas_baseline(raw_records)

    # Log sensor stats để debug
    gas_values = [r['gas_resistance'] for r in raw_records]
    logger.info(
        f"📊 gas_resistance — "
        f"min: {min(gas_values):.0f}, "
        f"max: {max(gas_values):.0f}, "
        f"mean: {sum(gas_values)/len(gas_values):.0f}, "
        f"baseline(q75): {gas_baseline:.0f}"
    )

    processed_records = []
    for r in raw_records:
        iaq = calc_iaq_score(r['gas_resistance'], r['humidity'], gas_baseline)
        processed_records.append({
            **r,
            'gas_resistance_raw': r['gas_resistance'],   # Giữ raw để feature engineering dùng
            'gas_baseline':       gas_baseline,           # Lưu baseline vào record để trace
            'iaq_score':          iaq,
        })

    # ── 4. Save to S3 ─────────────────────────────────────────────
    df  = pd.DataFrame(processed_records)
    now = datetime.utcnow()
    s3_path = (
        f"bronze/bme680/"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"hour={now.hour:02d}/"
        f"data_{now.strftime('%H%M%S')}.parquet"
    )

    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
        buffer.seek(0)

        s3_hook = S3Hook(aws_conn_id='aws_default')
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=s3_path,
            bucket_name=S3_BUCKET,
            replace=True
        )

        logger.info(f"✅ Saved {len(df)} records to s3://{S3_BUCKET}/{s3_path}")
        logger.info(f"📊 IAQ range: min={df['iaq_score'].min():.1f}, max={df['iaq_score'].max():.1f}")

        # Xóa message CHỈ sau khi S3 thành công
        delete_sqs_messages_batch(sqs_hook, [m['ReceiptHandle'] for m in all_messages])

    except Exception as e:
        logger.error(f"S3 upload failed — messages NOT deleted: {e}", exc_info=True)
        raise


with DAG(
    dag_id='iot_bme680_ingestion_pipeline_v3',
    start_date=datetime(2026, 1, 1),
    schedule='*/5 * * * *',
    catchup=False,
    max_active_runs=4,
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    },
    tags=['iot', 'bronze', 'sqs']
) as dag:

    process_sqs_to_s3 = PythonOperator(
        task_id='process_sqs_to_s3',
        python_callable=pull_and_process_sqs,
    )
