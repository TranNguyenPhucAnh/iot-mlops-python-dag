"""
IoT BME680 Data Pipeline v4.0
- Feature toggles via Airflow Params (configurable on UI trigger)
- Toggle: partition mode (message timestamp vs process time)
- Toggle: SQS URL, max_batches, wait_time_seconds
- Toggle: drain mode (high throughput để drain queue nhanh)
"""

from airflow import DAG
from airflow.providers.amazon.aws.hooks.sqs import SqsHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
import json
import pandas as pd
from datetime import datetime, timedelta
import io
import logging

logger = logging.getLogger(__name__)

# ==================== Default Config ====================
DEFAULT_SQS_URL      = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
DEFAULT_MAX_BATCHES  = 10
DEFAULT_WAIT_SECONDS = 20
S3_BUCKET            = "iot-bme680-data-lake-prod"

IAQ_HUMIDITY_OPTIMAL_LOW  = 30
IAQ_HUMIDITY_OPTIMAL_HIGH = 60
IAQ_HUMIDITY_SCORE_FACTOR = 5
IAQ_MAX_SCORE             = 500


# ==================== DAG Params ====================
dag_params = {
    # SQS source — đổi sang queue ngoài terraform khi cần drain
    "sqs_queue_url": Param(
        default=DEFAULT_SQS_URL,
        type="string",
        description="SQS Queue URL để pull messages. Đổi sang queue khác khi cần drain pre-loaded data.",
    ),

    # Throughput control
    "max_batches": Param(
        default=DEFAULT_MAX_BATCHES,
        type="integer",
        description="Số batches tối đa mỗi run (mỗi batch = 10 messages). Tăng lên 100+ khi drain queue lớn.",
        minimum=1,
        maximum=500,
    ),
    "wait_time_seconds": Param(
        default=DEFAULT_WAIT_SECONDS,
        type="integer",
        description="SQS long polling wait time (giây). Đặt 0 khi drain queue có sẵn nhiều messages.",
        minimum=0,
        maximum=20,
    ),

    # Drain mode — bật để tối ưu throughput tối đa
    "drain_mode": Param(
        default=False,
        type="boolean",
        description=(
            "Bật drain mode: tự động set wait_time=0, tăng batch processing. "
            "Dùng khi cần drain nhanh queue lớn (ví dụ 10,000 messages pre-loaded)."
        ),
    ),

    # Partition mode toggle
    "partition_by_message_time": Param(
        default=True,
        type="boolean",
        description=(
            "True = partition theo timestamp của message lúc push lên SQS (dùng khi drain historical data). "
            "False = partition theo thời điểm Airflow process (dùng cho real-time ingestion bình thường)."
        ),
    ),
}


# ==================== Helpers ====================

def parse_message_body(body_str):
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


def estimate_gas_baseline(records: list[dict]) -> float:
    if len(records) >= 10:
        gas_values = [r['gas_resistance'] for r in records]
        baseline = float(pd.Series(gas_values).quantile(0.75))
        logger.info(f"Dynamic gas baseline (q75 of batch): {baseline:.0f}")
        return baseline
    logger.warning(f"Batch chỉ có {len(records)} records — dùng fallback 80000.")
    return 80_000.0


def delete_sqs_messages_batch(sqs_hook, receipt_handles, sqs_queue_url):
    if not receipt_handles:
        return
    deleted_count = 0
    for i in range(0, len(receipt_handles), 10):
        batch = receipt_handles[i:i + 10]
        try:
            sqs_hook.get_conn().delete_message_batch(
                QueueUrl=sqs_queue_url,
                Entries=[
                    {'Id': str(j), 'ReceiptHandle': rh}
                    for j, rh in enumerate(batch)
                ]
            )
            deleted_count += len(batch)
        except Exception as e:
            logger.error(f"Failed to delete batch {i // 10 + 1}: {e}")
    logger.info(f"Deleted {deleted_count}/{len(receipt_handles)} messages from SQS")


def build_partition_key(timestamp_str: str, use_message_time: bool) -> str:
    """
    Tính partition key dựa trên toggle.

    use_message_time=True  → dùng timestamp trong message (historical data)
    use_message_time=False → dùng datetime.utcnow() (real-time ingestion)
    """
    if use_message_time:
        try:
            return pd.to_datetime(timestamp_str).strftime('year=%Y/month=%m/day=%d/hour=%H')
        except Exception as e:
            logger.warning(f"Cannot parse message timestamp '{timestamp_str}': {e}. Fallback to process time.")

    now = datetime.utcnow()
    return now.strftime('year=%Y/month=%m/day=%d/hour=%H')


def build_s3_path(partition_key: str) -> str:
    return f"bronze/bme680/{partition_key}/data_{datetime.utcnow().strftime('%H%M%S%f')}.parquet"


# ==================== Main Task ====================

def pull_and_process_sqs(**context):
    # ── Đọc params từ DAG run ──────────────────────────────────────
    params                  = context['params']
    sqs_queue_url           = params['sqs_queue_url']
    max_batches             = params['max_batches']
    wait_time_seconds       = params['wait_time_seconds']
    drain_mode              = params['drain_mode']
    partition_by_msg_time   = params['partition_by_message_time']

    # Drain mode override
    if drain_mode:
        wait_time_seconds = 0
        logger.info(
            f"DRAIN MODE ON — wait_time forced to 0, "
            f"max_batches={max_batches}, queue={sqs_queue_url}"
        )
    else:
        logger.info(
            f"Normal mode — max_batches={max_batches}, "
            f"wait_time={wait_time_seconds}s, "
            f"partition_by_message_time={partition_by_msg_time}, "
            f"queue={sqs_queue_url}"
        )

    sqs_hook = SqsHook(aws_conn_id='aws_default')

    # ── 1. Pull messages ──────────────────────────────────────────
    all_messages = []

    for batch_num in range(max_batches):
        response = sqs_hook.get_conn().receive_message(
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=300
        )

        messages = response.get('Messages', [])
        if not messages:
            logger.info(f"No more messages after {batch_num} batches")
            break

        all_messages.extend(messages)
        logger.info(f"Batch {batch_num + 1}: pulled {len(messages)} messages, total={len(all_messages)}")

    if not all_messages:
        logger.warning("No messages in SQS queue")
        raise AirflowSkipException("No messages to process")

    logger.info(f"Total messages pulled: {len(all_messages)}")

    # ── 2. Parse & validate ───────────────────────────────────────
    raw_records   = []
    invalid_count = 0

    for idx, msg in enumerate(all_messages):
        try:
            data = parse_message_body(msg['Body'])

            if idx == 0:
                logger.info(f"Sample parsed data: {data}")

            if not isinstance(data, dict):
                invalid_count += 1
                continue

            sensor_data = data.get('sensors', data)
            if not isinstance(sensor_data, dict):
                invalid_count += 1
                continue

            required = ['temperature', 'humidity', 'pressure', 'gas_resistance']
            missing  = [k for k in required if k not in sensor_data or sensor_data[k] is None]
            if missing:
                invalid_count += 1
                logger.warning(f"Message {idx} missing fields: {missing}")
                continue

            timestamp_str   = data.get('timestamp')
            partition_key   = build_partition_key(timestamp_str, partition_by_msg_time)

            raw_records.append({
                'timestamp':        timestamp_str,
                'device_id':        data.get('device_id'),
                'version':          data.get('version'),
                'temperature':      float(sensor_data['temperature']),
                'humidity':         float(sensor_data['humidity']),
                'pressure':         float(sensor_data['pressure']),
                'gas_resistance':   float(sensor_data['gas_resistance']),
                'processed_at':     datetime.utcnow().isoformat(),
                'partition_key':    partition_key,
            })

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            invalid_count += 1
            logger.error(f"Parse error on message {idx}: {e}")

    logger.info(f"Parsed: {len(raw_records)} valid, {invalid_count} invalid")

    if not raw_records:
        logger.warning("No valid records after parsing")
        delete_sqs_messages_batch(sqs_hook, [m['ReceiptHandle'] for m in all_messages], sqs_queue_url)
        raise AirflowSkipException("No valid records")

    # ── 3. Tính IAQ ───────────────────────────────────────────────
    gas_baseline = estimate_gas_baseline(raw_records)
    gas_values   = [r['gas_resistance'] for r in raw_records]
    logger.info(
        f"gas_resistance — min: {min(gas_values):.0f}, "
        f"max: {max(gas_values):.0f}, "
        f"mean: {sum(gas_values)/len(gas_values):.0f}, "
        f"baseline(q75): {gas_baseline:.0f}"
    )

    processed_records = []
    for r in raw_records:
        iaq = calc_iaq_score(r['gas_resistance'], r['humidity'], gas_baseline)
        processed_records.append({
            **r,
            'gas_resistance_raw': r['gas_resistance'],
            'gas_baseline':       gas_baseline,
            'iaq_score':          iaq,
        })

    # ── 4. Group theo partition rồi save S3 ───────────────────────
    # Khi drain historical data, records có thể thuộc nhiều partitions khác nhau
    # Group lại để mỗi partition được save vào đúng path
    df = pd.DataFrame(processed_records)

    partitions_written = []
    for partition_key, group_df in df.groupby('partition_key'):
        s3_path = build_s3_path(y)
        try:
            buffer = io.BytesIO()
            group_df.drop(columns=['partition_key']).to_parquet(
                buffer, index=False, engine='pyarrow', compression='snappy'
            )
            buffer.seek(0)

            s3_hook = S3Hook(aws_conn_id='aws_default')
            s3_hook.load_file_obj(
                file_obj=buffer,
                key=s3_path,
                bucket_name=S3_BUCKET,
                replace=True
            )

            logger.info(f"Saved {len(group_df)} records → s3://{S3_BUCKET}/{s3_path}")
            partitions_written.append(partition_key)

        except Exception as e:
            logger.error(f"S3 upload failed for partition {partition_key}: {e}", exc_info=True)
            raise

    logger.info(
        f"Done — {len(df)} records across {len(partitions_written)} partitions. "
        f"IAQ range: min={df['iaq_score'].min():.1f}, max={df['iaq_score'].max():.1f}"
    )

    # Xóa messages CHỈ sau khi tất cả partitions S3 thành công
    delete_sqs_messages_batch(sqs_hook, [m['ReceiptHandle'] for m in all_messages], sqs_queue_url)


# ==================== DAG Definition ====================

with DAG(
    dag_id='iot_bme680_ingestion_pipeline_v4',
    start_date=datetime(2026, 1, 1),
    schedule='*/5 * * * *',
    catchup=False,
    max_active_runs=4,
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    },
    params=dag_params,
    tags=['iot', 'bronze', 'sqs'],
    doc_md="""
## IoT BME680 Ingestion Pipeline v4

### Feature Toggles (configurable khi trigger DAG)

| Param | Default | Mô tả |
|---|---|---|
| `sqs_queue_url` | production queue | Đổi sang queue khác để drain pre-loaded data |
| `max_batches` | 10 | Tăng lên 100-500 để drain nhanh |
| `wait_time_seconds` | 20 | Đặt 0 khi queue đã có sẵn nhiều messages |
| `drain_mode` | False | Bật để auto-override wait_time=0 |
| `partition_by_message_time` | True | True=partition theo message timestamp, False=process time |

### Drain 10,000 messages nhanh

1. Trigger DAG manually
2. Set `drain_mode=True`, `max_batches=100`
3. Set `sqs_queue_url` sang pre-loaded queue
4. Set `max_active_runs=8` nếu cần nhanh hơn nữa
    """,
) as dag:

    process_sqs_to_s3 = PythonOperator(
        task_id='process_sqs_to_s3',
        python_callable=pull_and_process_sqs,
    )
