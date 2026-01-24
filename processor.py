from airflow import DAG
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from airflow.providers.amazon.aws.hooks.sqs import SqsHook
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import json
import pandas as pd
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)

SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
S3_BUCKET = "iot-bme680-data-lake-prod"

def process_iot_data(**context):
    """Process IoT messages from SQS → S3 Bronze Lake"""
    # 1. Lấy messages từ SqsSensor XCom
    messages = context['ti'].xcom_pull(task_ids='wait_for_sqs_messages')
    
    if not messages:
        logger.info("No SQS messages to process")
        return

    processed_records = []
    
    for msg in messages:
        try:
            data = json.loads(msg['Body'])
            
            # Validation + defaults
            required = ['temperature', 'humidity', 'pressure', 'gas_resistance']
            if not all(data.get(k) for k in required):
                logger.warning(f"Invalid message: {msg['Body'][:100]}...")
                continue
                
            # 2. IAQ Score calculation (simplified Bosch formula)
            gas_res = float(data.get('gas_resistance', 0))
            hum = float(data.get('humidity', 0))
            
            iaq_score = max(0, min(500, (gas_res * 0.75) + (hum * 0.25)))  # 0-500 scale
            
            # Enriched record
            enriched = data.copy()
            enriched.update({
                'iaq_score': round(iaq_score, 2),
                'processed_at': datetime.utcnow().isoformat(),
                'partition_key': datetime.utcnow().strftime('year=%Y/month=%m/day=%d/hour=%H')
            })
            processed_records.append(enriched)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Parse error: {e}, msg: {msg['Body'][:100]}...")

    if not processed_records:
        logger.warning("No valid records after processing")
        return

    # 3. DataFrame + Parquet
    df = pd.DataFrame(processed_records)
    logger.info(f"Processing {len(df)} valid records")

    # Partition path
    now = datetime.utcnow()
    s3_path = f"bronze/bme680/year={now.year}/month={now.month:02d}/day={now.day:02d}/data_{now.strftime('%H%M%S')}.parquet"

    # Memory buffer Parquet
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')  # pyarrow nhanh hơn
    
    buffer.seek(0)

    # 4. S3 upload + delete SQS messages
    s3_hook = S3Hook(aws_conn_id=None)  # IRSA
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=s3_path,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # ✅ CRITICAL: Delete processed messages (SqsSensor KHÔNG auto-delete)
    sqs_hook = SqsHook(aws_conn_id=None)
    receipt_handles = [msg['ReceiptHandle'] for msg in messages]
    sqs_hook.delete_messages(SQS_QUEUE_URL, receipt_handles)
    
    logger.info(f"Saved {len(df)} records to s3://{S3_BUCKET}/{s3_path}")
