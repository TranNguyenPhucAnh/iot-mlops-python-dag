"""
IoT BME680 Data Pipeline v3.1 (Bug Fixes)
- Fixed: SQS batch delete limit (max 10)
- Fixed: Handle double-encoded JSON, SNS wrapping
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

def parse_message_body(body_str):
    """
    Parse SQS message body with robust handling:
    - Double-encoded JSON
    - SNS notification wrapper
    - Nested Message field
    """
    data = json.loads(body_str)
    
    # Handle SNS notification wrapper
    if isinstance(data, dict) and data.get('Type') == 'Notification':
        if 'Message' in data:
            data = json.loads(data['Message'])
            logger.debug("Unwrapped SNS notification")
    
    # Handle nested Message field
    elif isinstance(data, dict) and 'Message' in data and isinstance(data['Message'], str):
        try:
            data = json.loads(data['Message'])
            logger.debug("Extracted nested Message field")
        except json.JSONDecodeError:
            pass  # Message field is not JSON, keep original data
    
    # Handle double-encoded JSON
    elif isinstance(data, str):
        data = json.loads(data)
        logger.debug("Decoded double-encoded JSON")
    
    return data

def delete_sqs_messages_batch(sqs_hook, receipt_handles):
    """Delete SQS messages in batches of 10 (AWS limit)"""
    if not receipt_handles:
        return
    
    deleted_count = 0
    for i in range(0, len(receipt_handles), 10):
        batch = receipt_handles[i:i+10]
        try:
            sqs_hook.get_conn().delete_message_batch(
                QueueUrl=SQS_QUEUE_URL,
                Entries=[{'Id': str(j), 'ReceiptHandle': rh} for j, rh in enumerate(batch)]
            )
            deleted_count += len(batch)
        except Exception as e:
            logger.error(f"Failed to delete batch {i//10 + 1}: {e}")
    
    logger.info(f"✅ Deleted {deleted_count}/{len(receipt_handles)} messages from SQS")

def pull_and_process_sqs(**context):
    """Pull messages from SQS, process, and save to S3"""
    sqs_hook = SqsHook(aws_conn_id='aws_default')
    
    # 1. Pull messages from SQS
    all_messages = []
    max_batches = 10
    
    for batch_num in range(max_batches):
        response = sqs_hook.get_conn().receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5,  # Reduced from 20 for faster batching
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
    
    # 2. Process messages
    processed_records = []
    invalid_count = 0
    
    for idx, msg in enumerate(all_messages):
        try:
            # Parse with robust handling
            data = parse_message_body(msg['Body'])
            
            # Debug first message
            if idx == 0:
                logger.info(f"Sample parsed data: {data}")
                logger.info(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            
            # Validate required fields
            required = ['temperature', 'humidity', 'pressure', 'gas_resistance']
            
            if not isinstance(data, dict):
                invalid_count += 1
                logger.warning(f"Message {idx} is not a dict: {type(data)}")
                continue
            
            missing = [k for k in required if k not in data or data[k] is None]
            if missing:
                invalid_count += 1
                logger.warning(f"Message {idx} missing fields: {missing}")
                logger.debug(f"Available keys: {list(data.keys())}")
                continue
            
            # Calculate IAQ Score (simplified)
            gas_res = float(data['gas_resistance'])
            hum = float(data['humidity'])
            
            # Normalize IAQ calculation
            gas_baseline = 50000
            if gas_res >= gas_baseline:
                gas_score = 0
            else:
                gas_score = (1 - gas_res / gas_baseline) * 250
            
            if 30 <= hum <= 60:
                hum_score = 0
            elif hum < 30:
                hum_score = (30 - hum) * 5
            else:
                hum_score = (hum - 60) * 5
            
            iaq_score = min(500, gas_score + hum_score)
            
            # Enrich data
            enriched = data.copy()
            enriched.update({
                'iaq_score': round(iaq_score, 2),
                'processed_at': datetime.utcnow().isoformat(),
                'partition_key': datetime.utcnow().strftime('year=%Y/month=%m/day=%d/hour=%H')
            })
            processed_records.append(enriched)
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            invalid_count += 1
            logger.error(f"Parse error on message {idx}: {e}")
            logger.debug(f"Raw body: {msg.get('Body', '')[:300]}")
    
    logger.info(f"Processed: {len(processed_records)} valid, {invalid_count} invalid")
    
    # Handle case: all messages invalid
    if not processed_records:
        logger.warning("No valid records after processing")
        receipt_handles = [msg['ReceiptHandle'] for msg in all_messages]
        delete_sqs_messages_batch(sqs_hook, receipt_handles)
        raise AirflowSkipException("No valid records")
    
    # 3. Save to S3
    df = pd.DataFrame(processed_records)
    now = datetime.utcnow()
    s3_path = f"bronze/bme680/year={now.year}/month={now.month:02d}/day={now.day:02d}/data_{now.strftime('%H%M%S')}.parquet"
    
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
        
        # 4. Delete messages from SQS (only after S3 success)
        receipt_handles = [msg['ReceiptHandle'] for msg in all_messages]
        delete_sqs_messages_batch(sqs_hook, receipt_handles)
        
    except Exception as e:
        logger.error(f"S3 upload failed - messages NOT deleted: {e}", exc_info=True)
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
