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

def pull_and_process_sqs(**context):
    """Pull messages from SQS, process, and save to S3"""
    sqs_hook = SqsHook(aws_conn_id='aws_default')
    
    # 1. Pull messages từ SQS
    all_messages = []
    max_batches = 10  # Tối đa 10 batches
    
    for batch_num in range(max_batches):
        response = sqs_hook.get_conn().receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=10,  # SQS max = 10 per call
            WaitTimeSeconds=20,       # Long polling
            VisibilityTimeout=300    # 5 phút
        )
        
        messages = response.get('Messages', [])
        if not messages:
            logger.info(f"No more messages after {batch_num} batches")
            break
            
        all_messages.extend(messages)
        logger.info(f"Batch {batch_num + 1}: pulled {len(messages)} messages")
        
        # Stop nếu đủ 50 messages
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
            data = json.loads(msg['Body'])
            
            required = ['temperature', 'humidity', 'pressure', 'gas_resistance']
            if not all(data.get(k) for k in required):
                invalid_count += 1
                logger.warning(f"Message {idx} missing fields: {[k for k in required if not data.get(k)]}")
                continue
                
            gas_res = float(data.get('gas_resistance', 0))
            hum = float(data.get('humidity', 0))
            
            iaq_score = max(0, min(500, (gas_res * 0.75) + (hum * 0.25)))
            
            enriched = data.copy()
            enriched.update({
                'iaq_score': round(iaq_score, 2),
                'processed_at': datetime.utcnow().isoformat(),
                'partition_key': datetime.utcnow().strftime('year=%Y/month=%m/day=%d/hour=%H')
            })
            processed_records.append(enriched)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            invalid_count += 1
            logger.error(f"Parse error on message {idx}: {e}")

    logger.info(f"Processed: {len(processed_records)} valid, {invalid_count} invalid")

    if not processed_records:
        logger.warning("No valid records after processing")
        # Vẫn xóa messages lỗi để tránh reprocess
        receipt_handles = [msg['ReceiptHandle'] for msg in all_messages]
        sqs_hook.get_conn().delete_message_batch(
            QueueUrl=SQS_QUEUE_URL,
            Entries=[{'Id': str(i), 'ReceiptHandle': rh} for i, rh in enumerate(receipt_handles)]
        )
        raise AirflowSkipException("No valid records")

    # 3. Save to S3
    df = pd.DataFrame(processed_records)
    now = datetime.utcnow()
    s3_path = f"bronze/bme680/year={now.year}/month={now.month:02d}/day={now.day:02d}/data_{now.strftime('%H%M%S')}.parquet"

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
    
    # 4. Delete messages from SQS
    receipt_handles = [msg['ReceiptHandle'] for msg in all_messages]
    
    # SQS DeleteMessageBatch max = 10, cần chia batches
    for i in range(0, len(receipt_handles), 10):
        batch = receipt_handles[i:i+10]
        sqs_hook.get_conn().delete_message_batch(
            QueueUrl=SQS_QUEUE_URL,
            Entries=[{'Id': str(j), 'ReceiptHandle': rh} for j, rh in enumerate(batch)]
        )
    
    logger.info(f"✅ Deleted {len(receipt_handles)} messages from SQS")

with DAG(
    dag_id='iot_bme680_ingestion_pipeline_v3',
    start_date=datetime(2026, 1, 1),
    schedule='*/5 * * * *',
    catchup=False,
    max_active_runs=4,
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
    }
) as dag:

    process_sqs_to_s3 = PythonOperator(
        task_id='process_sqs_to_s3',
        python_callable=pull_and_process_sqs,
    )
