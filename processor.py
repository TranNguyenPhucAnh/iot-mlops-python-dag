from airflow import DAG
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import json
import pandas as pd
from datetime import datetime
import io

# Cấu hình
SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/iot-data-queue"
S3_BUCKET = "iot-mlops-data-lake-408279620390"

def process_iot_data(**context):
    # 1. Lấy dữ liệu từ SqsSensor (XCom)
    messages = context['ti'].xcom_pull(task_ids='wait_for_sqs_messages')
    
    if not messages:
        print("Không có message nào để xử lý.")
        return

    processed_records = []
    
    for msg in messages:
        data = json.loads(msg['Body'])
        
        # 2. Logic Transformation: Tính toán IAQ cơ bản
        # IAQ Score dựa trên Gas Resistance và Humidity (Công thức đơn giản hóa)
        gas_res = data.get('gas_resistance', 0)
        hum = data.get('humidity', 0)
        
        # IAQ đơn giản: Càng cao càng tốt (Trong thực tế Bosch dùng thuật toán phức tạp hơn)
        iaq_score = (gas_res * 0.75) + (hum * 0.25) 
        
        data['iaq_score'] = round(iaq_score, 2)
        data['processed_at'] = datetime.utcnow().isoformat()
        processed_records.append(data)

    # 3. Chuyển thành DataFrame và nén thành Parquet
    df = pd.DataFrame(processed_records)
    
    # Tạo đường dẫn S3 theo Partition: year/month/day/hour
    now = datetime.utcnow()
    s3_path = f"bronze/bme680/{now.strftime('year=%Y/month=%m/day=%d')}/data_{now.strftime('%H%M%S')}.parquet"

    # Ghi file Parquet vào memory buffer
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='fastparquet')
    
    # 4. Upload lên S3 dùng S3Hook
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=s3_path,
        bucket_name=S3_BUCKET,
        replace=True
    )
    print(f"Đã lưu {len(processed_records)} bản ghi vào: {s3_path}")

with DAG(
    dag_id='iot_bme680_ingestion_pipeline',
    start_date=datetime(2026, 1, 1),
    schedule='@hourly', # Gom dữ liệu mỗi giờ
    catchup=False
) as dag:

    # Đợi và lấy tối đa 10 messages từ SQS
    wait_for_sqs = SqsSensor(
        task_id='wait_for_sqs_messages',
        sqs_queue_url=SQS_QUEUE_URL,
        max_messages=10,
        wait_time_seconds=20,
        aws_conn_id='aws_default'
    )

    process_data = PythonOperator(
        task_id='transform_and_save_to_s3',
        python_callable=process_iot_data
    )

    wait_for_sqs >> process_data
