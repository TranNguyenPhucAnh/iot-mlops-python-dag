# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import boto3

# def test_sqs_connection(**context):
#     """Test đọc SQS trực tiếp không qua Sensor"""
#     sqs = boto3.client('sqs', region_name='ap-southeast-1')
#     queue_url = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
    
#     try:
#         # Thử receive message
#         response = sqs.receive_message(
#             QueueUrl=queue_url,
#             MaxNumberOfMessages=1,
#             WaitTimeSeconds=20
#         )
        
#         print(f"Response: {response}")
        
#         if 'Messages' in response:
#             print(f"✅ Found {len(response['Messages'])} messages")
#             for msg in response['Messages']:
#                 print(f"Message Body: {msg.get('Body')}")
#             return True
#         else:
#             print("❌ No messages in queue")
#             return False
            
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         raise

# with DAG(
#     dag_id='test_sqs_direct',
#     start_date=datetime(2024, 1, 1),
#     schedule=None,
#     catchup=False
# ) as dag:
    
#     test_task = PythonOperator(
#         task_id='test_sqs',
#         python_callable=test_sqs_connection
#     )

from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from datetime import datetime, timedelta

SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
S3_BUCKET = "iot-bme680-data-lake-prod"

default_args = {
    'owner': 'phucanh',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='iot_sqs_to_s3_test_v2',
    default_args=default_args,
    schedule='@hourly',
    catchup=False,
    tags=['iot', 'test']
) as dag:

    wait_for_message = SqsSensor(
        task_id='wait_for_iot_data',
        sqs_queue=SQS_QUEUE_URL,
        max_messages=1,
        wait_time_seconds=15,
        
        # ✅ THAY ĐỔI QUAN TRỌNG:
        message_filtering='literal',  # hoặc 'jsonpath'
        message_filtering_match_values=None,  # None = accept any message
        delete_message_on_reception=False,  # Không xóa message khi nhận (để debug)
        
        mode='poke',  # ← THAY ĐỔI: Dùng poke thay vì reschedule để debug dễ hơn
        poke_interval=30,  # Kiểm tra mỗi 30s
        timeout=300,  # Timeout 5 phút
        
        aws_conn_id='aws_default'
    )

    write_to_s3 = S3CreateObjectOperator(
        task_id='write_test_to_s3',
        s3_bucket=S3_BUCKET,
        s3_key='test/hello_from_airflow.txt',
        data='Dữ liệu IoT giả lập từ Airflow Webserver',
        replace=True,
        aws_conn_id='aws_default'
    )

    wait_for_message >> write_to_s3
