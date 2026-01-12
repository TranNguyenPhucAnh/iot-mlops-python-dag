from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3

def test_sqs_connection(**context):
    """Test đọc SQS trực tiếp không qua Sensor"""
    sqs = boto3.client('sqs', region_name='ap-southeast-1')
    queue_url = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
    
    try:
        # Thử receive message
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )
        
        print(f"Response: {response}")
        
        if 'Messages' in response:
            print(f"✅ Found {len(response['Messages'])} messages")
            for msg in response['Messages']:
                print(f"Message Body: {msg.get('Body')}")
            return True
        else:
            print("❌ No messages in queue")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

with DAG(
    dag_id='test_sqs_direct',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    
    test_task = PythonOperator(
        task_id='test_sqs',
        python_callable=test_sqs_connection
    )
