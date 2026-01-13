from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from datetime import datetime, timedelta

# THAY THẾ CÁC GIÁ TRỊ NÀY THEO TERRAFORM CỦA BẠN
SQS_QUEUE_URL = "https://sqs.ap-southeast-1.amazonaws.com/408279620390/bme680-sensor-data"
S3_BUCKET = "iot-bme680-data-lake-prod" # Tên bucket bạn tạo bằng Terraform

default_args = {
    'owner': 'phucanh',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='iot_sqs_to_s3_test',
    default_args=default_args,
    schedule='@hourly',
    catchup=False,
    tags=['iot', 'test']
) as dag:

    # 1. Đợi tin nhắn từ SQS (Kiểm tra quyền Read SQS)
    wait_for_message = SqsSensor(
        task_id='wait_for_iot_data',
        sqs_queue=SQS_QUEUE_URL,
        max_messages=1,
        # ✅ KEY FIX: No filtering (đã test thành công!)
        message_filtering=None,  # ← KHÔNG dùng 'literal'
        message_filtering_match_values=None,
        message_filtering_config=None,
        
        delete_message_on_reception=True, # SQSSensor xoá message khỏi queue ngay sau khi receive message

        wait_time_seconds=20, # sqs polling duration
        mode='reschedule',  # THÊM DÒNG NÀY: Giải phóng worker khi đang đợi
        poke_interval=45,   # Cứ mỗi 40 giây gọi API polling 1 lần
        timeout=600,        # Thời gian tối đa cho cả task là 10 phút
        aws_conn_id='aws_default' # Airflow sẽ tự dùng IRSA nếu connection này để trống
    )

    # 2. Ghi một file dummy lên S3 (Kiểm tra quyền Write S3)
    write_to_s3 = S3CreateObjectOperator(
        task_id='write_test_to_s3',
        s3_bucket=S3_BUCKET,
        s3_key='test/hello_from_airflow.txt',
        data='Dữ liệu IoT giả lập từ Airflow Webserver',
        replace=True,
        aws_conn_id='aws_default'
    )

    wait_for_message >> write_to_s3
