"""
DAG để debug S3CreateObjectOperator
File: dags/debug_s3_only.py
"""
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from datetime import datetime, timedelta
import logging
import boto3
from botocore.config import Config

S3_BUCKET = "iot-bme680-data-lake-prod"

default_args = {
    'owner': 'phucanh',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=3),  # Global timeout
}

with DAG(
    dag_id='debug_s3_write_only',
    default_args=default_args,
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['debug', 's3', 'test']
) as dag:
    
    # ========================================
    # TEST 1: S3CreateObjectOperator
    # ========================================
    test_s3_operator = S3CreateObjectOperator(
        task_id='test_s3_operator',
        s3_bucket=S3_BUCKET,
        s3_key=f'test/s3operator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
        data='Test from S3CreateObjectOperator',
        replace=True,
        aws_conn_id='aws_default',
        verify=True,
        execution_timeout=timedelta(seconds=120),
        retries=0,  # No retry để debug nhanh
    )
    
    # ========================================
    # TEST 2: PythonOperator với boto3 client
    # ========================================
    @task(execution_timeout=timedelta(seconds=120))
    def test_boto3_client():
        """Test S3 upload với boto3 trực tiếp"""
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("=" * 60)
            logger.info("🔵 BẮT ĐẦU TEST BOTO3 CLIENT")
            logger.info("=" * 60)
            
            # Config boto3 với timeout rõ ràng
            config = Config(
                region_name='ap-southeast-1',
                retries={'max_attempts': 2, 'mode': 'standard'},
                connect_timeout=30,
                read_timeout=30,
                max_pool_connections=10
            )
            
            logger.info("🔵 Khởi tạo S3 client...")
            s3_client = boto3.client('s3', config=config)
            
            logger.info("🔵 Kiểm tra credentials...")
            sts_client = boto3.client('sts', config=config)
            caller_identity = sts_client.get_caller_identity()
            logger.info(f"✅ Caller Identity: {caller_identity}")
            
            logger.info(f"🔵 Đang ghi file lên bucket: {S3_BUCKET}")
            key = f'test/boto3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            response = s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=b'Test from boto3 client directly',
                ContentType='text/plain'
            )
            
            logger.info("=" * 60)
            logger.info(f"✅ GHI FILE THÀNH CÔNG!")
            logger.info(f"   - Bucket: {S3_BUCKET}")
            logger.info(f"   - Key: {key}")
            logger.info(f"   - ETag: {response['ETag']}")
            logger.info(f"   - HTTP Status: {response['ResponseMetadata']['HTTPStatusCode']}")
            logger.info("=" * 60)
            
            return {
                'bucket': S3_BUCKET,
                'key': key,
                'etag': response['ETag'],
                'status': 'success'
            }
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ LỖI: {str(e)}")
            logger.error(f"   - Type: {type(e).__name__}")
            logger.error("=" * 60)
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # ========================================
    # TEST 3: S3Hook từ Airflow
    # ========================================
    @task(execution_timeout=timedelta(seconds=120))
    def test_s3_hook():
        """Test S3 upload với S3Hook"""
        logger = logging.getLogger(__name__)
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook
        
        try:
            logger.info("=" * 60)
            logger.info("🔵 BẮT ĐẦU TEST S3HOOK")
            logger.info("=" * 60)
            
            logger.info("🔵 Khởi tạo S3Hook...")
            hook = S3Hook(aws_conn_id='aws_default')
            
            logger.info("🔵 Kiểm tra connection...")
            conn = hook.get_connection('aws_default')
            logger.info(f"   - Connection type: {conn.conn_type}")
            logger.info(f"   - Extra: {conn.extra}")
            
            key = f'test/s3hook_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            logger.info(f"🔵 Đang ghi file với S3Hook...")
            hook.load_string(
                string_data='Test from S3Hook',
                key=key,
                bucket_name=S3_BUCKET,
                replace=True
            )
            
            logger.info("=" * 60)
            logger.info(f"✅ GHI FILE THÀNH CÔNG!")
            logger.info(f"   - Bucket: {S3_BUCKET}")
            logger.info(f"   - Key: {key}")
            logger.info("=" * 60)
            
            return {
                'bucket': S3_BUCKET,
                'key': key,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ LỖI: {str(e)}")
            logger.error("=" * 60)
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # ========================================
    # TEST 4: Verify files trên S3
    # ========================================
    @task(execution_timeout=timedelta(seconds=60))
    def verify_s3_files():
        """List files trong test/ folder"""
        logger = logging.getLogger(__name__)
        
        try:
            config = Config(
                region_name='ap-southeast-1',
                connect_timeout=30,
                read_timeout=30
            )
            
            s3_client = boto3.client('s3', config=config)
            
            logger.info("=" * 60)
            logger.info("🔵 LISTING FILES IN test/ FOLDER")
            logger.info("=" * 60)
            
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix='test/',
                MaxKeys=10
            )
            
            if 'Contents' in response:
                logger.info(f"✅ Found {len(response['Contents'])} files:")
                for obj in response['Contents']:
                    logger.info(f"   - {obj['Key']} ({obj['Size']} bytes)")
            else:
                logger.warning("⚠️  No files found in test/ folder")
            
            logger.info("=" * 60)
            
            return {'file_count': len(response.get('Contents', []))}
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi list files: {str(e)}")
            raise
    
    # ========================================
    # Task Dependencies
    # ========================================
    boto3_result = test_boto3_client()
    hook_result = test_s3_hook()
    verify_result = verify_s3_files()
    
    # Run tests in parallel, then verify
    test_s3_operator >> verify_result
    boto3_result >> verify_result
    hook_result >> verify_result
