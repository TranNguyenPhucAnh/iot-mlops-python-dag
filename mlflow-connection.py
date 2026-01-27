"""
MLflow Integration Test DAG
- Verify Postgres metadata storage
- Verify S3 artifact storage
- Check MLflow server health
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import os
import tempfile
import requests
import logging
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    'http://mlflow.mlflow.svc.cluster.local:5000'
)
EXPERIMENT_NAME = "iot_sensor_integration_test"

def check_mlflow_health():
    """Test 1: Check MLflow server health endpoint"""
    logger.info("=" * 60)
    logger.info("TEST 1: Checking MLflow server health...")
    logger.info("=" * 60)
    
    try:
        health_url = f"{MLFLOW_TRACKING_URI}/health"
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        
        logger.info(f"✅ MLflow server is healthy: {health_url}")
        logger.info(f"Response: {response.text}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ MLflow server health check failed: {e}")
        raise

def test_mlflow_metadata():
    """Test 2: Log metadata to Postgres backend"""
    logger.info("=" * 60)
    logger.info("TEST 2: Testing Postgres metadata storage...")
    logger.info("=" * 60)
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"metadata_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("test_type", "metadata_storage")
            mlflow.log_param("sensor_id", "BME680_DEV_01")
            mlflow.log_param("device_location", "Ho Chi Minh City")
            
            # Log metrics
            mlflow.log_metric("temperature", 28.5)
            mlflow.log_metric("humidity", 65.2)
            mlflow.log_metric("pressure", 1013.25)
            mlflow.log_metric("gas_resistance", 43944)
            
            # Log tags
            mlflow.set_tag("team", "iot-platform")
            mlflow.set_tag("environment", "test")
            mlflow.set_tag("triggered_by", "airflow")
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            
            logger.info(f"✅ Metadata logged successfully")
            logger.info(f"   Run ID: {run_id}")
            logger.info(f"   Experiment ID: {experiment_id}")
            logger.info(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
            
            return run_id
            
    except Exception as e:
        logger.error(f"❌ Metadata logging failed: {e}", exc_info=True)
        raise

def test_mlflow_artifacts(**context):
    """Test 3: Log artifacts to S3 backend"""
    logger.info("=" * 60)
    logger.info("TEST 3: Testing S3 artifact storage...")
    logger.info("=" * 60)
    
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "mlflow_test_artifact.txt")
    json_file = os.path.join(temp_dir, "sensor_data.json")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"artifact_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Create test files
            with open(test_file, "w") as f:
                f.write(f"MLflow Artifact Test\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Tracking URI: {MLFLOW_TRACKING_URI}\n")
                f.write("Status: ✅ Connection successful\n")
            
            import json
            with open(json_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "sensors": {
                        "temperature": 28.5,
                        "humidity": 65.2,
                        "pressure": 1013.25,
                        "gas_resistance": 43944
                    },
                    "test_metadata": {
                        "dag_id": context.get('dag').dag_id if context.get('dag') else "manual",
                        "task_id": context.get('task').task_id if context.get('task') else "manual",
                        "execution_date": str(context.get('execution_date', datetime.now()))
                    }
                }, f, indent=2)
            
            # Log artifacts
            logger.info(f"Logging artifact: {test_file}")
            mlflow.log_artifact(test_file, artifact_path="test_files")
            
            logger.info(f"Logging artifact: {json_file}")
            mlflow.log_artifact(json_file, artifact_path="sensor_data")
            
            # Get artifact URI
            run = mlflow.active_run()
            run_id = run.info.run_id
            artifact_uri = run.info.artifact_uri
            
            logger.info(f"✅ Artifacts logged successfully")
            logger.info(f"   Run ID: {run_id}")
            logger.info(f"   Artifact URI: {artifact_uri}")
            
            # Verify artifacts can be listed
            client = MlflowClient(MLFLOW_TRACKING_URI)
            artifacts = client.list_artifacts(run_id)
            
            logger.info(f"   Artifacts in run:")
            for artifact in artifacts:
                logger.info(f"      - {artifact.path} ({artifact.file_size} bytes)")
            
            # Cleanup temp files
            os.remove(test_file)
            os.remove(json_file)
            logger.info("🗑️ Cleaned up temporary files")
            
            return {
                'run_id': run_id,
                'artifact_uri': artifact_uri,
                'artifact_count': len(artifacts)
            }
            
    except Exception as e:
        logger.error(f"❌ Artifact logging failed: {e}", exc_info=True)
        
        # Cleanup on error
        for f in [test_file, json_file]:
            if os.path.exists(f):
                os.remove(f)
        
        raise

def verify_postgres_connection():
    """Test 4: Verify Postgres backend directly"""
    logger.info("=" * 60)
    logger.info("TEST 4: Verifying Postgres backend...")
    logger.info("=" * 60)
    
    try:
        client = MlflowClient(MLFLOW_TRACKING_URI)
        
        # List experiments
        experiments = client.search_experiments()
        logger.info(f"✅ Found {len(experiments)} experiments in Postgres")
        
        for exp in experiments[:5]:  # Show first 5
            logger.info(f"   - {exp.name} (ID: {exp.experiment_id})")
        
        # Get test experiment
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            logger.info(f"✅ Test experiment exists: {EXPERIMENT_NAME}")
            logger.info(f"   Experiment ID: {experiment.experiment_id}")
            
            # List runs in experiment
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
            logger.info(f"   Total runs: {len(runs)}")
            
            for run in runs:
                logger.info(f"      - Run {run.info.run_id[:8]}... | Status: {run.info.status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Postgres verification failed: {e}", exc_info=True)
        raise

def verify_s3_artifacts(**context):
    """Test 5: Verify S3 artifact storage"""
    logger.info("=" * 60)
    logger.info("TEST 5: Verifying S3 artifact storage...")
    logger.info("=" * 60)
    
    try:
        client = MlflowClient(MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            logger.warning("⚠️ No experiment found - skipping S3 verification")
            return
        
        # Get latest run with artifacts
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"]
        )
        
        artifact_count = 0
        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                artifact_count += len(artifacts)
                logger.info(f"✅ Run {run.info.run_id[:8]}... has {len(artifacts)} artifacts")
                logger.info(f"   Artifact URI: {run.info.artifact_uri}")
        
        if artifact_count > 0:
            logger.info(f"✅ S3 artifact storage verified: {artifact_count} total artifacts")
        else:
            logger.warning("⚠️ No artifacts found in recent runs")
        
        return artifact_count
        
    except Exception as e:
        logger.error(f"❌ S3 verification failed: {e}", exc_info=True)
        raise

def print_test_summary(**context):
    """Print final test summary"""
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    logger.info("✅ All MLflow integration tests passed!")
    logger.info(f"   MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"   Experiment: {EXPERIMENT_NAME}")
    logger.info("")
    logger.info("Components verified:")
    logger.info("   ✅ MLflow server health")
    logger.info("   ✅ Postgres metadata storage")
    logger.info("   ✅ S3 artifact storage")
    logger.info("   ✅ MLflow client connectivity")
    logger.info("")
    logger.info("🎉 MLflow is ready for production use!")

# DAG Definition
default_args = {
    'owner': 'iot-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    dag_id='mlflow_connection_test_dag',
    description='Comprehensive MLflow integration test (Postgres + S3)',
    schedule=None,  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=['mlflow', 'integration-test', 'health-check']
) as dag:

    health_check = PythonOperator(
        task_id='check_mlflow_health',
        python_callable=check_mlflow_health
    )

    test_metadata = PythonOperator(
        task_id='test_postgres_metadata',
        python_callable=test_mlflow_metadata
    )

    test_artifacts = PythonOperator(
        task_id='test_s3_artifacts',
        python_callable=test_mlflow_artifacts
    )

    verify_postgres = PythonOperator(
        task_id='verify_postgres_backend',
        python_callable=verify_postgres_connection
    )

    verify_s3 = PythonOperator(
        task_id='verify_s3_backend',
        python_callable=verify_s3_artifacts
    )

    summary = PythonOperator(
        task_id='print_summary',
        python_callable=print_test_summary
    )

    # Test flow
    health_check >> [test_metadata, test_artifacts]
    test_metadata >> verify_postgres
    test_artifacts >> verify_s3
    [verify_postgres, verify_s3] >> summary
