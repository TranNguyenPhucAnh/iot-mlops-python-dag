"""
IoT BME680 Real-time Inference Pipeline
========================================
1. Load latest data from Bronze layer
2. Load production model from MLflow
3. Run inference
4. Save predictions to Gold layer
5. Send alerts for detected anomalies

Schedule: Every 15 minutes
Author: IoT ML Team
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import logging
from io import BytesIO
import joblib

logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:80"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"
S3_BUCKET = "iot-bme680-data-lake-prod"
S3_BRONZE_PREFIX = "bronze/bme680/"
S3_GOLD_PREFIX = "gold/bme680_predictions/"
ANOMALY_THRESHOLD = -0.5  # Isolation Forest score threshold

def load_production_model(**context):
    """Load latest production model from MLflow"""
    logger.info("=" * 60)
    logger.info("Loading Production Model from MLflow")
    logger.info("=" * 60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Get production model
        prod_versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME,
            stages=["Production"]
        )
        
        if not prod_versions:
            # Fallback to Staging
            logger.warning("⚠️ No Production model, using Staging")
            prod_versions = client.get_latest_versions(
                REGISTERED_MODEL_NAME,
                stages=["Staging"]
            )
        
        if not prod_versions:
            raise ValueError("No model found in Production or Staging")
        
        model_version = prod_versions[0]
        logger.info(f"✅ Using model version: {model_version.version}")
        logger.info(f"   Stage: {model_version.current_stage}")
        logger.info(f"   Run ID: {model_version.run_id}")
        
        # ✅ Save model URI to XCom instead of model object
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version.current_stage}"
        scaler_uri = f"runs:/{model_version.run_id}/preprocessor/scaler.pkl"
        
        logger.info("✅ Model URIs prepared")
        
        # Store URIs in XCom (lightweight, JSON-serializable)
        context['ti'].xcom_push(key='model_uri', value=model_uri)
        context['ti'].xcom_push(key='scaler_uri', value=scaler_uri)
        context['ti'].xcom_push(key='model_version', value=model_version.version)
        context['ti'].xcom_push(key='run_id', value=model_version.run_id)
        
        return {
            'model_name': REGISTERED_MODEL_NAME,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'run_id': model_version.run_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise

def load_recent_data(**context):
    """Load last 15 minutes of data for inference"""
    logger.info("=" * 60)
    logger.info("Loading Recent Data")
    logger.info("=" * 60)
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    # Airflow 3.0+: use logical_date instead of execution_date
    execution_date = context.get('logical_date') or context.get('execution_date')
    
    # Look for data from last 15 minutes
    prefix = (
        f"{S3_BRONZE_PREFIX}"
        f"year={execution_date.year}/"
        f"month={execution_date.month:02d}/"
        f"day={execution_date.day:02d}/"
    )
    
    keys = s3_hook.list_keys(bucket_name=S3_BUCKET, prefix=prefix)
    
    if not keys:
        logger.warning(f"⚠️ No data found in {prefix}")
        return None
    
    parquet_files = [k for k in keys if k.endswith('.parquet')]
    logger.info(f"📁 Found {len(parquet_files)} Parquet files")
    
    # Read files
    all_data = []
    for file_key in parquet_files[-3:]:  # Last 3 files (15 min window)
        obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        df = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
        all_data.append(df)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Loaded {len(df_combined):,} records")
    
    context['ti'].xcom_push(key='raw_data', value=df_combined.to_dict('records'))
    
    return len(df_combined)

def run_inference(**context):
    """Run anomaly detection on new data"""
    logger.info("=" * 60)
    logger.info("Running Inference")
    logger.info("=" * 60)

    # Load data
    raw_data = context['ti'].xcom_pull(task_ids='load_data', key='raw_data')
    if not raw_data:
        logger.warning("⚠️ No data to process")
        return None

    df = pd.DataFrame(raw_data)

    # Load model URIs from XCom
    model_uri  = context['ti'].xcom_pull(task_ids='load_model', key='model_uri')
    scaler_uri = context['ti'].xcom_pull(task_ids='load_model', key='scaler_uri')

    logger.info(f"📦 Loading model from:  {model_uri}")
    logger.info(f"📦 Loading scaler from: {scaler_uri}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)

    scaler_path = mlflow.artifacts.download_artifacts(scaler_uri)
    scaler = joblib.load(scaler_path)

    logger.info(f"📊 Running inference on {len(df)} records")

    # =============================================
    # Feature Engineering
    # =============================================
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['hour']               = df['timestamp'].dt.hour
    df['day_of_week']        = df['timestamp'].dt.dayofweek
    df['is_weekend']         = (df['day_of_week'] >= 5).astype(int)
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
    df['gas_pressure_ratio']  = df['gas_resistance'] / (df['pressure'] + 1e-6)

    for col in ['temperature', 'humidity', 'iaq_score']:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'{col}_rolling_std']  = df[col].rolling(window=10, min_periods=1).std().fillna(0)

    # =============================================
    # ✅ KEY FIX: Align features với scaler
    # =============================================
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score',
        'hour', 'day_of_week', 'is_weekend',
        'temp_humidity_ratio', 'gas_pressure_ratio',
        'temperature_rolling_mean', 'temperature_rolling_std',
        'humidity_rolling_mean', 'humidity_rolling_std',
        'iaq_score_rolling_mean', 'iaq_score_rolling_std'
    ]

    # Lấy đúng thứ tự features từ scaler (nếu có)
    if hasattr(scaler, 'feature_names_in_'):
        train_features  = list(scaler.feature_names_in_)
        infer_features  = list(df.columns)

        # Log để debug
        logger.info(f"Scaler trained on {len(train_features)} features: {train_features}")
        logger.info(f"Inference has {len(df.columns)} columns")

        # Check missing/extra
        missing = set(train_features) - set(infer_features)
        extra   = set(infer_features) - set(train_features)

        if missing:
            logger.error(f"❌ Missing features: {missing}")
            raise ValueError(
                f"Missing features required by scaler: {missing}\n"
                f"Add these columns in feature engineering step."
            )

        if extra:
            logger.warning(f"⚠️ Extra features (will be dropped): {extra}")

        # ✅ Reorder theo đúng thứ tự của scaler
        X = df[train_features]
        logger.info(f"✅ Aligned features to scaler order: {train_features}")

    else:
        # Scaler không có feature_names_in_ (fit bằng numpy array)
        # Dùng thứ tự mặc định và chuyển sang numpy
        logger.warning("⚠️ Scaler has no feature_names_in_, using default feature_cols order")

        # Kiểm tra feature_cols đủ không
        missing = set(feature_cols) - set(df.columns)
        if missing:
            logger.error(f"❌ Missing features: {missing}")
            raise ValueError(f"Missing features: {missing}")

        X = df[feature_cols]

        # ✅ Chuyển sang numpy để tránh lỗi feature name mismatch
        X = pd.DataFrame(X.values, columns=feature_cols)
        logger.info(f"✅ Using default feature_cols with numpy conversion")

    logger.info(f"📐 Feature matrix shape: {X.shape}")

    # =============================================
    # ✅ Debug: Validate data trước khi scale
    # =============================================
    logger.info("🔍 Sample raw data (first 3 rows):")
    logger.info(f"\n{X.head(3).to_string()}")

    # Kiểm tra giá trị hợp lý
    sensor_ranges = {
        'temperature': (-20, 80),
        'humidity':    (0, 100),
        'pressure':    (900, 1100),
        'iaq_score':   (0, 500),
    }
    for col, (lo, hi) in sensor_ranges.items():
        if col in X.columns:
            out = ((X[col] < lo) | (X[col] > hi)).sum()
            if out > 0:
                logger.warning(
                    f"⚠️ {col}: {out} values out of range [{lo}, {hi}] "
                    f"(min={X[col].min():.2f}, max={X[col].max():.2f})"
                )

    logger.info("🔍 Scaler parameters:")
    logger.info(f"  Mean:  {dict(zip(X.columns, scaler.mean_.round(3)))}")
    logger.info(f"  Scale: {dict(zip(X.columns, scaler.scale_.round(3)))}")

    # =============================================
    # Scale → Predict
    # =============================================
    X_scaled     = scaler.transform(X)
    predictions  = model.predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)

    df['prediction']    = predictions
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly']    = (predictions == -1).astype(int)

    anomaly_count = int(df['is_anomaly'].sum())
    anomaly_rate  = anomaly_count / len(df)

    logger.info(f"✅ Inference complete")
    logger.info(f"📊 {anomaly_count} anomalies detected ({anomaly_rate:.2%})")

    # =============================================
    # Save to Gold layer
    # =============================================
    now       = datetime.utcnow()
    gold_path = (
        f"{S3_GOLD_PREFIX}"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"predictions_{now.strftime('%H%M%S')}.parquet"
    )

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, compression='snappy')
    buffer.seek(0)

    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=gold_path,
        bucket_name=S3_BUCKET,
        replace=True
    )

    logger.info(f"💾 Saved predictions to s3://{S3_BUCKET}/{gold_path}")

    # =============================================
    # Push anomalies for alerting
    # =============================================
    if anomaly_count > 0:
        anomaly_cols = ['timestamp', 'device_id', 'temperature',
                        'humidity', 'iaq_score', 'anomaly_score']
        anomaly_cols = [c for c in anomaly_cols if c in df.columns]

        anomaly_df = df[df['is_anomaly'] == 1][anomaly_cols].copy()

        # ✅ FIX 1: Convert tất cả non-serializable types
        for col in anomaly_df.columns:
            # Timestamp → string
            if pd.api.types.is_datetime64_any_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
            # numpy types → python native
            elif pd.api.types.is_float_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].astype(float)
            elif pd.api.types.is_integer_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].astype(int)

        anomalies = anomaly_df.to_dict('records')
        context['ti'].xcom_push(key='anomalies', value=anomalies)

    return {
        'total_records': len(df),
        'anomaly_count': anomaly_count,
        'anomaly_rate':  float(anomaly_rate)
    }
    
def send_alerts(**context):
    """Send alerts for detected anomalies"""
    logger.info("=" * 60)
    logger.info("Checking for Alerts")
    logger.info("=" * 60)
    
    anomalies = context['ti'].xcom_pull(task_ids='run_inference', key='anomalies')
    
    if not anomalies:
        logger.info("✅ No anomalies detected")
        return
    
    logger.warning(f"⚠️ Detected {len(anomalies)} anomalies!")
    
    for anomaly in anomalies[:5]:  # Show first 5
        logger.warning(f"""
Anomaly detected:
  Time: {anomaly['timestamp']}
  Device: {anomaly['device_id']}
  Temperature: {anomaly['temperature']:.1f}°C
  Humidity: {anomaly['humidity']:.1f}%
  IAQ Score: {anomaly['iaq_score']:.1f}
  Anomaly Score: {anomaly['anomaly_score']:.3f}
""")
    
    # TODO: Send to alerting system
    # - Slack webhook
    # - PagerDuty
    # - Email
    # - SNS topic
    
    return len(anomalies)

# DAG Definition
default_args = {
    'owner': 'iot-ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='iot_ml_inference_pipeline',
    description='Real-time anomaly detection inference',
    schedule='*/15 * * * *',  # Every 15 minutes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'inference', 'real-time', 'anomaly-detection']
) as dag:

    load_model = PythonOperator(
        task_id='load_model',
        python_callable=load_production_model
    )

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_recent_data
    )

    inference = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference
    )

    alert = PythonOperator(
        task_id='send_alerts',
        python_callable=send_alerts
    )

    [load_model, load_data] >> inference >> alert
