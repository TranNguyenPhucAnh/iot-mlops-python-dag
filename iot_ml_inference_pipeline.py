"""
IoT BME680 Real-time Inference Pipeline v2
============================================
Flow mới: Silver layer → Inference → Gold layer
Feature engineering đã được Transform DAG xử lý.

Schedule: Every 15 minutes
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import logging
from io import BytesIO
import joblib

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI   = "http://mlflow.mlflow.svc.cluster.local:80"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"
S3_BUCKET             = "iot-bme680-data-lake-prod"
S3_SILVER_PREFIX      = "silver/bme680_features/"
S3_GOLD_PREFIX        = "gold/bme680_predictions/"

FEATURE_COLS = [
    'temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score',
    'hour', 'day_of_week', 'is_weekend',
    'temp_humidity_ratio', 'gas_pressure_ratio',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_mean', 'humidity_rolling_std',
    'iaq_score_rolling_mean', 'iaq_score_rolling_std'
]


def load_production_model(**context):
    """Load latest Production/Staging model từ MLflow"""
    logger.info("=" * 60)
    logger.info("STEP 1: Load Production Model")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])

        if not prod_versions:
            logger.warning("⚠️ No Production model, fallback to Staging")
            prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"])

        if not prod_versions:
            raise ValueError("No model found in Production or Staging")

        model_version = prod_versions[0]
        run_id        = model_version.run_id

        logger.info(f"✅ Model version: {model_version.version} ({model_version.current_stage})")
        logger.info(f"   Run ID: {run_id}")

        context['ti'].xcom_push(key='model_uri',     value=f"models:/{REGISTERED_MODEL_NAME}/{model_version.current_stage}")
        context['ti'].xcom_push(key='scaler_uri',    value=f"runs:/{run_id}/preprocessor/scaler.pkl")
        context['ti'].xcom_push(key='run_id',        value=run_id)
        context['ti'].xcom_push(key='model_version', value=model_version.version)

        return {'version': model_version.version, 'stage': model_version.current_stage}

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise


def load_recent_silver(**context):
    """
    Load Silver data của giờ hiện tại.
    Silver đã có đầy đủ features — không cần transform thêm.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Load Recent Silver Data")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    # Silver chạy mỗi 30 phút → lấy giờ hiện tại, fallback giờ trước nếu chưa có
    all_files = []
    for hours_back in range(2):
        check_dt = execution_date - timedelta(hours=hours_back)
        prefix   = (
            f"{S3_SILVER_PREFIX}"
            f"year={check_dt.year}/"
            f"month={check_dt.month:02d}/"
            f"day={check_dt.day:02d}/"
            f"hour={check_dt.hour:02d}/"
        )
        keys = s3_hook.list_keys(bucket_name=S3_BUCKET, prefix=prefix)
        if keys:
            parquet_files = [k for k in keys if k.endswith('.parquet')]
            if parquet_files:
                all_files.extend(parquet_files)
                logger.info(f"  ✅ {prefix}: {len(parquet_files)} files")

        if all_files:
            break  # Lấy giờ gần nhất có data là đủ

    if not all_files:
        logger.warning("⚠️ Không có Silver data — skip inference")
        raise AirflowSkipException("No Silver data available")

    all_data = []
    for file_key in all_files[-3:]:  # 3 files gần nhất trong giờ
        obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
        all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Loaded {len(df_combined):,} records từ Silver")

    # Kiểm tra feature columns
    missing = set(FEATURE_COLS) - set(df_combined.columns)
    if missing:
        raise ValueError(f"❌ Silver data thiếu features: {missing}")

    context['ti'].xcom_push(key='silver_data', value=df_combined.to_dict('records'))
    return len(df_combined)


def run_inference(**context):
    """Scale → Predict → Save Gold"""
    logger.info("=" * 60)
    logger.info("STEP 3: Run Inference")
    logger.info("=" * 60)

    silver_data = context['ti'].xcom_pull(task_ids='load_silver', key='silver_data')
    if not silver_data:
        logger.warning("⚠️ No data to process")
        return None

    df = pd.DataFrame(silver_data)

    # Load model + scaler
    model_uri  = context['ti'].xcom_pull(task_ids='load_model', key='model_uri')
    scaler_uri = context['ti'].xcom_pull(task_ids='load_model', key='scaler_uri')
    run_id     = context['ti'].xcom_pull(task_ids='load_model', key='run_id')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model       = mlflow.sklearn.load_model(model_uri)
    scaler_path = mlflow.artifacts.download_artifacts(scaler_uri)
    scaler      = joblib.load(scaler_path)

    logger.info(f"📦 Model: {model_uri}")
    logger.info(f"📦 Scaler: {scaler_uri}")

    # Load thresholds để log context
    try:
        thresholds_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/label_thresholds.json")
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        logger.info(f"📋 Training thresholds: {thresholds}")
    except Exception:
        logger.warning("⚠️ Không load được label_thresholds.json")

    # ── Align features với scaler ─────────────────────────────────
    if hasattr(scaler, 'feature_names_in_'):
        train_features = list(scaler.feature_names_in_)
        missing = set(train_features) - set(df.columns)
        if missing:
            raise ValueError(f"❌ Missing features: {missing}")
        X = df[train_features]
        logger.info(f"✅ Aligned {len(train_features)} features theo scaler")
    else:
        X = pd.DataFrame(df[FEATURE_COLS].values, columns=FEATURE_COLS)
        logger.info("✅ Using default FEATURE_COLS order")

    # ── Debug ─────────────────────────────────────────────────────
    logger.info(f"📐 Feature matrix: {X.shape}")
    logger.info(f"🔍 Sample (first 3 rows):\n{X.head(3).to_string()}")
    logger.info(f"🔍 Scaler mean:  {dict(zip(X.columns, scaler.mean_.round(3)))}")
    logger.info(f"🔍 Scaler scale: {dict(zip(X.columns, scaler.scale_.round(3)))}")

    # ── Scale → Predict ───────────────────────────────────────────
    X_scaled       = scaler.transform(X)
    predictions    = model.predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)

    df['prediction']    = predictions
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly']    = (predictions == -1).astype(int)

    anomaly_count = int(df['is_anomaly'].sum())
    anomaly_rate  = anomaly_count / len(df)

    logger.info(f"✅ Inference complete: {anomaly_count} anomalies ({anomaly_rate:.2%})")

    # ── Save Gold ─────────────────────────────────────────────────
    now       = datetime.utcnow()
    gold_path = (
        f"{S3_GOLD_PREFIX}"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"hour={now.hour:02d}/"
        f"predictions_{now.strftime('%H%M%S')}.parquet"
    )

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, compression='snappy')
    buffer.seek(0)

    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file_obj(file_obj=buffer, key=gold_path, bucket_name=S3_BUCKET, replace=True)
    logger.info(f"💾 Gold saved: s3://{S3_BUCKET}/{gold_path}")

    # ── Push anomalies để alert ───────────────────────────────────
    if anomaly_count > 0:
        anomaly_cols = [c for c in ['timestamp', 'device_id', 'temperature', 'humidity', 'iaq_score', 'anomaly_score'] if c in df.columns]
        anomaly_df   = df[df['is_anomaly'] == 1][anomaly_cols].copy()

        for col in anomaly_df.columns:
            if pd.api.types.is_datetime64_any_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
            elif pd.api.types.is_float_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].astype(float)
            elif pd.api.types.is_integer_dtype(anomaly_df[col]):
                anomaly_df[col] = anomaly_df[col].astype(int)

        context['ti'].xcom_push(key='anomalies', value=anomaly_df.to_dict('records'))

    return {'total_records': len(df), 'anomaly_count': anomaly_count, 'anomaly_rate': float(anomaly_rate)}


def send_alerts(**context):
    logger.info("=" * 60)
    logger.info("STEP 4: Send Alerts")
    logger.info("=" * 60)

    anomalies = context['ti'].xcom_pull(task_ids='run_inference', key='anomalies')

    if not anomalies:
        logger.info("✅ No anomalies detected")
        return

    logger.warning(f"⚠️ {len(anomalies)} anomalies detected!")

    for anomaly in anomalies[:5]:
        logger.warning(
            f"  🚨 [{anomaly.get('timestamp')}] "
            f"device={anomaly.get('device_id')} | "
            f"temp={anomaly.get('temperature', 'N/A'):.1f}°C | "
            f"humidity={anomaly.get('humidity', 'N/A'):.1f}% | "
            f"iaq={anomaly.get('iaq_score', 'N/A'):.1f} | "
            f"score={anomaly.get('anomaly_score', 'N/A'):.3f}"
        )

    # TODO: Slack / PagerDuty / SNS
    return len(anomalies)


# ==================== DAG ====================

default_args = {
    'owner':            'iot-ml-team',
    'depends_on_past':  False,
    'email_on_failure': True,
    'retries':          1,
    'retry_delay':      timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10)
}

with DAG(
    dag_id='iot_ml_inference_pipeline',
    description='Silver → Inference → Gold',
    # schedule='*/15 * * * *',
    schedule=[SILVER_DATASET], # Inference downstream tự chạy ngay sau mỗi lần Transform upstream ghi Silver xong, không cần quan tâm đến schedule nữa
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'inference', 'real-time', 'anomaly-detection']
) as dag:

    t_model    = PythonOperator(task_id='load_model',   python_callable=load_production_model)
    t_data     = PythonOperator(task_id='load_silver',  python_callable=load_recent_silver)
    t_infer    = PythonOperator(task_id='run_inference', python_callable=run_inference)
    t_alert    = PythonOperator(task_id='send_alerts',  python_callable=send_alerts)

    [t_model, t_data] >> t_infer >> t_alert
