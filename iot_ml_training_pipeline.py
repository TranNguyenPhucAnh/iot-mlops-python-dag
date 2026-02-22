"""
IoT BME680 ML Training Pipeline v2
====================================
Flow mới: Silver layer → Train → MLflow
Feature engineering đã được Transform DAG xử lý.

Schedule: Daily at 2 AM UTC
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import joblib

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
MLFLOW_TRACKING_URI   = "http://mlflow.mlflow.svc.cluster.local:80"
EXPERIMENT_NAME       = "iot_bme680_anomaly_detection"
S3_BUCKET             = "iot-bme680-data-lake-prod"
S3_SILVER_PREFIX      = "silver/bme680_features/"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"

N_ESTIMATORS  = 100
RANDOM_STATE  = 42
MIN_ROC_AUC   = 0.70
MIN_PRECISION = 0.50

FEATURE_COLS = [
    'temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score',
    'hour', 'day_of_week', 'is_weekend',
    'temp_humidity_ratio', 'gas_pressure_ratio',
    'temperature_rolling_mean', 'temperature_rolling_std',
    'humidity_rolling_mean', 'humidity_rolling_std',
    'iaq_score_rolling_mean', 'iaq_score_rolling_std'
]

# ==================== Tasks ====================

def load_silver_data(**context):
    """
    Đọc Silver data từ 7 ngày gần nhất — tất cả hour.
    Silver đã clean + feature engineering sẵn từ Transform DAG.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Silver Data")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    all_files = []
    for days_back in range(7):
        check_date = execution_date - timedelta(days=days_back)

        for hour in range(24):
            prefix = (
                f"{S3_SILVER_PREFIX}"
                f"year={check_date.year}/"
                f"month={check_date.month:02d}/"
                f"day={check_date.day:02d}/"
                f"hour={hour:02d}/"
            )
            keys = s3_hook.list_keys(bucket_name=S3_BUCKET, prefix=prefix)
            if keys:
                parquet_files = [k for k in keys if k.endswith('.parquet')]
                if parquet_files:
                    all_files.extend(parquet_files)
                    logger.info(f"  ✅ {check_date.date()} hour={hour:02d}: {len(parquet_files)} files")

    if not all_files:
        raise ValueError("❌ Không có Silver data trong 7 ngày gần nhất")

    logger.info(f"📁 Tổng Silver files: {len(all_files)}")

    all_data = []
    for file_key in all_files:
        obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
        all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Tổng records từ Silver: {len(df_combined):,}")
    logger.info(f"📊 Columns: {df_combined.columns.tolist()}")

    # Kiểm tra feature columns đủ không
    missing = set(FEATURE_COLS) - set(df_combined.columns)
    if missing:
        raise ValueError(f"❌ Silver data thiếu feature columns: {missing}")

    # Sort theo thời gian
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    # ── Label dựa trên phân phối thực tế (percentile) ────────────
    gas_low  = df_combined['gas_resistance'].quantile(0.02)
    iaq_high = df_combined['iaq_score'].quantile(0.98)
    temp_hi  = df_combined['temperature'].quantile(0.98)
    temp_lo  = df_combined['temperature'].quantile(0.02)
    hum_hi   = df_combined['humidity'].quantile(0.98)

    df_combined['is_anomaly'] = (
        (df_combined['gas_resistance'] < gas_low) |
        (df_combined['iaq_score']      > iaq_high) |
        (df_combined['temperature']    > temp_hi)  |
        (df_combined['temperature']    < temp_lo)  |
        (df_combined['humidity']       > hum_hi)
    ).astype(int)

    anomaly_rate = float(df_combined['is_anomaly'].mean())
    logger.info(f"📊 Anomaly rate: {anomaly_rate:.2%}")

    if anomaly_rate < 0.02:
        logger.warning("⚠️ Anomaly rate < 2% — data có thể quá uniform")

    thresholds = {
        'gas_resistance_p02': float(gas_low),
        'iaq_score_p98':      float(iaq_high),
        'temperature_p02':    float(temp_lo),
        'temperature_p98':    float(temp_hi),
        'humidity_p98':       float(hum_hi),
        'anomaly_rate':       anomaly_rate
    }
    
    # ✅ Convert timestamp sang string TRƯỚC khi push XCom
    df_combined['timestamp'] = df_combined['timestamp'].astype(str)

    context['ti'].xcom_push(key='silver_data',    value=df_combined[FEATURE_COLS + ['is_anomaly', 'timestamp']].to_dict('records'))
    context['ti'].xcom_push(key='label_thresholds', value=thresholds)

    return {'record_count': len(df_combined), 'anomaly_rate': anomaly_rate}

def train_anomaly_model(**context):
    logger.info("=" * 60)
    logger.info("STEP 2: Training Anomaly Detection Model")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    silver_data = context['ti'].xcom_pull(task_ids='load_silver_data', key='silver_data')
    thresholds  = context['ti'].xcom_pull(task_ids='load_silver_data', key='label_thresholds')

    df = pd.DataFrame(silver_data)
    X  = df[FEATURE_COLS]
    y  = np.array(df['is_anomaly'])

    anomaly_rate  = thresholds['anomaly_rate']
    contamination = float(np.clip(anomaly_rate, 0.01, 0.45))
    logger.info(f"📊 Anomaly rate: {anomaly_rate:.2%} → contamination={contamination:.4f}")

    # Split theo thời gian
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

    if y_test.sum() < 5:
        logger.warning(f"⚠️ Test set chỉ có {y_test.sum()} anomalies — metrics có thể không ổn định")

    with mlflow.start_run(run_name=f"anomaly_detection_{context['ds_nodash']}") as run:
        run_id = run.info.run_id
        logger.info(f"🔬 MLflow Run ID: {run_id}")

        mlflow.log_params({
            'model_type':    'IsolationForest',
            'contamination': contamination,
            'n_estimators':  N_ESTIMATORS,
            'random_state':  RANDOM_STATE,
            'train_size':    len(X_train),
            'test_size':     len(X_test),
            'n_features':    len(FEATURE_COLS),
            'anomaly_rate':  anomaly_rate,
            'split_method':  'time_based',
            'training_date': context['ds'],
            'data_source':   f's3://{S3_BUCKET}/{S3_SILVER_PREFIX}'
        })
        mlflow.log_params({f"threshold_{k}": v for k, v in thresholds.items()})

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_scaled)
        logger.info("✅ Model training complete")

        test_scores_continuous = -model.score_samples(X_test_scaled)
        y_test_pred_binary     = (model.predict(X_test_scaled) == -1).astype(int)
        y_train_pred_binary    = (model.predict(X_train_scaled) == -1).astype(int)

        roc_auc = roc_auc_score(y_test, test_scores_continuous) \
            if len(np.unique(y_test)) == 2 else 0.0

        if roc_auc == 0.0:
            logger.warning("⚠️ Test set chỉ có 1 class — ROC-AUC = 0")

        test_metrics = {
            'test_roc_auc':   roc_auc,
            'test_accuracy':  accuracy_score(y_test, y_test_pred_binary),
            'test_precision': precision_score(y_test, y_test_pred_binary, zero_division=0),
            'test_recall':    recall_score(y_test, y_test_pred_binary, zero_division=0),
            'test_f1':        f1_score(y_test, y_test_pred_binary, zero_division=0)
        }
        train_metrics = {
            'train_accuracy':  accuracy_score(y_train, y_train_pred_binary),
            'train_precision': precision_score(y_train, y_train_pred_binary, zero_division=0),
            'train_recall':    recall_score(y_train, y_train_pred_binary, zero_division=0),
            'train_f1':        f1_score(y_train, y_train_pred_binary, zero_division=0)
        }
        mlflow.log_metrics({**train_metrics, **test_metrics})

        logger.info("\n=== Model Performance ===")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Confusion matrix
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_test_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (ROC-AUC={roc_auc:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('/tmp/confusion_matrix.png')
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()

        # Log thresholds artifact
        with open('/tmp/label_thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)
        mlflow.log_artifact('/tmp/label_thresholds.json')

        # Log scaler
        scaler_path = '/tmp/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path='preprocessor')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            signature=mlflow.models.infer_signature(X_train_scaled, y_train_pred_binary)
        )
        mlflow.set_tags({
            'team':           'iot-ml',
            'project':        'bme680-anomaly-detection',
            'environment':    'production',
            'label_strategy': 'percentile_based',
            'data_layer':     'silver'
        })

        context['ti'].xcom_push(key='model_metrics', value=test_metrics)
        context['ti'].xcom_push(key='run_id',        value=run_id)

        return {'run_id': run_id, 'test_roc_auc': roc_auc}

def decide_model_registration(**context):
    logger.info("=" * 60)
    logger.info("STEP 3: Model Registration Decision")
    logger.info("=" * 60)

    metrics   = context['ti'].xcom_pull(task_ids='train_model', key='model_metrics')
    roc_auc   = metrics['test_roc_auc']
    precision = metrics['test_precision']

    logger.info(f"📊 ROC-AUC:   {roc_auc:.4f}  (threshold: {MIN_ROC_AUC})")
    logger.info(f"📊 Precision: {precision:.4f} (threshold: {MIN_PRECISION})")

    if metrics['test_f1'] == 0 and roc_auc == 0:
        logger.warning("⚠️ Cả F1 lẫn ROC-AUC = 0 — test set có thể không có anomaly")

    if roc_auc >= MIN_ROC_AUC and precision >= MIN_PRECISION:
        logger.info("✅ Model đủ điều kiện — sẽ register")
        return 'register_model'

    logger.warning("⚠️ Model chưa đủ điều kiện — bỏ qua registration")
    return 'skip_registration'


def register_model(**context):
    logger.info("=" * 60)
    logger.info("STEP 4: Registering Model")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client  = MlflowClient()
    run_id  = context['ti'].xcom_pull(task_ids='train_model', key='run_id')

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=REGISTERED_MODEL_NAME
    )
    version = model_version.version
    logger.info(f"✅ Registered: {REGISTERED_MODEL_NAME} version {version}")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=f"Trained on {context['ds']} from Silver layer using Isolation Forest"
    )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    logger.info(f"✅ Version {version} → Staging")

    try:
        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if prod_versions:
            logger.info(f"ℹ️ Current production version: {prod_versions[0].version}")
            logger.info("ℹ️ New model in Staging — manual promotion required")
        else:
            logger.info("ℹ️ No production model yet")
    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")

    context['ti'].xcom_push(key='model_version', value=version)
    return {'model_name': REGISTERED_MODEL_NAME, 'version': version, 'stage': 'Staging'}

def send_notification(**context):
    logger.info("=" * 60)
    logger.info("STEP 5: Sending Notification")
    logger.info("=" * 60)

    run_id        = context['ti'].xcom_pull(task_ids='train_model',    key='run_id')
    metrics       = context['ti'].xcom_pull(task_ids='train_model',    key='model_metrics')
    model_version = context['ti'].xcom_pull(task_ids='register_model', key='model_version')

    message = f"""
🎉 IoT ML Training Pipeline Completed

📅 Training Date: {context['ds']}
📦 Data Source:   Silver layer
🔬 MLflow Run ID: {run_id}

📊 Model Performance:
   - ROC-AUC:   {metrics['test_roc_auc']:.4f}
   - F1 Score:  {metrics['test_f1']:.4f}
   - Precision: {metrics['test_precision']:.4f}
   - Recall:    {metrics['test_recall']:.4f}
   - Accuracy:  {metrics['test_accuracy']:.4f}

📦 Model Registry:
   - Name:    {REGISTERED_MODEL_NAME}
   - Version: {model_version if model_version else 'Not registered'}
   - Stage:   {'Staging' if model_version else 'N/A'}

🔗 MLflow UI: {MLFLOW_TRACKING_URI}
"""
    logger.info(message)
    return message

# ==================== DAG ====================

default_args = {
    'owner':             'iot-ml-team',
    'depends_on_past':   False,
    'email':             ['iot-ml@company.com'],
    'email_on_failure':  True,
    'email_on_retry':    False,
    'retries':           2,
    'retry_delay':       timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

with DAG(
    dag_id='iot_ml_training_pipeline',
    description='Silver → Train → MLflow Registry',
    schedule='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'iot', 'anomaly-detection', 'mlflow', 'production']
) as dag:

    t_load   = PythonOperator(task_id='load_silver_data',   python_callable=load_silver_data)
    t_train  = PythonOperator(task_id='train_model',        python_callable=train_anomaly_model)
    t_decide = BranchPythonOperator(task_id='decide_registration', python_callable=decide_model_registration)
    t_reg    = PythonOperator(task_id='register_model',     python_callable=register_model)
    t_skip   = EmptyOperator(task_id='skip_registration')
    t_notify = PythonOperator(task_id='send_notification',  python_callable=send_notification, trigger_rule='none_failed')

    t_load >> t_train >> t_decide >> [t_reg, t_skip] >> t_notify
