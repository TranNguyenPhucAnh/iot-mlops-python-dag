"""
IoT BME680 ML Training Pipeline v4
====================================
Thay đổi so với v3:
  - [REMOVE] Bỏ generate_synthetic_anomalies hoàn toàn
  - [FIX] contamination tính trực tiếp từ real anomaly_rate
  - [FIX] train/test metrics nhất quán — không còn augmented vs real

Flow: Silver layer → Train → MLflow Registry
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
from sklearn.model_selection import train_test_split
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

N_ESTIMATORS = 100
RANDOM_STATE = 42

# ── Registration thresholds ──────────────────────────────────
MIN_ROC_AUC   = 0.75
MIN_PRECISION = 0.60
MIN_RECALL    = 0.50

# ── Domain-rule thresholds ───────────────────────────────────
DOMAIN_THRESHOLDS = {
    'iaq_score_max':   150.0,
    'temperature_max':  33.0,   # ← tăng từ 31 lên 33
    'temperature_min':  28.0,   # ← hạ từ 27 xuống 28
    'humidity_max':     70.0,   # ← tăng từ 74 lên 70 (thực tế max=67.4)
    'humidity_min':     60.0,   # ← hạ từ 62 xuống 60
}

FEATURE_COLS = [
    'temperature', 'humidity', 'pressure', 'gas_resistance'
]

# ==================== Tasks ====================

def load_silver_data(**context):
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

    missing = set(FEATURE_COLS) - set(df_combined.columns)
    if missing:
        raise ValueError(f"❌ Silver data thiếu feature columns: {missing}")

    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    logger.info("\n📋 Sensor ranges trong 7 ngày (dùng để tune DOMAIN_THRESHOLDS):")
    for col in ['iaq_score', 'temperature', 'humidity', 'gas_resistance', 'pressure']:
        if col in df_combined.columns:
            logger.info(
                f"  {col:25s}: "
                f"min={df_combined[col].min():.1f}, "
                f"p02={df_combined[col].quantile(0.02):.1f}, "
                f"mean={df_combined[col].mean():.1f}, "
                f"p98={df_combined[col].quantile(0.98):.1f}, "
                f"max={df_combined[col].max():.1f}"
            )

    df_combined['is_anomaly'] = (
        (df_combined['iaq_score']   > DOMAIN_THRESHOLDS['iaq_score_max']) |
        (df_combined['temperature'] > DOMAIN_THRESHOLDS['temperature_max']) |
        (df_combined['temperature'] < DOMAIN_THRESHOLDS['temperature_min']) |
        (df_combined['humidity']    > DOMAIN_THRESHOLDS['humidity_max'])    |
        (df_combined['humidity']    < DOMAIN_THRESHOLDS['humidity_min'])
    ).astype(int)

    anomaly_rate = float(df_combined['is_anomaly'].mean())
    logger.info(f"\n📊 Anomaly rate: {anomaly_rate:.2%} ({df_combined['is_anomaly'].sum()} / {len(df_combined)} records)")

    if anomaly_rate == 0:
        raise ValueError(
            "❌ Anomaly rate = 0% — DOMAIN_THRESHOLDS quá cao so với data thực tế.\n"
            f"   iaq_score max thực tế = {df_combined['iaq_score'].max():.1f} "
            f"(threshold hiện tại = {DOMAIN_THRESHOLDS['iaq_score_max']})\n"
            f"   temperature range = [{df_combined['temperature'].min():.1f}, "
            f"{df_combined['temperature'].max():.1f}]\n"
            f"   humidity range = [{df_combined['humidity'].min():.1f}, "
            f"{df_combined['humidity'].max():.1f}]"
        )

    if anomaly_rate > 0.20:
        logger.warning(f"⚠️ Anomaly rate {anomaly_rate:.2%} > 20% — xem xét tăng DOMAIN_THRESHOLDS")

    if anomaly_rate < 0.01:
        logger.warning(f"⚠️ Anomaly rate {anomaly_rate:.2%} < 1% — test set có thể không đủ anomaly")

    logger.info("\n📋 Anomaly breakdown theo condition:")
    logger.info(f"  iaq_score > {DOMAIN_THRESHOLDS['iaq_score_max']}:    "
                f"{(df_combined['iaq_score'] > DOMAIN_THRESHOLDS['iaq_score_max']).sum()}")
    logger.info(f"  temperature > {DOMAIN_THRESHOLDS['temperature_max']}: "
                f"{(df_combined['temperature'] > DOMAIN_THRESHOLDS['temperature_max']).sum()}")
    logger.info(f"  temperature < {DOMAIN_THRESHOLDS['temperature_min']}: "
                f"{(df_combined['temperature'] < DOMAIN_THRESHOLDS['temperature_min']).sum()}")
    logger.info(f"  humidity > {DOMAIN_THRESHOLDS['humidity_max']}:       "
                f"{(df_combined['humidity'] > DOMAIN_THRESHOLDS['humidity_max']).sum()}")
    logger.info(f"  humidity < {DOMAIN_THRESHOLDS['humidity_min']}:       "
                f"{(df_combined['humidity'] < DOMAIN_THRESHOLDS['humidity_min']).sum()}")

    thresholds = {**DOMAIN_THRESHOLDS, 'anomaly_rate': anomaly_rate}

    df_combined['timestamp'] = df_combined['timestamp'].astype(str)
    context['ti'].xcom_push(
        key='silver_data',
        value=df_combined[FEATURE_COLS + ['is_anomaly', 'timestamp']].to_dict('records')
    )
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

    df           = pd.DataFrame(silver_data)
    X            = df[FEATURE_COLS]
    y            = np.array(df['is_anomaly'])
    anomaly_rate = thresholds['anomaly_rate']

    # ── Stratified split — đảm bảo anomaly phân bổ đều train/test ─
    # Time-based split bị lỗi khi anomaly tập trung ở một khoảng thời gian
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"📊 Train: {X_train.shape} | anomaly: {y_train.sum()} ({y_train.mean():.2%})")
    logger.info(f"📊 Test : {X_test.shape}  | anomaly: {y_test.sum()} ({y_test.mean():.2%})")

    if y_test.sum() < 5:
        logger.warning(f"⚠️ Test set chỉ có {y_test.sum()} anomalies — metrics có thể không ổn định")

    # ── Contamination từ real anomaly_rate ───────────────────────
    # Không còn synthetic — contamination chính là tỷ lệ anomaly thực tế
    contamination = float(np.clip(anomaly_rate, 0.01, 0.50))
    # Nới cap lên 0.50 vì với threshold nhạy, anomaly_rate có thể 10-20%
    logger.info(f"📊 Anomaly rate: {anomaly_rate:.2%} → contamination={contamination:.4f}")

    with mlflow.start_run(run_name=f"anomaly_detection_{context['ds_nodash']}") as run:
        run_id = run.info.run_id
        logger.info(f"🔬 MLflow Run ID: {run_id}")

        mlflow.log_params({
            'model_type':          'IsolationForest',
            'contamination':       contamination,
            'n_estimators':        N_ESTIMATORS,
            'random_state':        RANDOM_STATE,
            'train_size':          len(X_train),
            'test_size':           len(X_test),
            'n_features':          len(FEATURE_COLS),
            'anomaly_rate':        anomaly_rate,
            'augmentation':        'none',
            'split_method':        'stratified_random_80_20',
            'label_strategy':      'domain_rules_iaq',
            'training_date':       context['ds'],
            'data_source':         f's3://{S3_BUCKET}/{S3_SILVER_PREFIX}',
            'pipeline_version':    'v4_no_synthetic',
        })
        mlflow.log_params({f"threshold_{k}": v for k, v in thresholds.items()})

        # ── Scale ─────────────────────────────────────────────────
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Train ─────────────────────────────────────────────────
        model = IsolationForest(
            contamination=contamination,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_scaled)
        logger.info("✅ Model training complete")

        # ── Predict ───────────────────────────────────────────────
        test_scores_continuous = -model.score_samples(X_test_scaled)
        y_test_pred            = (model.predict(X_test_scaled) == -1).astype(int)
        y_train_pred           = (model.predict(X_train_scaled) == -1).astype(int)

        # ── Metrics ───────────────────────────────────────────────
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, test_scores_continuous)
        else:
            roc_auc = 0.0
            logger.warning("⚠️ Test set chỉ có 1 class — ROC-AUC = 0. Kiểm tra lại DOMAIN_THRESHOLDS.")

        test_metrics = {
            'test_roc_auc':   roc_auc,
            'test_accuracy':  accuracy_score(y_test,  y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall':    recall_score(y_test,    y_test_pred, zero_division=0),
            'test_f1':        f1_score(y_test,        y_test_pred, zero_division=0),
        }
        train_metrics = {
            'train_accuracy':  accuracy_score(y_train,  y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall':    recall_score(y_train,    y_train_pred, zero_division=0),
            'train_f1':        f1_score(y_train,        y_train_pred, zero_division=0),
        }
        mlflow.log_metrics({**train_metrics, **test_metrics})

        logger.info("\n=== Model Performance ===")
        logger.info(f"  {'Metric':<20} {'Train':>12} {'Test':>12}  {'Threshold':>10}")
        logger.info(f"  {'-'*58}")
        logger.info(f"  {'ROC-AUC':<20} {'N/A':>12} {roc_auc:>12.4f}  {MIN_ROC_AUC:>10}")
        logger.info(f"  {'Precision':<20} {train_metrics['train_precision']:>12.4f} "
                    f"{test_metrics['test_precision']:>12.4f}  {MIN_PRECISION:>10}")
        logger.info(f"  {'Recall':<20} {train_metrics['train_recall']:>12.4f} "
                    f"{test_metrics['test_recall']:>12.4f}  {MIN_RECALL:>10}")
        logger.info(f"  {'F1':<20} {train_metrics['train_f1']:>12.4f} "
                    f"{test_metrics['test_f1']:>12.4f}")
        logger.info(f"  {'Accuracy':<20} {train_metrics['train_accuracy']:>12.4f} "
                    f"{test_metrics['test_accuracy']:>12.4f}")

        # ── Confusion matrix ──────────────────────────────────────
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.title(
            f'Confusion Matrix\n'
            f'ROC-AUC={roc_auc:.3f}  '
            f'Precision={test_metrics["test_precision"]:.3f}  '
            f'Recall={test_metrics["test_recall"]:.3f}'
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/tmp/confusion_matrix.png', dpi=100)
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()

        with open('/tmp/label_thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)
        mlflow.log_artifact('/tmp/label_thresholds.json')

        scaler_path = '/tmp/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path='preprocessor')

        # Sau khi train xong, export ra path cố định để Jenkins lấy
        s3_hook = S3Hook(aws_conn_id='aws_default')
        joblib.dump(model, '/tmp/model.pkl')
        s3_hook.load_file('/tmp/model.pkl',  'models/latest/model.pkl',  bucket_name=S3_BUCKET, replace=True)
        s3_hook.load_file(scaler_path,       'models/latest/scaler.pkl', bucket_name=S3_BUCKET, replace=True)
      
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            signature=mlflow.models.infer_signature(X_train_scaled, y_train_pred)
        )
        mlflow.set_tags({
            'team':             'iot-ml',
            'project':          'bme680-anomaly-detection',
            'environment':      'production',
            'label_strategy':   'domain_rules_iaq',
            'augmentation':     'none',
            'data_layer':       'silver',
            'pipeline_version': 'v4_no_synthetic',
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
    recall    = metrics['test_recall']

    pass_roc_auc   = roc_auc   >= MIN_ROC_AUC
    pass_precision = precision >= MIN_PRECISION
    pass_recall    = recall    >= MIN_RECALL

    logger.info(f"📊 ROC-AUC:   {roc_auc:.4f}  {'✅' if pass_roc_auc   else '❌'} (threshold: {MIN_ROC_AUC})")
    logger.info(f"📊 Precision: {precision:.4f} {'✅' if pass_precision else '❌'} (threshold: {MIN_PRECISION})")
    logger.info(f"📊 Recall:    {recall:.4f}    {'✅' if pass_recall    else '❌'} (threshold: {MIN_RECALL})")

    if metrics['test_f1'] == 0 and roc_auc == 0:
        logger.warning(
            "⚠️ F1 = 0 và ROC-AUC = 0 — test set (20% cuối) có thể không có anomaly nào. "
            "Xem lại sensor ranges và DOMAIN_THRESHOLDS."
        )

    if pass_roc_auc and pass_precision and pass_recall:
        logger.info("✅ Model đủ điều kiện cả 3 metrics — sẽ register vào Staging")
        return 'register_model'

    failed = [
        name for name, passed in [
            ('ROC-AUC', pass_roc_auc),
            ('Precision', pass_precision),
            ('Recall', pass_recall)
        ] if not passed
    ]
    logger.warning(f"⚠️ Model chưa đủ điều kiện — failed: {', '.join(failed)}")
    return 'skip_registration'


def register_model(**context):
    logger.info("=" * 60)
    logger.info("STEP 4: Registering Model")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=REGISTERED_MODEL_NAME
    )
    version = model_version.version
    logger.info(f"✅ Registered: {REGISTERED_MODEL_NAME} version {version}")

    thresholds = context['ti'].xcom_pull(task_ids='load_silver_data', key='label_thresholds')
    metrics    = context['ti'].xcom_pull(task_ids='train_model',       key='model_metrics')

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"Trained on {context['ds']} | Silver layer | IsolationForest | "
            f"Label: domain_rules_iaq | No synthetic augmentation | "
            f"ROC-AUC={metrics['test_roc_auc']:.3f} "
            f"Precision={metrics['test_precision']:.3f} "
            f"Recall={metrics['test_recall']:.3f} | "
            f"Thresholds: iaq>{thresholds['iaq_score_max']} "
            f"temp=[{thresholds['temperature_min']},{thresholds['temperature_max']}] "
            f"hum=[{thresholds['humidity_min']},{thresholds['humidity_max']}]"
        )
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
            logger.info(f"ℹ️ Production hiện tại: version {prod_versions[0].version}")
            logger.info(f"   New Staging version {version} cần manual promotion sau khi validate")
        else:
            logger.info("ℹ️ Chưa có Production model — promote Staging lên Production thủ công sau khi validate")
    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")

    context['ti'].xcom_push(key='model_version', value=version)
    return {'model_name': REGISTERED_MODEL_NAME, 'version': version, 'stage': 'Staging'}


def send_notification(**context):
    logger.info("=" * 60)
    logger.info("STEP 5: Sending Notification")
    logger.info("=" * 60)

    run_id        = context['ti'].xcom_pull(task_ids='train_model',      key='run_id')
    metrics       = context['ti'].xcom_pull(task_ids='train_model',      key='model_metrics')
    thresholds    = context['ti'].xcom_pull(task_ids='load_silver_data', key='label_thresholds')
    model_version = context['ti'].xcom_pull(task_ids='register_model',   key='model_version')

    registered  = model_version is not None
    status_icon = "🎉" if registered else "⚠️"

    message = f"""
{status_icon} IoT ML Training Pipeline v4 — {'REGISTERED' if registered else 'NOT REGISTERED'}

📅 Training Date : {context['ds']}
📦 Data Source   : Silver layer (domain-rule labels, no synthetic)
🔬 MLflow Run ID : {run_id}

📊 Model Performance:
   ROC-AUC   : {metrics['test_roc_auc']:.4f}  (threshold: {MIN_ROC_AUC})
   Precision : {metrics['test_precision']:.4f}  (threshold: {MIN_PRECISION})
   Recall    : {metrics['test_recall']:.4f}  (threshold: {MIN_RECALL})
   F1 Score  : {metrics['test_f1']:.4f}
   Accuracy  : {metrics['test_accuracy']:.4f}

🏷️  Label Strategy : domain_rules_iaq (v4 — no synthetic augmentation)
   iaq_score   > {thresholds.get('iaq_score_max')}
   temperature > {thresholds.get('temperature_max')} or < {thresholds.get('temperature_min')}
   humidity    > {thresholds.get('humidity_max')} or < {thresholds.get('humidity_min')}
   Anomaly rate: {thresholds.get('anomaly_rate', 0):.2%}

📦 Model Registry:
   Name    : {REGISTERED_MODEL_NAME}
   Version : {model_version if registered else 'Not registered'}
   Stage   : {'Staging (manual promotion to Production required)' if registered else 'N/A'}

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
    description='Silver → Train (domain-rule labels, no synthetic) → MLflow Registry | v4',
    schedule='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'iot', 'anomaly-detection', 'mlflow', 'production', 'v4']
) as dag:

    t_load   = PythonOperator(task_id='load_silver_data',         python_callable=load_silver_data)
    t_train  = PythonOperator(task_id='train_model',              python_callable=train_anomaly_model)
    t_decide = BranchPythonOperator(task_id='decide_registration', python_callable=decide_model_registration)
    t_reg    = PythonOperator(task_id='register_model',           python_callable=register_model)
    t_skip   = EmptyOperator(task_id='skip_registration')
    t_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='none_failed'
    )

    t_load >> t_train >> t_decide >> [t_reg, t_skip] >> t_notify
