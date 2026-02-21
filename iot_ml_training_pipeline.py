"""
IoT BME680 ML Training Pipeline
================================
End-to-end ML workflow:
1. Extract data from S3 Bronze (Parquet)
2. Feature engineering & data quality checks
3. Train anomaly detection model
4. Log model + metrics to MLflow
5. Register best model to MLflow Model Registry
6. Deploy model for inference

Schedule: Daily at 2 AM UTC
Author: IoT ML Team
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
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:80"
EXPERIMENT_NAME = "iot_bme680_anomaly_detection"
S3_BUCKET = "iot-bme680-data-lake-prod"
S3_BRONZE_PREFIX = "bronze/bme680/"
S3_SILVER_PREFIX = "silver/bme680_features/"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"

N_ESTIMATORS = 100
RANDOM_STATE = 42

# Đổi threshold sang ROC-AUC thay F1
MIN_ROC_AUC = 0.70
MIN_PRECISION = 0.50  # Hạ xuống cho phù hợp thực tế

# ==================== Helper Functions ====================

def get_latest_partition_path(**context):
    """Find latest data partition in S3 Bronze layer"""
    logger.info("=" * 60)
    logger.info("STEP 1: Finding latest data partition")
    logger.info("=" * 60)
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')
    
    # Look for data from last 7 days
    partitions = []
    for days_back in range(7):
        check_date = execution_date - timedelta(days=days_back)
        prefix = (
            f"{S3_BRONZE_PREFIX}"
            f"year={check_date.year}/"
            f"month={check_date.month:02d}/"
            f"day={check_date.day:02d}/"
        )
        
        keys = s3_hook.list_keys(
            bucket_name=S3_BUCKET,
            prefix=prefix
        )
        
        if keys:
            parquet_files = [k for k in keys if k.endswith('.parquet')]
            if parquet_files:
                partitions.append({
                    'date': check_date,
                    'prefix': prefix,
                    'files': parquet_files,
                    'file_count': len(parquet_files)
                })
                logger.info(f"✅ Found partition: {prefix} ({len(parquet_files)} files)")
    
    if not partitions:
        logger.error("❌ No data found in last 7 days")
        raise ValueError("No training data available")
    
    # Use most recent partition
    latest = partitions[0]
    logger.info(f"📅 Using latest partition: {latest['date'].date()}")
    logger.info(f"📁 Files: {latest['file_count']}")
    
    # Push to XCom
    context['ti'].xcom_push(key='data_partition', value=latest)
    return latest

def extract_and_validate_data(**context):
    """Extract Parquet files from S3 and validate data quality"""
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting & Validating Data")
    logger.info("=" * 60)
    
    partition = context['ti'].xcom_pull(task_ids='get_latest_partition', key='data_partition')
    s3_hook = S3Hook(aws_conn_id='aws_default')
    
    # Read all Parquet files
    all_data = []
    for file_key in partition['files']:
        logger.info(f"📥 Reading: s3://{S3_BUCKET}/{file_key}")
        
        obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        parquet_bytes = obj.get()['Body'].read()
        
        df = pd.read_parquet(BytesIO(parquet_bytes))
        all_data.append(df)
    
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Total records: {len(df_combined):,}")
    logger.info(f"📊 Columns: {df_combined.columns.tolist()}")
    
    # Data Quality Checks
    logger.info("\n=== Data Quality Checks ===")
    
    required_columns = ['temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score']
    missing_cols = [col for col in required_columns if col not in df_combined.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info("✅ All required columns present")
    
    # Check for nulls
    null_counts = df_combined[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"⚠️ Null values found:\n{null_counts[null_counts > 0]}")
        df_combined = df_combined.dropna(subset=required_columns)
        logger.info(f"Dropped nulls, remaining records: {len(df_combined):,}")
    else:
        logger.info("✅ No null values")
    
    # Check value ranges
    validation_rules = {
        'temperature': (0, 85),
        'humidity': (0, 100),
        'pressure': (800, 1200),
        'gas_resistance': (100, 500000)
    }
    
    invalid_records = 0
    for col, (min_val, max_val) in validation_rules.items():
        mask = (df_combined[col] < min_val) | (df_combined[col] > max_val)
        invalid_count = mask.sum()
        if invalid_count > 0:
            logger.warning(f"⚠️ {col}: {invalid_count} values out of range [{min_val}, {max_val}]")
            invalid_records += invalid_count
    
    if invalid_records > len(df_combined) * 0.1:  # >10% invalid
        raise ValueError(f"Too many invalid records: {invalid_records}")
    
    # Remove outliers
    df_clean = df_combined[
        (df_combined['temperature'].between(0, 85)) &
        (df_combined['humidity'].between(0, 100)) &
        (df_combined['pressure'].between(800, 1200)) &
        (df_combined['gas_resistance'].between(100, 500000))
    ]
    
    logger.info(f"✅ Data validated: {len(df_clean):,} clean records")
    
    # Data statistics
    logger.info("\n=== Data Statistics ===")
    logger.info(f"\n{df_clean[required_columns].describe()}")
    
    # Save to XCom (for small datasets) or S3 (for large)
    if len(df_clean) < 10000:
        context['ti'].xcom_push(key='clean_data', value=df_clean.to_dict('records'))
    else:
        # Save to S3 silver layer
        silver_path = f"{S3_SILVER_PREFIX}train_data_{context['ds_nodash']}.parquet"
        buffer = BytesIO()
        df_clean.to_parquet(buffer, index=False, compression='snappy')
        buffer.seek(0)
        
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=silver_path,
            bucket_name=S3_BUCKET,
            replace=True
        )
        logger.info(f"💾 Saved to s3://{S3_BUCKET}/{silver_path}")
        context['ti'].xcom_push(key='silver_data_path', value=silver_path)
    
    # Push metrics
    metrics = {
        'total_records': len(df_combined),
        'clean_records': len(df_clean),
        'dropped_records': len(df_combined) - len(df_clean),
        'data_quality_score': len(df_clean) / len(df_combined)
    }
    context['ti'].xcom_push(key='data_metrics', value=metrics)
    
    return metrics

def feature_engineering(**context):
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)
    
    clean_data = context['ti'].xcom_pull(task_ids='extract_validate_data', key='clean_data')
    silver_path = context['ti'].xcom_pull(task_ids='extract_validate_data', key='silver_data_path')
    
    if clean_data:
        df = pd.DataFrame(clean_data)
    else:
        s3_hook = S3Hook(aws_conn_id='aws_default')
        obj = s3_hook.get_key(silver_path, bucket_name=S3_BUCKET)
        df = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
    
    logger.info(f"📊 Engineering features for {len(df):,} records")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
    df['gas_pressure_ratio'] = df['gas_resistance'] / (df['pressure'] + 1e-6)
    
    df = df.sort_values('timestamp')
    for col in ['temperature', 'humidity', 'iaq_score']:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10, min_periods=1).std().fillna(0)
    
    # ============================================================
    # ✅ THAY ĐỔI CHÍNH: Label dựa trên phân phối thực tế
    # ============================================================
    logger.info("\n=== Sensor Statistics (dùng để tính ngưỡng) ===")
    for col in ['temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score']:
        logger.info(
            f"  {col}: min={df[col].min():.1f}, "
            f"mean={df[col].mean():.1f}, "
            f"max={df[col].max():.1f}, "
            f"p5={df[col].quantile(0.05):.1f}, "
            f"p95={df[col].quantile(0.95):.1f}"
        )
    
    # Thay vì p5/p95 (luôn tạo ~10% anomaly)
    # Dùng p2/p98 hoặc p1/p99 — chỉ label những điểm thực sự cực đoan
    gas_low_threshold  = df['gas_resistance'].quantile(0.02)
    iaq_high_threshold = df['iaq_score'].quantile(0.98)
    temp_high          = df['temperature'].quantile(0.98)
    temp_low           = df['temperature'].quantile(0.02)
    humidity_high      = df['humidity'].quantile(0.98)
    
    df['is_anomaly'] = (
        (df['gas_resistance'] < gas_low_threshold) |   # Có mùi lạ / khí lạ
        (df['iaq_score'] > iaq_high_threshold)       |   # Không khí kém
        (df['temperature'] > temp_high)               |   # Nhiệt độ bất thường cao
        (df['temperature'] < temp_low)                |   # Nhiệt độ bất thường thấp
        (df['humidity'] > humidity_high)                   # Độ ẩm bất thường
    ).astype(int)
    
    anomaly_rate = float(df['is_anomaly'].mean())
    logger.info(f"\n✅ Anomaly rate với percentile label: {anomaly_rate:.2%}")
    logger.info(f"   gas_resistance threshold (p5):  {gas_low_threshold:.0f}")
    logger.info(f"   iaq_score threshold (p95):      {iaq_high_threshold:.1f}")
    logger.info(f"   temperature range: [{temp_low:.1f}, {temp_high:.1f}]")
    logger.info(f"   humidity threshold (p97):       {humidity_high:.1f}")
    
    # Cảnh báo nếu data quá uniform (sensor chưa gặp sự kiện bất thường)
    if anomaly_rate < 0.02:
        logger.warning(
            "⚠️ Anomaly rate < 2% — data có thể quá uniform. "
            "Cân nhắc thu thập thêm data có bất thường (xịt nước hoa, thay đổi nhiệt độ...)"
        )
    
    # Lưu thresholds để dùng trong inference (quan trọng!)
    thresholds = {
        'gas_resistance_p5':  float(gas_low_threshold),
        'iaq_score_p95':      float(iaq_high_threshold),
        'temperature_p03':    float(temp_low),
        'temperature_p97':    float(temp_high),
        'humidity_p97':       float(humidity_high),
        'anomaly_rate':       anomaly_rate
    }
    context['ti'].xcom_push(key='label_thresholds', value=thresholds)
    
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score',
        'hour', 'day_of_week', 'is_weekend',
        'temp_humidity_ratio', 'gas_pressure_ratio',
        'temperature_rolling_mean', 'temperature_rolling_std',
        'humidity_rolling_mean', 'humidity_rolling_std',
        'iaq_score_rolling_mean', 'iaq_score_rolling_std'
    ]
    
    feature_data = {
        'features': df[feature_cols].to_dict('records'),
        'labels': df['is_anomaly'].tolist(),
        'feature_names': feature_cols,
        'timestamps': df['timestamp'].astype(str).tolist()
    }
    context['ti'].xcom_push(key='feature_data', value=feature_data)
    
    return {
        'feature_count': len(feature_cols),
        'record_count': len(df),
        'anomaly_rate': anomaly_rate,
        'thresholds': thresholds
    }

def train_anomaly_model(**context):
    logger.info("=" * 60)
    logger.info("STEP 4: Training Anomaly Detection Model")
    logger.info("=" * 60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    feature_data  = context['ti'].xcom_pull(task_ids='feature_engineering', key='feature_data')
    thresholds    = context['ti'].xcom_pull(task_ids='feature_engineering', key='label_thresholds')
    
    X = pd.DataFrame(feature_data['features'])
    y = np.array(feature_data['labels'])
    feature_names = feature_data['feature_names']
    
    # ✅ Contamination động theo anomaly rate thực tế
    anomaly_rate  = thresholds['anomaly_rate']
    contamination = float(np.clip(anomaly_rate, 0.01, 0.45))
    logger.info(f"📊 Anomaly rate: {anomaly_rate:.2%} → contamination={contamination:.4f}")
    
    # ✅ Split theo thời gian, không random (tránh data leakage từ rolling features)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Kiểm tra test set có đủ anomaly không
    if y_test.sum() < 5:
        logger.warning(
            f"⚠️ Test set chỉ có {y_test.sum()} anomalies — "
            "metrics có thể không ổn định. Cần thêm data."
        )
    
    with mlflow.start_run(run_name=f"anomaly_detection_{context['ds_nodash']}") as run:
        run_id = run.info.run_id
        logger.info(f"🔬 MLflow Run ID: {run_id}")
        
        params = {
            'model_type':    'IsolationForest',
            'contamination': contamination,
            'n_estimators':  N_ESTIMATORS,
            'random_state':  RANDOM_STATE,
            'train_size':    len(X_train),
            'test_size':     len(X_test),
            'n_features':    len(feature_names),
            'anomaly_rate':  anomaly_rate,
            'split_method':  'time_based'  # Ghi rõ để tracking
        }
        mlflow.log_params(params)
        mlflow.log_param('training_date', context['ds'])
        
        # Log thresholds dùng để label
        mlflow.log_params({f"threshold_{k}": v for k, v in thresholds.items()})
        
        scaler = StandardScaler()
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
        
        # ✅ Dùng anomaly score liên tục để tính ROC-AUC (phù hợp hơn F1)
        from sklearn.metrics import (
            roc_auc_score, accuracy_score,
            precision_score, recall_score, f1_score
        )
        
        # score_samples trả về giá trị càng âm = càng anomaly
        # Negate để chiều dương = anomaly (đúng convention của roc_auc)
        test_scores_continuous = -model.score_samples(X_test_scaled)
        
        y_test_pred_binary = (model.predict(X_test_scaled) == -1).astype(int)
        y_train_pred_binary = (model.predict(X_train_scaled) == -1).astype(int)
        
        # ROC-AUC chỉ tính được khi có cả 2 class
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, test_scores_continuous)
        else:
            logger.warning("⚠️ Test set chỉ có 1 class, ROC-AUC không tính được, gán = 0")
            roc_auc = 0.0
        
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
        
        # Confusion matrix & artifacts (giữ nguyên)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_test_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (ROC-AUC={roc_auc:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('/tmp/confusion_matrix.png')
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()
        
        # ✅ Log thresholds dưới dạng artifact để inference pipeline tham chiếu
        import json
        with open('/tmp/label_thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)
        mlflow.log_artifact('/tmp/label_thresholds.json')
        
        scaler_path = '/tmp/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path='preprocessor')
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            signature=mlflow.models.infer_signature(X_train_scaled, y_train_pred_binary)
        )
        
        mlflow.set_tags({
            'team': 'iot-ml',
            'project': 'bme680-anomaly-detection',
            'environment': 'production',
            'label_strategy': 'percentile_based'  # ✅ Tag rõ strategy
        })
        
        context['ti'].xcom_push(key='model_metrics', value=test_metrics)
        context['ti'].xcom_push(key='run_id', value=run_id)
        
        return {'run_id': run_id, 'test_roc_auc': roc_auc}

def decide_model_registration(**context):
    logger.info("=" * 60)
    logger.info("STEP 5: Model Registration Decision")
    logger.info("=" * 60)
    
    metrics = context['ti'].xcom_pull(task_ids='train_model', key='model_metrics')
    
    roc_auc   = metrics['test_roc_auc']
    precision = metrics['test_precision']
    
    logger.info(f"📊 ROC-AUC:  {roc_auc:.4f}  (threshold: {MIN_ROC_AUC})")
    logger.info(f"📊 Precision: {precision:.4f} (threshold: {MIN_PRECISION})")
    
    # Nếu test set không có anomaly → ROC-AUC = 0 → cảnh báo rõ ràng
    if metrics['test_f1'] == 0 and roc_auc == 0:
        logger.warning(
            "⚠️ Cả F1 lẫn ROC-AUC = 0. Khả năng cao test set không có anomaly. "
            "Cần thu thập thêm data có sự kiện bất thường trước khi train."
        )
    
    if roc_auc >= MIN_ROC_AUC and precision >= MIN_PRECISION:
        logger.info("✅ Model đủ điều kiện — sẽ register")
        return 'register_model'
    else:
        logger.warning("⚠️ Model chưa đủ điều kiện — bỏ qua registration")
        return 'skip_registration'
        
def register_model(**context):
    """Register model to MLflow Model Registry"""
    logger.info("=" * 60)
    logger.info("STEP 6: Registering Model to MLflow Registry")
    logger.info("=" * 60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    logger.info(f"📝 Registering model from run: {run_id}")
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )
    
    version = model_version.version
    logger.info(f"✅ Registered as {REGISTERED_MODEL_NAME} version {version}")
    
    # Add model version description
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=f"Anomaly detection model trained on {context['ds']} using Isolation Forest"
    )
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    
    logger.info(f"✅ Model version {version} transitioned to Staging")
    
    # Compare with production model (if exists)
    try:
        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0]
            logger.info(f"ℹ️ Current production version: {prod_version.version}")
            
            # Auto-promote if better (in real scenario, add A/B testing)
            # For now, keep in Staging for manual review
            logger.info("ℹ️ New model in Staging - manual promotion required")
        else:
            logger.info("ℹ️ No production model - consider promoting after validation")
    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")
    
    context['ti'].xcom_push(key='model_version', value=version)
    
    return {
        'model_name': REGISTERED_MODEL_NAME,
        'version': version,
        'stage': 'Staging'
    }

def send_notification(**context):
    """Send training completion notification"""
    logger.info("=" * 60)
    logger.info("STEP 7: Sending Notification")
    logger.info("=" * 60)
    
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')
    metrics = context['ti'].xcom_pull(task_ids='train_model', key='model_metrics')
    model_version = context['ti'].xcom_pull(task_ids='register_model', key='model_version')
    
    message = f"""
🎉 IoT ML Pipeline Completed Successfully

📅 Training Date: {context['ds']}
🔬 MLflow Run ID: {run_id}
📊 Model Performance:
   - F1 Score: {metrics['test_f1']:.4f}
   - Precision: {metrics['test_precision']:.4f}
   - Recall: {metrics['test_recall']:.4f}
   - Accuracy: {metrics['test_accuracy']:.4f}

📦 Model Registry:
   - Name: {REGISTERED_MODEL_NAME}
   - Version: {model_version if model_version else 'Not registered'}
   - Stage: {'Staging' if model_version else 'N/A'}

🔗 MLflow UI: {MLFLOW_TRACKING_URI}

Next steps:
1. Review model in MLflow UI
2. Validate model performance on staging data
3. Promote to Production if satisfactory
"""
    
    logger.info(message)
    
    # TODO: Send to Slack/Email
    # slack_webhook = Variable.get("slack_webhook_url")
    # requests.post(slack_webhook, json={"text": message})
    
    return message

# ==================== DAG Definition ====================

default_args = {
    'owner': 'iot-ml-team',
    'depends_on_past': False,
    'email': ['iot-ml@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

with DAG(
    dag_id='iot_ml_training_pipeline',
    description='End-to-end ML pipeline: Data → Features → Training → MLflow',
    schedule='0 2 * * *',  # Daily at 2 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'iot', 'anomaly-detection', 'mlflow', 'production']
) as dag:

    # Pipeline Tasks
    get_partition = PythonOperator(
        task_id='get_latest_partition',
        python_callable=get_latest_partition_path
    )

    extract_validate = PythonOperator(
        task_id='extract_validate_data',
        python_callable=extract_and_validate_data
    )

    engineer_features = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_anomaly_model
    )

    decide_registration = BranchPythonOperator(
        task_id='decide_registration',
        python_callable=decide_model_registration
    )

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model
    )

    skip_registration = EmptyOperator(
        task_id='skip_registration'
    )

    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='none_failed'  # Run even if registration skipped
    )

    # Pipeline Flow
    get_partition >> extract_validate >> engineer_features >> train
    train >> decide_registration >> [register, skip_registration]
    [register, skip_registration] >> notify
