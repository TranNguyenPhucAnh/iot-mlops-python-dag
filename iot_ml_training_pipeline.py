"""
IoT BME680 ML Training Pipeline v3
====================================
Thay đổi so với v2:
  - [FIX] Label strategy: percentile → domain rules (IAQ WHO standard)
  - [FIX] Dùng iaq_score đã tính từ Silver thay vì tính lại từ gas_resistance raw
  - [FIX] Contamination cap từ 0.45 → 0.10 (phù hợp indoor sensor)
  - [ADD] Guard rail: raise nếu anomaly_rate = 0 (threshold quá cao)
  - [ADD] Recall check trong registration decision
  - [ADD] Log sensor ranges để dễ debug/điều chỉnh threshold

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
# Tăng bar so với v2 vì label tốt hơn → model phải tốt hơn mới đáng register
MIN_ROC_AUC   = 0.75
MIN_PRECISION = 0.60
MIN_RECALL    = 0.50   # [NEW] Bỏ sót anomaly nguy hiểm hơn false alarm với IoT

# ── Domain-rule thresholds (thay thế percentile) ─────────────
# Dựa trên IAQ WHO standard + điều kiện môi trường trong nhà hợp lý.
# ⚠️  Kiểm tra log Transform DAG (sensor stats) trước khi chạy lần đầu:
#     nếu iaq_score max < 150 trong 7 ngày → hạ iaq_score_max xuống
#     ví dụ: nếu max chỉ là 80 thì đặt 60-70 là ngưỡng bất thường
DOMAIN_THRESHOLDS = {
    'iaq_score_max':    150.0,   # WHO: >150 Unhealthy (dùng iaq_score từ Silver, không tính lại)
    'temperature_max':   35.0,   # Trong nhà bất thường
    'temperature_min':   10.0,   # Quá lạnh
    'humidity_max':      80.0,   # Nguy cơ mốc
    'humidity_min':      20.0,   # Quá khô
    # gas_resistance KHÔNG đặt ở đây vì iaq_score đã encode gas + humidity rồi
    # → tránh double counting và tránh vấn đề baseline drift giữa các DAG
}

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
    Silver đã clean + feature engineering + iaq_score sẵn từ Transform DAG.

    Label dùng domain rules thay vì percentile để tránh:
      - Anomaly rate bị cố định theo phân phối (luôn ~8-10%)
      - Ranh giới anomaly/normal mờ vì data indoor quá ổn định
      - Circular reasoning: contamination ← anomaly_rate ← percentile
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Silver Data")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    # ── Collect Silver files (7 ngày × 24 giờ) ───────────────────
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

    # ── Load & concat ─────────────────────────────────────────────
    all_data = []
    for file_key in all_files:
        obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
        all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Tổng records từ Silver: {len(df_combined):,}")
    logger.info(f"📊 Columns: {df_combined.columns.tolist()}")

    # ── Kiểm tra feature columns đủ không ────────────────────────
    missing = set(FEATURE_COLS) - set(df_combined.columns)
    if missing:
        raise ValueError(f"❌ Silver data thiếu feature columns: {missing}")

    # ── Sort theo thời gian ───────────────────────────────────────
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    # ── Log sensor ranges để debug/điều chỉnh threshold ──────────
    # Đây là thông tin quan trọng: nếu max của iaq_score < DOMAIN_THRESHOLDS['iaq_score_max']
    # thì anomaly_rate sẽ = 0 và pipeline sẽ raise lỗi bên dưới
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

    # ── Label dựa trên domain knowledge (thay percentile) ────────
    # Dùng iaq_score đã tính sẵn từ Silver (có gas + humidity encoded)
    # KHÔNG tính lại từ gas_resistance để tránh baseline drift giữa các DAG
    df_combined['is_anomaly'] = (
        (df_combined['iaq_score']   > DOMAIN_THRESHOLDS['iaq_score_max']) |
        (df_combined['temperature'] > DOMAIN_THRESHOLDS['temperature_max']) |
        (df_combined['temperature'] < DOMAIN_THRESHOLDS['temperature_min']) |
        (df_combined['humidity']    > DOMAIN_THRESHOLDS['humidity_max'])    |
        (df_combined['humidity']    < DOMAIN_THRESHOLDS['humidity_min'])
    ).astype(int)

    anomaly_rate = float(df_combined['is_anomaly'].mean())
    logger.info(f"\n📊 Anomaly rate: {anomaly_rate:.2%} ({df_combined['is_anomaly'].sum()} / {len(df_combined)} records)")

    # ── Guard rails ───────────────────────────────────────────────
    if anomaly_rate == 0:
        raise ValueError(
            "❌ Anomaly rate = 0% — DOMAIN_THRESHOLDS quá cao so với data thực tế.\n"
            "   Xem sensor ranges ở log trên và hạ thresholds phù hợp.\n"
            f"   iaq_score max thực tế = {df_combined['iaq_score'].max():.1f} "
            f"(threshold hiện tại = {DOMAIN_THRESHOLDS['iaq_score_max']})\n"
            f"   temperature range = [{df_combined['temperature'].min():.1f}, "
            f"{df_combined['temperature'].max():.1f}]\n"
            f"   humidity range = [{df_combined['humidity'].min():.1f}, "
            f"{df_combined['humidity'].max():.1f}]"
        )

    if anomaly_rate > 0.20:
        logger.warning(
            f"⚠️ Anomaly rate {anomaly_rate:.2%} > 20% — "
            "xem xét tăng DOMAIN_THRESHOLDS hoặc kiểm tra sensor bị lỗi"
        )

    if anomaly_rate < 0.01:
        logger.warning(
            f"⚠️ Anomaly rate {anomaly_rate:.2%} < 1% — "
            "môi trường rất ổn định, test set có thể không có đủ anomaly để tính metrics"
        )

    # ── Log breakdown anomaly theo từng condition ─────────────────
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

    # ── Convert timestamp sang string trước khi push XCom ─────────
    df_combined['timestamp'] = df_combined['timestamp'].astype(str)

    context['ti'].xcom_push(
        key='silver_data',
        value=df_combined[FEATURE_COLS + ['is_anomaly', 'timestamp']].to_dict('records')
    )
    context['ti'].xcom_push(key='label_thresholds', value=thresholds)

    return {'record_count': len(df_combined), 'anomaly_rate': anomaly_rate}


def generate_synthetic_anomalies(df: pd.DataFrame, anomaly_ratio: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Tạo synthetic anomalies và recalculate derived features để đảm bảo consistency.
    Chỉ gọi hàm này trên TRAIN SET — không đụng test set.
    """
    np.random.seed(random_state)
    df_synth   = df.copy()
    n_anomalies = int(len(df_synth) * anomaly_ratio)

    anomaly_indices = np.random.choice(df_synth.index, size=n_anomalies, replace=False)
    logger.info(f"🧪 Generating {n_anomalies} synthetic anomalies ({anomaly_ratio:.0%} of train set)...")

    scenario_counts = {'smoke_spike': 0, 'extreme_heat': 0, 'sensor_drift': 0}

    for i in anomaly_indices:
        # smoke/heat xảy ra thực tế nhiều hơn sensor_drift
        scenario = np.random.choice(
            ['smoke_spike', 'extreme_heat', 'sensor_drift'],
            p=[0.50, 0.35, 0.15]
        )
        scenario_counts[scenario] += 1

        if scenario == 'smoke_spike':
            # Khói/nấu ăn: IAQ vọt, gas_resistance giảm mạnh
            df_synth.loc[i, 'iaq_score']     = np.random.uniform(250, 450)
            df_synth.loc[i, 'gas_resistance'] = df_synth.loc[i, 'gas_resistance'] * 0.1

        elif scenario == 'extreme_heat':
            # Thiết bị quá nhiệt / gần nguồn nhiệt
            df_synth.loc[i, 'temperature'] = np.random.uniform(40, 55)
            df_synth.loc[i, 'humidity']    = np.random.uniform(10, 20)

        elif scenario == 'sensor_drift':
            # Lỗi cảm biến: giá trị bất hợp lý
            # Dùng 490 thay vì 500 để tránh vượt IAQ_MAX_SCORE
            df_synth.loc[i, 'iaq_score'] = np.random.choice([0.1, 490])

        df_synth.loc[i, 'is_anomaly'] = 1

    # ── Recalculate derived features sau khi inject ───────────────
    # Bắt buộc: nếu bỏ bước này model sẽ học features mâu thuẫn nhau
    # (vd: iaq_score=350 nhưng iaq_score_rolling_mean=45)

    # Ratio features
    df_synth['temp_humidity_ratio'] = df_synth['temperature'] / (df_synth['humidity'] + 1e-6)
    df_synth['gas_pressure_ratio']  = df_synth['gas_resistance'] / (df_synth['pressure'] + 1e-6)

    # Rolling features — window=10 khớp với Transform DAG
    for col in ['temperature', 'humidity', 'iaq_score']:
        df_synth[f'{col}_rolling_mean'] = (
            df_synth[col].rolling(window=10, min_periods=1).mean()
        )
        df_synth[f'{col}_rolling_std'] = (
            df_synth[col].rolling(window=10, min_periods=1).std().fillna(0)
        )

    logger.info(f"   Scenarios injected : {scenario_counts}")
    logger.info(f"   Anomalies in train : {int(df_synth['is_anomaly'].sum())} "
                f"({df_synth['is_anomaly'].mean():.2%})")

    return df_synth


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

    # ── Split theo thời gian TRƯỚC khi augment ────────────────────
    # Quan trọng: split trước, augment sau → test set luôn là real data
    split_idx       = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
    y_train, y_test = y[:split_idx],        y[split_idx:]

    logger.info(f"📊 Train: {X_train.shape} | anomaly: {y_train.sum()} ({y_train.mean():.2%})")
    logger.info(f"📊 Test : {X_test.shape}  | anomaly: {y_test.sum()} ({y_test.mean():.2%})")

    if y_test.sum() < 5:
        logger.warning(
            f"⚠️ Test set chỉ có {y_test.sum()} anomalies — "
            "metrics có thể không ổn định"
        )

    # ── Synthetic augmentation — chỉ trên train set ──────────────
    SYNTHETIC_RATIO = 0.05  # 5%: đủ để model học boundary, không overfit synthetic

    df_train            = X_train.copy()
    df_train['is_anomaly'] = y_train

    df_train_augmented  = generate_synthetic_anomalies(
        df_train,
        anomaly_ratio=SYNTHETIC_RATIO,
        random_state=RANDOM_STATE
    )

    X_train_aug = df_train_augmented[FEATURE_COLS]
    y_train_aug = np.array(df_train_augmented['is_anomaly'])

    # Contamination tính từ augmented rate (real + synthetic anomalies)
    # Cap 0.15 — cao hơn v3 một chút vì có synthetic, nhưng vẫn hợp lý indoor
    augmented_anomaly_rate = float(y_train_aug.mean())
    contamination          = float(np.clip(augmented_anomaly_rate * 1.2, 0.01, 0.15))

    logger.info(f"📊 Anomaly rate thực  : {anomaly_rate:.2%}")
    logger.info(f"📊 Anomaly rate augmented: {augmented_anomaly_rate:.2%} → contamination={contamination:.4f}")

    with mlflow.start_run(run_name=f"anomaly_detection_{context['ds_nodash']}") as run:
        run_id = run.info.run_id
        logger.info(f"🔬 MLflow Run ID: {run_id}")

        # ── Log params ────────────────────────────────────────────
        mlflow.log_params({
            'model_type':               'IsolationForest',
            'contamination':            contamination,
            'n_estimators':             N_ESTIMATORS,
            'random_state':             RANDOM_STATE,
            'train_size_original':      len(X_train),
            'train_size_augmented':     len(X_train_aug),
            'test_size':                len(X_test),
            'n_features':               len(FEATURE_COLS),
            'anomaly_rate_real':        anomaly_rate,
            'anomaly_rate_augmented':   augmented_anomaly_rate,
            'synthetic_anomaly_ratio':  SYNTHETIC_RATIO,
            'augmentation':             'synthetic_anomalies_train_only',
            'split_method':             'time_based_80_20',
            'label_strategy':           'domain_rules_iaq',
            'training_date':            context['ds'],
            'data_source':              f's3://{S3_BUCKET}/{S3_SILVER_PREFIX}',
            'pipeline_version':         'v3_augmented',
        })
        mlflow.log_params({f"threshold_{k}": v for k, v in thresholds.items()})

        # ── Scale ─────────────────────────────────────────────────
        # fit_transform trên augmented train, chỉ transform trên test
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
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
            logger.warning(
                f"⚠️ Test set chỉ có 1 class — ROC-AUC = 0. "
                "Kiểm tra lại DOMAIN_THRESHOLDS."
            )

        test_metrics = {
            'test_roc_auc':   roc_auc,
            'test_accuracy':  accuracy_score(y_test,        y_test_pred),
            'test_precision': precision_score(y_test,       y_test_pred,  zero_division=0),
            'test_recall':    recall_score(y_test,          y_test_pred,  zero_division=0),
            'test_f1':        f1_score(y_test,              y_test_pred,  zero_division=0),
        }
        # Train metrics tính trên augmented data (bao gồm synthetic)
        train_metrics = {
            'train_accuracy':  accuracy_score(y_train_aug,  y_train_pred),
            'train_precision': precision_score(y_train_aug, y_train_pred, zero_division=0),
            'train_recall':    recall_score(y_train_aug,    y_train_pred, zero_division=0),
            'train_f1':        f1_score(y_train_aug,        y_train_pred, zero_division=0),
        }
        mlflow.log_metrics({**train_metrics, **test_metrics})

        logger.info("\n=== Model Performance ===")
        logger.info(f"  {'Metric':<20} {'Train (aug)':>12} {'Test (real)':>12}  {'Threshold':>10}")
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
            f'Confusion Matrix (Test — Real Data Only)\n'
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

        # ── Log artifacts ─────────────────────────────────────────
        with open('/tmp/label_thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)
        mlflow.log_artifact('/tmp/label_thresholds.json')

        scaler_path = '/tmp/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path='preprocessor')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            signature=mlflow.models.infer_signature(X_train_scaled, y_train_pred)
        )
        mlflow.set_tags({
            'team':              'iot-ml',
            'project':           'bme680-anomaly-detection',
            'environment':       'production',
            'label_strategy':    'domain_rules_iaq',
            'augmentation':      'synthetic_anomalies',
            'data_layer':        'silver',
            'pipeline_version':  'v3_augmented',
        })

        context['ti'].xcom_push(key='model_metrics', value=test_metrics)
        context['ti'].xcom_push(key='run_id',        value=run_id)

        return {'run_id': run_id, 'test_roc_auc': roc_auc}

def decide_model_registration(**context):
    """
    Kiểm tra 3 điều kiện: ROC-AUC, Precision, Recall.
    Thêm Recall để tránh tình huống model precision cao nhưng bỏ sót hầu hết anomaly.
    """
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

    # Cảnh báo đặc biệt khi cả 3 đều = 0 → khả năng cao test set không có anomaly
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
    logger.warning(f"⚠️ Model chưa đủ điều kiện — failed: {', '.join(failed)} — bỏ qua registration")
    return 'skip_registration'


def register_model(**context):
    """
    Register model vào MLflow Model Registry → Staging.
    Production promotion vẫn cần manual review.
    """
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
            f"Label: domain_rules_iaq | "
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

    # Kiểm tra model đang Production để so sánh
    try:
        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if prod_versions:
            prod_v = prod_versions[0]
            logger.info(f"ℹ️ Production hiện tại: version {prod_v.version}")
            logger.info(f"   New Staging version {version} cần manual promotion sau khi validate")
        else:
            logger.info("ℹ️ Chưa có Production model — promote Staging lên Production thủ công sau khi validate")
    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")

    context['ti'].xcom_push(key='model_version', value=version)
    return {'model_name': REGISTERED_MODEL_NAME, 'version': version, 'stage': 'Staging'}


def send_notification(**context):
    """
    Log summary kết quả pipeline.
    TODO: Tích hợp Slack / SNS / PagerDuty nếu cần alert tự động.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Sending Notification")
    logger.info("=" * 60)

    run_id        = context['ti'].xcom_pull(task_ids='train_model',    key='run_id')
    metrics       = context['ti'].xcom_pull(task_ids='train_model',    key='model_metrics')
    thresholds    = context['ti'].xcom_pull(task_ids='load_silver_data', key='label_thresholds')
    model_version = context['ti'].xcom_pull(task_ids='register_model', key='model_version')

    registered = model_version is not None
    status_icon = "🎉" if registered else "⚠️"

    message = f"""
{status_icon} IoT ML Training Pipeline v3 — {'REGISTERED' if registered else 'NOT REGISTERED'}

📅 Training Date : {context['ds']}
📦 Data Source   : Silver layer (domain-rule labels)
🔬 MLflow Run ID : {run_id}

📊 Model Performance:
   ROC-AUC   : {metrics['test_roc_auc']:.4f}  (threshold: {MIN_ROC_AUC})
   Precision : {metrics['test_precision']:.4f}  (threshold: {MIN_PRECISION})
   Recall    : {metrics['test_recall']:.4f}  (threshold: {MIN_RECALL})
   F1 Score  : {metrics['test_f1']:.4f}
   Accuracy  : {metrics['test_accuracy']:.4f}

🏷️  Label Strategy : domain_rules_iaq (v3)
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
    description='Silver → Train (domain-rule labels) → MLflow Registry | v3',
    schedule='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'iot', 'anomaly-detection', 'mlflow', 'production', 'v3']
) as dag:

    t_load   = PythonOperator(task_id='load_silver_data',      python_callable=load_silver_data)
    t_train  = PythonOperator(task_id='train_model',           python_callable=train_anomaly_model)
    t_decide = BranchPythonOperator(task_id='decide_registration', python_callable=decide_model_registration)
    t_reg    = PythonOperator(task_id='register_model',        python_callable=register_model)
    t_skip   = EmptyOperator(task_id='skip_registration')
    t_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='none_failed'
    )

    t_load >> t_train >> t_decide >> [t_reg, t_skip] >> t_notify
