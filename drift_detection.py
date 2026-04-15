"""
IoT BME680 Drift Detection DAG
================================
Phát hiện data drift giữa training distribution (Silver) và
production distribution (Gold inference results).

Metrics:
  - PSI  (Population Stability Index)   — feature distribution shift
  - KL   (KL Divergence)                — information-theoretic distance
  - KS   (Kolmogorov-Smirnov test)      — statistical test với p-value
  - Mean/Std shift                      — simple sanity check

Kết quả:
  - Ghi drift report JSON vào S3 drift/
  - Nếu drift = True  → TriggerDagRunOperator kick off training pipeline
  - Luôn gửi notification (Slack log) tóm tắt kết quả

Schedule: Daily at 1 AM UTC (1 giờ trước training DAG 2 AM)
Trigger: Cũng có thể trigger thủ công hoặc từ monitoring alert
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import logging
from io import BytesIO
import joblib

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
MLFLOW_TRACKING_URI   = "http://mlflow.mlflow.svc.cluster.local:80"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"
S3_BUCKET             = "iot-bme680-data-lake-prod"
S3_SILVER_PREFIX      = "silver/bme680_features/"
S3_GOLD_PREFIX        = "gold/bme680_predictions/"
S3_DRIFT_PREFIX       = "drift/reports/"

# Features dùng cho drift detection — đúng với FEATURE_COLS của training
FEATURE_COLS = [
    'temperature', 'humidity', 'pressure', 'gas_resistance'
]

# Window production data để so sánh (7 ngày gần nhất)
PRODUCTION_WINDOW_DAYS = 7

# ── Drift thresholds ─────────────────────────────────────────────────────────
# PSI:  < 0.10 = no drift | 0.10–0.20 = moderate | > 0.20 = significant
# KL:   > 0.10 = đáng chú ý | > 0.20 = significant
# KS:   p_value < 0.05 = distribution khác biệt có ý nghĩa thống kê
PSI_THRESHOLD      = 0.20   # trigger retrain nếu PSI của bất kỳ feature nào vượt
KL_THRESHOLD       = 0.15   # trigger retrain nếu KL mean của tất cả features vượt
KS_PVALUE          = 0.05   # trigger retrain nếu >= N_KS_VIOLATIONS features có p < 0.05
N_KS_VIOLATIONS    = 2      # số features vi phạm KS threshold để trigger

# Cần ít nhất bao nhiêu records production để drift detection có ý nghĩa
MIN_PRODUCTION_RECORDS = 500

# Số bins để tính PSI / KL histogram
N_BINS = 20


# ==================== Helper Functions ====================

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Population Stability Index (PSI).
    expected = training distribution (reference)
    actual   = production distribution (current)

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)
    Thêm epsilon để tránh log(0) và division by zero.
    """
    # Tạo bins từ expected distribution (training)
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if max_val == min_val:
        return 0.0  # không có variance → không có drift

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= 1e-6   # đảm bảo min value rơi vào bin đầu
    bins[-1] += 1e-6   # đảm bảo max value rơi vào bin cuối

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _   = np.histogram(actual,   bins=bins)

    # Convert sang percentage, add epsilon để tránh log(0)
    eps = 1e-6
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct   = (actual_counts   / len(actual))   + eps

    # Renormalize sau khi thêm epsilon
    expected_pct /= expected_pct.sum()
    actual_pct   /= actual_pct.sum()

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def compute_kl_divergence(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    KL Divergence: KL(actual || expected) — asymmetric.
    Đo lượng thông tin mất đi khi dùng expected để xấp xỉ actual.
    """
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if max_val == min_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= 1e-6
    bins[-1] += 1e-6

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _   = np.histogram(actual,   bins=bins)

    eps = 1e-8
    p = (actual_counts   / len(actual))   + eps
    q = (expected_counts / len(expected)) + eps
    p /= p.sum()
    q /= q.sum()

    # KL(P || Q) = Σ p * log(p / q)
    kl = float(np.sum(p * np.log(p / q)))
    return max(kl, 0.0)   # numerical stability


def compute_ks_test(expected: np.ndarray, actual: np.ndarray):
    """
    Kolmogorov-Smirnov two-sample test.
    Returns (statistic, p_value).
    p_value < 0.05 → reject H0 rằng 2 samples từ cùng distribution.
    """
    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return float(ks_stat), float(p_value)


def describe_distribution(arr: np.ndarray) -> dict:
    """Thống kê cơ bản của một distribution."""
    return {
        'count':  int(len(arr)),
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
        'min':    float(np.min(arr)),
        'p05':    float(np.percentile(arr, 5)),
        'p25':    float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75':    float(np.percentile(arr, 75)),
        'p95':    float(np.percentile(arr, 95)),
        'max':    float(np.max(arr)),
    }


# ==================== Tasks ====================

def load_training_reference(**context):
    """
    Load training distribution từ Silver data của thời điểm train gần nhất.
    Đây là reference distribution để so sánh drift.

    Strategy: Lấy Silver data của 30 ngày gần nhất làm reference.
    Lý do dùng Silver (không phải MLflow artifacts) vì Silver chứa
    đúng các feature columns mà model được train trên đó.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Training Reference Distribution")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    all_files = []
    # Lấy 30 ngày Silver làm reference (rộng hơn production window để ổn định)
    for days_back in range(1, 31):
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
                all_files.extend(parquet_files)

    if not all_files:
        raise ValueError("❌ Không tìm thấy Silver reference data trong 30 ngày gần nhất")

    logger.info(f"📁 Reference Silver files: {len(all_files)}")

    all_data = []
    for file_key in all_files:
        try:
            obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
            df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
            all_data.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Không đọc được {file_key}: {e}")

    df_ref = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Reference records: {len(df_ref):,}")

    # Chỉ giữ FEATURE_COLS có trong data
    available_features = [f for f in FEATURE_COLS if f in df_ref.columns]
    if not available_features:
        raise ValueError(f"❌ Không tìm thấy feature columns {FEATURE_COLS} trong Silver data")

    # Log distribution của reference
    logger.info("\n📋 Reference distribution summary:")
    for col in available_features:
        clean = df_ref[col].dropna()
        logger.info(
            f"  {col:25s}: mean={clean.mean():.3f}, std={clean.std():.3f}, "
            f"min={clean.min():.3f}, max={clean.max():.3f}"
        )

    # Serialize sang dict để XCom (chỉ lấy FEATURE_COLS, bỏ NaN)
    df_clean = df_ref[available_features].dropna()
    context['ti'].xcom_push(key='reference_data',     value=df_clean.to_dict('records'))
    context['ti'].xcom_push(key='available_features', value=available_features)
    context['ti'].xcom_push(key='reference_count',    value=len(df_clean))

    return {'reference_count': len(df_clean), 'features': available_features}


def load_production_distribution(**context):
    """
    Load production distribution từ Gold layer (inference results).
    Gold chứa predictions + raw feature values từ PRODUCTION_WINDOW_DAYS gần nhất.

    Mục đích: so sánh feature distribution của production với training reference.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Load Production Distribution (Gold)")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    all_files = []
    for days_back in range(PRODUCTION_WINDOW_DAYS):
        check_date = execution_date - timedelta(days=days_back)
        for hour in range(24):
            prefix = (
                f"{S3_GOLD_PREFIX}"
                f"year={check_date.year}/"
                f"month={check_date.month:02d}/"
                f"day={check_date.day:02d}/"
                f"hour={hour:02d}/"
            )
            keys = s3_hook.list_keys(bucket_name=S3_BUCKET, prefix=prefix)
            if keys:
                parquet_files = [k for k in keys if k.endswith('.parquet')]
                all_files.extend(parquet_files)

    if not all_files:
        raise ValueError(
            f"❌ Không tìm thấy Gold data trong {PRODUCTION_WINDOW_DAYS} ngày gần nhất"
        )

    logger.info(f"📁 Production Gold files: {len(all_files)}")

    all_data = []
    for file_key in all_files:
        try:
            obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
            df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
            all_data.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Không đọc được {file_key}: {e}")

    df_prod = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Production records: {len(df_prod):,}")

    if len(df_prod) < MIN_PRODUCTION_RECORDS:
        raise ValueError(
            f"❌ Chỉ có {len(df_prod)} production records — "
            f"cần tối thiểu {MIN_PRODUCTION_RECORDS} để drift detection có ý nghĩa"
        )

    # Chỉ lấy FEATURE_COLS (Gold chứa cả predictions, chỉ cần features)
    available_features = context['ti'].xcom_pull(
        task_ids='load_reference', key='available_features'
    )
    prod_features = [f for f in available_features if f in df_prod.columns]

    if not prod_features:
        raise ValueError(f"❌ Gold data không chứa feature columns: {available_features}")

    if len(prod_features) < len(available_features):
        missing = set(available_features) - set(prod_features)
        logger.warning(f"⚠️ Gold thiếu features so với training: {missing}")

    # Log production distribution
    logger.info("\n📋 Production distribution summary:")
    for col in prod_features:
        clean = df_prod[col].dropna()
        logger.info(
            f"  {col:25s}: mean={clean.mean():.3f}, std={clean.std():.3f}, "
            f"min={clean.min():.3f}, max={clean.max():.3f}"
        )

    df_clean = df_prod[prod_features].dropna()
    context['ti'].xcom_push(key='production_data',  value=df_clean.to_dict('records'))
    context['ti'].xcom_push(key='prod_features',    value=prod_features)
    context['ti'].xcom_push(key='production_count', value=len(df_clean))

    return {'production_count': len(df_clean), 'features': prod_features}


def compute_drift_metrics(**context):
    """
    Tính PSI, KL divergence, KS test cho từng feature.
    Quyết định drift = True/False dựa trên thresholds đã định nghĩa.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Compute Drift Metrics")
    logger.info("=" * 60)

    ref_data  = context['ti'].xcom_pull(task_ids='load_reference',   key='reference_data')
    prod_data = context['ti'].xcom_pull(task_ids='load_production',   key='production_data')
    features  = context['ti'].xcom_pull(task_ids='load_production',   key='prod_features')

    df_ref  = pd.DataFrame(ref_data)
    df_prod = pd.DataFrame(prod_data)

    feature_results = {}
    drift_flags     = {}

    logger.info(
        f"\n{'Feature':<25} {'PSI':>8} {'KL':>8} {'KS stat':>10} "
        f"{'KS p-val':>10} {'PSI flag':>10} {'KS flag':>10}"
    )
    logger.info("-" * 85)

    for feature in features:
        ref_arr  = df_ref[feature].dropna().values.astype(float)
        prod_arr = df_prod[feature].dropna().values.astype(float)

        if len(ref_arr) < 10 or len(prod_arr) < 10:
            logger.warning(f"⚠️ {feature}: không đủ data để tính drift")
            continue

        psi        = compute_psi(ref_arr, prod_arr)
        kl         = compute_kl_divergence(ref_arr, prod_arr)
        ks_stat, ks_pval = compute_ks_test(ref_arr, prod_arr)

        psi_flag = psi > PSI_THRESHOLD
        ks_flag  = ks_pval < KS_PVALUE

        feature_results[feature] = {
            'psi':          round(psi, 6),
            'kl_divergence': round(kl, 6),
            'ks_statistic': round(ks_stat, 6),
            'ks_pvalue':    round(ks_pval, 6),
            'psi_flag':     psi_flag,
            'ks_flag':      ks_flag,
            'reference':    describe_distribution(ref_arr),
            'production':   describe_distribution(prod_arr),
            'mean_shift':   round(float(prod_arr.mean() - ref_arr.mean()), 4),
            'std_shift':    round(float(prod_arr.std()  - ref_arr.std()),  4),
        }
        drift_flags[feature] = {'psi': psi_flag, 'ks': ks_flag}

        logger.info(
            f"{feature:<25} {psi:>8.4f} {kl:>8.4f} {ks_stat:>10.4f} "
            f"{ks_pval:>10.4f} {'🔴' if psi_flag else '🟢':>10} {'🔴' if ks_flag else '🟢':>10}"
        )

    # ── Aggregate drift decision ─────────────────────────────────────────────
    # Drift = True nếu:
    #   (a) Bất kỳ feature nào có PSI > threshold, HOẶC
    #   (b) Số features vi phạm KS >= N_KS_VIOLATIONS
    psi_violations = [f for f, r in feature_results.items() if r['psi_flag']]
    ks_violations  = [f for f, r in feature_results.items() if r['ks_flag']]
    kl_mean        = float(np.mean([r['kl_divergence'] for r in feature_results.values()]))

    drift_by_psi  = len(psi_violations) > 0
    drift_by_ks   = len(ks_violations) >= N_KS_VIOLATIONS
    drift_by_kl   = kl_mean > KL_THRESHOLD
    overall_drift = drift_by_psi or drift_by_ks or drift_by_kl

    logger.info("\n" + "=" * 60)
    logger.info("📊 DRIFT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  PSI violations   : {psi_violations} → drift={drift_by_psi}")
    logger.info(f"  KS  violations   : {ks_violations} ({len(ks_violations)}/{len(features)}) → drift={drift_by_ks}")
    logger.info(f"  KL mean          : {kl_mean:.4f} (threshold={KL_THRESHOLD}) → drift={drift_by_kl}")
    logger.info(f"  🔑 OVERALL DRIFT : {'⚠️  TRUE  — Trigger Retrain' if overall_drift else '✅  FALSE — No Action'}")

    drift_report = {
        'execution_date':    str(context.get('logical_date') or context.get('execution_date')),
        'overall_drift':     overall_drift,
        'drift_by_psi':      drift_by_psi,
        'drift_by_ks':       drift_by_ks,
        'drift_by_kl':       drift_by_kl,
        'psi_violations':    psi_violations,
        'ks_violations':     ks_violations,
        'kl_mean':           round(kl_mean, 6),
        'reference_count':   context['ti'].xcom_pull(task_ids='load_reference',  key='reference_count'),
        'production_count':  context['ti'].xcom_pull(task_ids='load_production', key='production_count'),
        'production_window_days': PRODUCTION_WINDOW_DAYS,
        'thresholds': {
            'psi':          PSI_THRESHOLD,
            'kl':           KL_THRESHOLD,
            'ks_pvalue':    KS_PVALUE,
            'ks_violations': N_KS_VIOLATIONS,
        },
        'features': feature_results,
    }

    context['ti'].xcom_push(key='drift_report',  value=drift_report)
    context['ti'].xcom_push(key='overall_drift', value=overall_drift)

    return drift_report


def save_drift_report(**context):
    """
    Lưu drift report JSON vào S3 drift/reports/ để audit và Grafana query.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Save Drift Report to S3")
    logger.info("=" * 60)

    drift_report = context['ti'].xcom_pull(task_ids='compute_drift', key='drift_report')
    s3_hook      = S3Hook(aws_conn_id='aws_default')
    now          = datetime.utcnow()

    report_key = (
        f"{S3_DRIFT_PREFIX}"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"drift_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
    )

    report_bytes = json.dumps(drift_report, indent=2, default=str).encode('utf-8')
    s3_hook.load_bytes(
        bytes_data=report_bytes,
        key=report_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    logger.info(f"💾 Drift report saved: s3://{S3_BUCKET}/{report_key}")

    # Cũng ghi latest report để Grafana/monitoring dễ query
    latest_key = f"{S3_DRIFT_PREFIX}latest/drift_report_latest.json"
    s3_hook.load_bytes(
        bytes_data=report_bytes,
        key=latest_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    logger.info(f"💾 Latest report updated: s3://{S3_BUCKET}/{latest_key}")

    context['ti'].xcom_push(key='report_s3_key', value=report_key)
    return report_key


def branch_on_drift(**context):
    overall_drift = context['ti'].xcom_pull(task_ids='compute_drift', key='overall_drift')

    if not overall_drift:
        return 'no_drift_action'

    # Kiểm tra có Staging model đang chờ review không
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    staging_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"])

    if staging_versions:
        logger.info(
            f"⚠️ Drift detected nhưng đã có Staging model v{staging_versions[0].version} "
            f"chờ review — skip trigger training để tránh retrain liên tục"
        )
        return 'no_drift_action'   # chờ human review xong rồi tính

    return 'trigger_training'

def send_drift_notification(**context):
    """
    Gửi notification tóm tắt kết quả drift detection.
    Luôn chạy dù drift hay không — human cần biết cả 2 trường hợp.
    TODO: Thay logger bằng Slack/SNS/PagerDuty thực tế.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Send Drift Notification")
    logger.info("=" * 60)

    drift_report = context['ti'].xcom_pull(task_ids='compute_drift', key='drift_report')
    report_key   = context['ti'].xcom_pull(task_ids='save_report',   key='report_s3_key')
    overall_drift = drift_report['overall_drift']

    # Build feature summary
    feature_lines = []
    for feat, result in drift_report.get('features', {}).items():
        psi_icon = '🔴' if result['psi_flag'] else '🟢'
        ks_icon  = '🔴' if result['ks_flag']  else '🟢'
        feature_lines.append(
            f"  {feat:<25} PSI={result['psi']:.4f}{psi_icon}  "
            f"KL={result['kl_divergence']:.4f}  "
            f"KS_p={result['ks_pvalue']:.4f}{ks_icon}  "
            f"mean_shift={result['mean_shift']:+.3f}"
        )

    status_icon = '🚨' if overall_drift else '✅'
    action_line = (
        '⚡ Action: Training DAG đã được trigger tự động'
        if overall_drift else
        '💤 Action: Không cần retrain — model hiện tại vẫn phù hợp với production data'
    )

    message = f"""
{status_icon} Drift Detection Report — {drift_report['execution_date']}

📊 Overall Drift: {'DETECTED — RETRAIN TRIGGERED' if overall_drift else 'NONE DETECTED'}

🔬 Drift Reasons:
   PSI violations  : {drift_report['psi_violations'] or 'none'}
   KS  violations  : {drift_report['ks_violations'] or 'none'} ({len(drift_report['ks_violations'])}/{len(drift_report.get('features', {}))})
   KL mean         : {drift_report['kl_mean']:.4f} (threshold: {drift_report['thresholds']['kl']})

📈 Feature Breakdown:
{chr(10).join(feature_lines)}

📦 Data Windows:
   Reference  : {drift_report['reference_count']:,} records (30-day Silver)
   Production : {drift_report['production_count']:,} records ({drift_report['production_window_days']}-day Gold)

{action_line}

🔗 Full report: s3://{S3_BUCKET}/{report_key}
"""
    logger.info(message)

    # TODO: uncomment khi có Slack webhook
    # import requests
    # requests.post(SLACK_WEBHOOK_URL, json={"text": message})

    return message


# ==================== DAG ====================

default_args = {
    'owner':             'iot-ml-team',
    'depends_on_past':   False,
    'email':             ['iot-ml@company.com'],
    'email_on_failure':  True,
    'email_on_retry':    False,
    'retries':           1,
    'retry_delay':       timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=30),
}

with DAG(
    dag_id='iot_ml_drift_detection',
    description='Drift detection: Silver (reference) vs Gold (production) | PSI + KL + KS',
    schedule='0 1 * * *',    # 1 AM UTC — chạy trước training DAG (2 AM)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'drift', 'monitoring', 'iot', 'anomaly-detection'],
    doc_md="""
## Drift Detection DAG

Chạy mỗi ngày lúc 1 AM UTC, trước training pipeline (2 AM).

**Flow:**
1. Load Silver data 30 ngày làm reference distribution
2. Load Gold (inference results) 7 ngày làm production distribution
3. Tính PSI, KL divergence, KS test cho từng feature
4. Lưu report JSON vào S3 `drift/reports/`
5. Nếu drift → trigger `iot_ml_training_pipeline`
6. Gửi notification

**Drift triggers retrain nếu:**
- PSI > 0.20 của BẤT KỲ feature nào, HOẶC
- Số features vi phạm KS (p < 0.05) >= 2, HOẶC
- KL divergence trung bình > 0.15
"""
) as dag:

    t_ref    = PythonOperator(task_id='load_reference',  python_callable=load_training_reference)
    t_prod   = PythonOperator(task_id='load_production', python_callable=load_production_distribution)
    t_drift  = PythonOperator(task_id='compute_drift',   python_callable=compute_drift_metrics)
    t_save   = PythonOperator(task_id='save_report',     python_callable=save_drift_report)
    t_branch = BranchPythonOperator(task_id='branch_drift', python_callable=branch_on_drift)

    t_trigger = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='iot_ml_training_pipeline',
        conf={'triggered_by': 'drift_detection', 'drift_date': '{{ ds }}'},
        wait_for_completion=False,   # Fire and forget — training DAG tự quản lý
    )

    t_no_drift = EmptyOperator(task_id='no_drift_action')

    t_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_drift_notification,
        trigger_rule='none_failed_min_one_success',   # chạy dù trigger hay no_drift
    )

    # ── DAG flow ──────────────────────────────────────────────────────────────
    # load_reference và load_production chạy song song → compute_drift
    [t_ref, t_prod] >> t_drift >> t_save >> t_branch >> [t_trigger, t_no_drift] >> t_notify
