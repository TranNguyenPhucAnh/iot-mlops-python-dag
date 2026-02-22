"""
IoT BME680 Transform DAG (Silver Layer)
========================================
Đọc data từ Bronze layer, clean + feature engineering,
ghi ra Silver layer để Training và Inference dùng chung.

Schedule: Every 30 minutes
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from io import BytesIO
from airflow import Dataset

logger = logging.getLogger(__name__)

S3_BUCKET        = "iot-bme680-data-lake-prod"
S3_BRONZE_PREFIX = "bronze/bme680/"
S3_SILVER_PREFIX = "silver/bme680_features/"
SILVER_DATASET = Dataset(f"s3://{S3_BUCKET}/{S3_SILVER_PREFIX}")

# Validation ranges
VALIDATION_RULES = {
    'temperature':    (0,   85),
    'humidity':       (0,   100),
    'pressure':       (800, 1200),
    'gas_resistance': (100, 500000),
}

# IAQ config
IAQ_HUMIDITY_OPTIMAL_LOW  = 30
IAQ_HUMIDITY_OPTIMAL_HIGH = 60
IAQ_HUMIDITY_SCORE_FACTOR = 5
IAQ_MAX_SCORE             = 500


# ==================== Helpers ====================

def calc_iaq_score(gas_resistance: float, humidity: float, gas_baseline: float) -> float:
    """Tính IAQ score với dynamic baseline — dùng chung với ingestion DAG."""
    if gas_resistance >= gas_baseline:
        gas_score = 0.0
    else:
        gas_score = (1.0 - gas_resistance / gas_baseline) * 300.0

    if IAQ_HUMIDITY_OPTIMAL_LOW <= humidity <= IAQ_HUMIDITY_OPTIMAL_HIGH:
        hum_score = 0.0
    elif humidity < IAQ_HUMIDITY_OPTIMAL_LOW:
        hum_score = (IAQ_HUMIDITY_OPTIMAL_LOW - humidity) * IAQ_HUMIDITY_SCORE_FACTOR
    else:
        hum_score = (humidity - IAQ_HUMIDITY_OPTIMAL_HIGH) * IAQ_HUMIDITY_SCORE_FACTOR

    return round(min(IAQ_MAX_SCORE, gas_score + hum_score), 2)


def list_bronze_files(s3_hook, execution_date) -> list[str]:
    """
    List tất cả Parquet files trong Bronze layer của 2 giờ gần nhất.
    Transform chạy mỗi 30 phút → lấy 2 giờ để đảm bảo không miss data.
    """
    files = []
    now = execution_date

    for hours_back in range(2):
        check_dt = now - timedelta(hours=hours_back)
        prefix   = (
            f"{S3_BRONZE_PREFIX}"
            f"year={check_dt.year}/"
            f"month={check_dt.month:02d}/"
            f"day={check_dt.day:02d}/"
            f"hour={check_dt.hour:02d}/"
        )
        keys = s3_hook.list_keys(bucket_name=S3_BUCKET, prefix=prefix)
        if keys:
            parquet_files = [k for k in keys if k.endswith('.parquet')]
            files.extend(parquet_files)
            logger.info(f"  ✅ {prefix}: {len(parquet_files)} files")

    return files


# ==================== Tasks ====================

def extract_bronze(**context):
    """Đọc Bronze files từ 2 giờ gần nhất"""
    logger.info("=" * 60)
    logger.info("STEP 1: Extract từ Bronze layer")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    bronze_files = list_bronze_files(s3_hook, execution_date)

    if not bronze_files:
        logger.warning("⚠️ Không có Bronze files mới — skip")
        raise AirflowSkipException("No new Bronze data")

    logger.info(f"📁 Tổng Bronze files: {len(bronze_files)}")

    all_data = []
    for file_key in bronze_files:
        obj   = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
        df    = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
        all_data.append(df)
        logger.info(f"  📥 {file_key}: {len(df)} records")

    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Tổng records từ Bronze: {len(df_combined):,}")

    context['ti'].xcom_push(key='bronze_data', value=df_combined.to_dict('records'))
    context['ti'].xcom_push(key='bronze_file_count', value=len(bronze_files))

    return len(df_combined)


def validate_and_clean(**context):
    """Validate ranges, drop nulls, remove outliers"""
    logger.info("=" * 60)
    logger.info("STEP 2: Validate & Clean")
    logger.info("=" * 60)

    bronze_data = context['ti'].xcom_pull(task_ids='extract_bronze', key='bronze_data')
    df          = pd.DataFrame(bronze_data)

    total_in = len(df)
    logger.info(f"📊 Input records: {total_in:,}")
    logger.info(f"📊 Columns: {df.columns.tolist()}")

    # ── Null check ────────────────────────────────────────────────
    required_cols = ['temperature', 'humidity', 'pressure', 'gas_resistance', 'timestamp']
    null_counts   = df[required_cols].isnull().sum()

    if null_counts.sum() > 0:
        logger.warning(f"⚠️ Null values:\n{null_counts[null_counts > 0]}")
        df = df.dropna(subset=required_cols)
        logger.info(f"  Dropped nulls → {len(df):,} records remaining")
    else:
        logger.info("✅ No null values")

    # ── Range validation ──────────────────────────────────────────
    for col, (lo, hi) in VALIDATION_RULES.items():
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            logger.warning(f"⚠️ {col}: {out_of_range} values out of [{lo}, {hi}]")

    mask_clean = (
        df['temperature'].between(*VALIDATION_RULES['temperature']) &
        df['humidity'].between(*VALIDATION_RULES['humidity']) &
        df['pressure'].between(*VALIDATION_RULES['pressure']) &
        df['gas_resistance'].between(*VALIDATION_RULES['gas_resistance'])
    )
    df_clean = df[mask_clean].copy()

    dropped = total_in - len(df_clean)
    quality_score = len(df_clean) / total_in if total_in > 0 else 0

    logger.info(f"✅ Clean records: {len(df_clean):,} (dropped {dropped}, quality={quality_score:.2%})")

    if quality_score < 0.8:
        logger.warning(f"⚠️ Data quality thấp: {quality_score:.2%} — kiểm tra lại sensor")

    # ── Deduplicate theo timestamp + device_id ────────────────────
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['timestamp', 'device_id'], keep='last')
    logger.info(f"✅ Deduplicated: {before_dedup - len(df_clean)} duplicates removed")

    context['ti'].xcom_push(key='clean_data', value=df_clean.to_dict('records'))
    context['ti'].xcom_push(key='quality_metrics', value={
        'total_in':     total_in,
        'clean_out':    len(df_clean),
        'dropped':      dropped,
        'quality_score': quality_score
    })

    return len(df_clean)


def feature_engineering(**context):
    """
    Feature engineering tập trung — single source of truth.
    Training DAG và Inference DAG đều đọc từ Silver này,
    không cần duplicate code feature engineering nữa.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)

    clean_data = context['ti'].xcom_pull(task_ids='validate_and_clean', key='clean_data')
    df         = pd.DataFrame(clean_data)

    logger.info(f"📊 Engineering features for {len(df):,} records")

    # ── Parse & sort timestamp ────────────────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── Recalculate IAQ với dynamic baseline ─────────────────────
    # Dùng q75 của batch này làm baseline — nhất quán với ingestion DAG
    gas_baseline = float(df['gas_resistance'].quantile(0.75))
    logger.info(f"📐 Gas baseline (q75): {gas_baseline:.0f}")

    df['iaq_score'] = df.apply(
        lambda r: calc_iaq_score(r['gas_resistance'], r['humidity'], gas_baseline),
        axis=1
    )
    df['gas_baseline'] = gas_baseline  # Lưu lại để trace

    # ── Time features ─────────────────────────────────────────────
    df['hour']       = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ── Ratio features ────────────────────────────────────────────
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
    df['gas_pressure_ratio']  = df['gas_resistance'] / (df['pressure'] + 1e-6)

    # ── Rolling features ──────────────────────────────────────────
    for col in ['temperature', 'humidity', 'iaq_score']:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'{col}_rolling_std']  = df[col].rolling(window=10, min_periods=1).std().fillna(0)

    # ── Sensor statistics để log & debug ─────────────────────────
    logger.info("\n=== Sensor Statistics ===")
    for col in ['temperature', 'humidity', 'pressure', 'gas_resistance', 'iaq_score']:
        logger.info(
            f"  {col}: "
            f"min={df[col].min():.1f}, "
            f"mean={df[col].mean():.1f}, "
            f"max={df[col].max():.1f}, "
            f"p2={df[col].quantile(0.02):.1f}, "
            f"p98={df[col].quantile(0.98):.1f}"
        )

    logger.info(f"✅ Feature engineering done — {df.shape[1]} columns, {len(df):,} records")
    
    # Trước khi push XCom, convert timestamp sang string
    df['timestamp'] = df['timestamp'].astype(str)

    context['ti'].xcom_push(key='feature_data', value=df.to_dict('records'))
    context['ti'].xcom_push(key='gas_baseline', value=gas_baseline)

    return {'record_count': len(df), 'feature_count': df.shape[1]}

def write_silver(**context):
    """Ghi Silver Parquet — partition theo year/month/day/hour"""
    logger.info("=" * 60)
    logger.info("STEP 4: Write Silver layer")
    logger.info("=" * 60)

    feature_data   = context['ti'].xcom_pull(task_ids='feature_engineering', key='feature_data')
    quality_metrics = context['ti'].xcom_pull(task_ids='validate_and_clean', key='quality_metrics')

    df  = pd.DataFrame(feature_data)
    now = datetime.utcnow()

    silver_path = (
        f"{S3_SILVER_PREFIX}"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"hour={now.hour:02d}/"
        f"features_{now.strftime('%H%M%S')}.parquet"
    )

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
    buffer.seek(0)

    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=silver_path,
        bucket_name=S3_BUCKET,
        replace=True
    )

    logger.info(f"✅ Saved {len(df):,} records to s3://{S3_BUCKET}/{silver_path}")
    logger.info(f"📊 Quality metrics: {quality_metrics}")

    return {
        'silver_path':  silver_path,
        'record_count': len(df),
        'quality':      quality_metrics
    }


# ==================== DAG ====================

default_args = {
    'owner':          'iot-ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries':        2,
    'retry_delay':    timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=20)
}

with DAG(
    dag_id='iot_bme680_transform_pipeline',
    description='Bronze → Silver: clean, validate, feature engineering',
    schedule='*/30 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['iot', 'silver', 'transform', 'feature-engineering']
) as dag:

    t_extract  = PythonOperator(task_id='extract_bronze',       python_callable=extract_bronze)
    t_validate = PythonOperator(task_id='validate_and_clean',   python_callable=validate_and_clean)
    t_features = PythonOperator(task_id='feature_engineering',  python_callable=feature_engineering)
    t_write    = PythonOperator(task_id='write_silver',         python_callable=write_silver, outlets=[SILVER_DATASET]  # ← báo cho Airflow biết đã ghi S3 Data Lake Silver)

    t_extract >> t_validate >> t_features >> t_write
