"""
IoT BME680 Model Comparison DAG
=================================
Tạo báo cáo so sánh Champion vs Challenger để human review và ra
quyết định promote thủ công.

DAG này KHÔNG tự promote model. Output là:
  1. HTML report đầy đủ → upload S3, gửi presigned URL qua notification
  2. JSON summary → lưu S3 để audit trail

Report bao gồm:
  - Performance metrics: ROC-AUC, Precision, Recall, F1, Accuracy
  - Confusion matrix cả 2 models (cùng test set)
  - Anomaly score distribution comparison
  - Feature importance (IsolationForest mean depth)
  - Model metadata: training date, data window, hyperparameters
  - Drift context (tại sao challenger được tạo ra)
  - AI-generated recommendation (tham khảo, không quyết định)

Trigger: Tự động sau khi Training DAG hoàn thành và register model
Schedule: Không có schedule — chỉ trigger từ training pipeline
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
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
import joblib
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ==================== Configuration ====================
MLFLOW_TRACKING_URI   = "http://mlflow.mlflow.svc.cluster.local:80"
REGISTERED_MODEL_NAME = "bme680-anomaly-detector"
S3_BUCKET             = "iot-bme680-data-lake-prod"
S3_SILVER_PREFIX      = "silver/bme680_features/"
S3_GOLD_PREFIX        = "gold/bme680_predictions/"
S3_REPORT_PREFIX      = "model-comparison/reports/"

# Đúng với training pipeline — chỉ 4 core sensor features
FEATURE_COLS = [
    'temperature', 'humidity', 'pressure', 'gas_resistance'
]

DOMAIN_THRESHOLDS = {
    'iaq_score_max':   150.0,
    'temperature_max':  33.0,
    'temperature_min':  28.0,
    'humidity_max':     70.0,
    'humidity_min':     60.0,
}

RANDOM_STATE = 42


# ==================== Tasks ====================

def load_models_from_registry(**context):
    """
    Load Champion (Production) và Challenger (Staging) từ MLflow.
    Nếu chưa có Production, so sánh 2 Staging versions gần nhất.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Champion & Challenger from MLflow")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ── Champion: Production stage ───────────────────────────────────────────
    prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    champion_info = None
    if prod_versions:
        v = prod_versions[0]
        champion_info = {
            'version':    v.version,
            'stage':      v.current_stage,
            'run_id':     v.run_id,
            'model_uri':  f"models:/{REGISTERED_MODEL_NAME}/Production",
            'scaler_uri': f"runs:/{v.run_id}/preprocessor/scaler.pkl",
            'created_at': str(v.creation_timestamp),
            'description': v.description or '',
        }
        logger.info(f"✅ Champion: version {v.version} (Production) — run_id={v.run_id}")
    else:
        logger.warning("⚠️ Không có Production model — sẽ dùng oldest Staging làm champion")

    # ── Challenger: Staging stage ─────────────────────────────────────────────
    staging_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        raise ValueError("❌ Không có Staging model để compare — training pipeline chưa register?")

    v = staging_versions[0]
    challenger_info = {
        'version':    v.version,
        'stage':      v.current_stage,
        'run_id':     v.run_id,
        'model_uri':  f"models:/{REGISTERED_MODEL_NAME}/Staging",
        'scaler_uri': f"runs:/{v.run_id}/preprocessor/scaler.pkl",
        'created_at': str(v.creation_timestamp),
        'description': v.description or '',
    }
    logger.info(f"✅ Challenger: version {v.version} (Staging) — run_id={v.run_id}")

    # Fallback: nếu không có Production, tạo champion giả từ version cũ hơn
    if champion_info is None:
        all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        older = [
            mv for mv in all_versions
            if mv.version != challenger_info['version']
        ]
        if older:
            older_sorted = sorted(older, key=lambda x: int(x.version), reverse=True)
            old_v = older_sorted[0]
            champion_info = {
                'version':    old_v.version,
                'stage':      old_v.current_stage,
                'run_id':     old_v.run_id,
                'model_uri':  f"runs:/{old_v.run_id}/model",
                'scaler_uri': f"runs:/{old_v.run_id}/preprocessor/scaler.pkl",
                'created_at': str(old_v.creation_timestamp),
                'description': old_v.description or '',
                'note': 'No Production model — using previous version as champion',
            }
            logger.warning(
                f"⚠️ Dùng version {old_v.version} làm champion (không có Production stage)"
            )
        else:
            raise ValueError("❌ Chỉ có 1 model version — không thể so sánh champion/challenger")

    # Lấy MLflow metrics của cả 2 từ registry (metrics đã log lúc training)
    for label, info in [('champion', champion_info), ('challenger', challenger_info)]:
        try:
            run = client.get_run(info['run_id'])
            info['logged_metrics'] = {
                k: round(v, 4) for k, v in run.data.metrics.items()
            }
            info['logged_params'] = dict(run.data.params)
            info['training_date'] = run.data.params.get('training_date', 'unknown')
        except Exception as e:
            logger.warning(f"⚠️ Không lấy được MLflow data cho {label}: {e}")
            info['logged_metrics'] = {}
            info['logged_params']  = {}
            info['training_date']  = 'unknown'

    # ── Guard: champion và challenger phải là 2 version khác nhau ─
    if champion_info['version'] == challenger_info['version']:
        raise ValueError(
            f"❌ Champion và Challenger cùng version {champion_info['version']} — "
            f"không có model mới để so sánh.\n"
            f"   Nguyên nhân có thể: training lần gần nhất fail threshold "
            f"(skip_registration) hoặc chưa có lần training thứ 2 thành công."
        )

    # ── Guard: challenger phải mới hơn champion ────────────────────
    if int(challenger_info['version']) <= int(champion_info['version']):
        # Lấy thêm context để debug
        all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        version_summary = [(mv.version, mv.current_stage) for mv in all_versions]
        raise ValueError(
            f"❌ Challenger version {challenger_info['version']} không mới hơn "
            f"Champion version {champion_info['version']} — "
            f"registry state bất thường, kiểm tra lại MLflow.\n"
            f"   All versions: {version_summary}"
        )

    logger.info(
        f"✅ Champion v{champion_info['version']} vs Challenger v{challenger_info['version']} "
        f"— hợp lệ để so sánh"
    )
  
    context['ti'].xcom_push(key='champion_info',   value=champion_info)
    context['ti'].xcom_push(key='challenger_info', value=challenger_info)

    logger.info(f"\n📋 Champion  metrics (logged): {champion_info.get('logged_metrics', {})}")
    logger.info(f"📋 Challenger metrics (logged): {challenger_info.get('logged_metrics', {})}")

    return {
        'champion_version':   champion_info['version'],
        'challenger_version': challenger_info['version'],
    }


def prepare_evaluation_dataset(**context):
    """
    Load Silver data 7 ngày gần nhất làm shared test set.
    Dùng CÙNG test set cho cả champion và challenger để so sánh fair.
    Áp dụng lại DOMAIN_THRESHOLDS để tạo labels.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Prepare Shared Evaluation Dataset")
    logger.info("=" * 60)

    s3_hook        = S3Hook(aws_conn_id='aws_default')
    execution_date = context.get('logical_date') or context.get('execution_date')

    # Lấy Silver 7 ngày gần nhất (không overlap với training window để avoid leakage)
    all_files = []
    for days_back in range(1, 8):
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
        raise ValueError("❌ Không có Silver data 7 ngày gần nhất để evaluation")

    all_data = []
    for file_key in all_files:
        try:
            obj = s3_hook.get_key(file_key, bucket_name=S3_BUCKET)
            df  = pd.read_parquet(BytesIO(obj.get()['Body'].read()))
            all_data.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Không đọc được {file_key}: {e}")

    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"📊 Total Silver records: {len(df):,}")

    # Tạo labels bằng DOMAIN_THRESHOLDS (giống training pipeline)
    df['is_anomaly'] = (
        (df['iaq_score']   > DOMAIN_THRESHOLDS['iaq_score_max']) |
        (df['temperature'] > DOMAIN_THRESHOLDS['temperature_max']) |
        (df['temperature'] < DOMAIN_THRESHOLDS['temperature_min']) |
        (df['humidity']    > DOMAIN_THRESHOLDS['humidity_max'])    |
        (df['humidity']    < DOMAIN_THRESHOLDS['humidity_min'])
    ).astype(int)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    df_clean = df[available_features + ['is_anomaly']].dropna()

    anomaly_rate = float(df_clean['is_anomaly'].mean())
    logger.info(f"📊 Eval dataset: {len(df_clean):,} records, anomaly_rate={anomaly_rate:.2%}")

    if df_clean['is_anomaly'].sum() < 10:
        logger.warning("⚠️ Test set có ít hơn 10 anomalies — metrics có thể không ổn định")

    # Không split — dùng toàn bộ làm evaluation set (đây là "held-out" của thời gian gần nhất)
    context['ti'].xcom_push(key='eval_data',         value=df_clean.to_dict('records'))
    context['ti'].xcom_push(key='available_features', value=available_features)
    context['ti'].xcom_push(key='eval_anomaly_rate', value=anomaly_rate)

    return {'eval_count': len(df_clean), 'anomaly_rate': anomaly_rate}


def evaluate_model(model_info: dict, df_eval: pd.DataFrame, features: list) -> dict:
    """
    Chạy inference và tính metrics cho một model.
    Helper function dùng chung cho champion và challenger.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load model
    model = mlflow.sklearn.load_model(model_info['model_uri'])

    # Load scaler
    scaler_path = mlflow.artifacts.download_artifacts(model_info['scaler_uri'])
    scaler      = joblib.load(scaler_path)

    # Align features với scaler
    if hasattr(scaler, 'feature_names_in_'):
        train_features = list(scaler.feature_names_in_)
        available = [f for f in train_features if f in df_eval.columns]
        X = df_eval[available]
    else:
        X = df_eval[features]

    y_true = df_eval['is_anomaly'].values

    # Scale và predict
    X_scaled    = scaler.transform(X)
    y_pred      = (model.predict(X_scaled) == -1).astype(int)
    scores      = -model.score_samples(X_scaled)   # higher = more anomalous

    # Metrics
    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, scores)) if has_both_classes else 0.0

    metrics = {
        'roc_auc':   round(roc_auc, 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_true,        y_pred, zero_division=0)), 4),
        'accuracy':  round(float(accuracy_score(y_true,  y_pred)), 4),
        'anomaly_rate_predicted': round(float(y_pred.mean()), 4),
        'anomaly_rate_true':      round(float(y_true.mean()), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Anomaly score stats (để so sánh score distribution)
    score_stats = {
        'mean':   round(float(scores.mean()), 4),
        'std':    round(float(scores.std()),  4),
        'p50':    round(float(np.percentile(scores, 50)), 4),
        'p95':    round(float(np.percentile(scores, 95)), 4),
        'p99':    round(float(np.percentile(scores, 99)), 4),
    }

    # Feature importance proxy: mean path length depth từ IsolationForest
    # Shorter path = more anomalous = feature contributes more to isolation
    importance = {}
    try:
        if hasattr(model, 'estimators_'):
            # Tính average depth per feature bằng cách đo variance của score
            # khi từng feature bị permute (permutation importance approximation)
            base_scores = scores.copy()
            for i, feat in enumerate(X.columns):
                X_permuted       = X_scaled.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_scores  = -model.score_samples(X_permuted)
                importance[feat] = round(float(np.mean(np.abs(permuted_scores - base_scores))), 4)
    except Exception as e:
        logger.warning(f"⚠️ Không tính được feature importance: {e}")

    return {
        'metrics':      metrics,
        'confusion_matrix': cm,
        'score_stats':  score_stats,
        'importance':   importance,
        'n_eval':       len(df_eval),
    }


def run_evaluation(**context):
    """
    Chạy evaluation cho cả champion và challenger trên cùng test set.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Run Evaluation — Champion vs Challenger")
    logger.info("=" * 60)

    champion_info   = context['ti'].xcom_pull(task_ids='load_models',   key='champion_info')
    challenger_info = context['ti'].xcom_pull(task_ids='load_models',   key='challenger_info')
    eval_data       = context['ti'].xcom_pull(task_ids='prepare_eval',  key='eval_data')
    features        = context['ti'].xcom_pull(task_ids='prepare_eval',  key='available_features')

    df_eval = pd.DataFrame(eval_data)

    logger.info("🏆 Evaluating Champion...")
    champion_eval   = evaluate_model(champion_info,   df_eval, features)

    logger.info("🆕 Evaluating Challenger...")
    challenger_eval = evaluate_model(challenger_info, df_eval, features)

    # ── Delta comparison ─────────────────────────────────────────────────────
    champ_m  = champion_eval['metrics']
    chall_m  = challenger_eval['metrics']

    deltas = {
        metric: round(chall_m[metric] - champ_m[metric], 4)
        for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']
    }

    logger.info("\n" + "=" * 70)
    logger.info(f"{'Metric':<20} {'Champion':>12} {'Challenger':>12} {'Delta':>10} {'Better?':>10}")
    logger.info("-" * 70)
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']:
        delta  = deltas[metric]
        better = '✅ Challenger' if delta > 0 else ('➡️ Equal' if delta == 0 else '⚠️ Champion')
        logger.info(
            f"  {metric:<18} {champ_m[metric]:>12.4f} {chall_m[metric]:>12.4f} "
            f"{delta:>+10.4f} {better:>10}"
        )

    eval_results = {
        'champion':   champion_eval,
        'challenger': challenger_eval,
        'deltas':     deltas,
    }

    context['ti'].xcom_push(key='eval_results', value=eval_results)
    return deltas


def generate_recommendation(**context):
    """
    Tạo AI-generated recommendation để human tham khảo.
    Logic dựa trên priority: recall > roc_auc > precision (domain-specific cho air anomaly).

    Output là structured recommendation object, không phải quyết định.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Generate Recommendation")
    logger.info("=" * 60)

    eval_results    = context['ti'].xcom_pull(task_ids='run_evaluation', key='eval_results')
    champion_info   = context['ti'].xcom_pull(task_ids='load_models',   key='champion_info')
    challenger_info = context['ti'].xcom_pull(task_ids='load_models',   key='challenger_info')

    champ_m  = eval_results['champion']['metrics']
    chall_m  = eval_results['challenger']['metrics']
    deltas   = eval_results['deltas']

    # ── Scoring logic — recall có trọng số cao nhất cho air anomaly ──────────
    # Lý do: miss một sự kiện air quality bất thường nguy hiểm hơn false alarm
    WEIGHTS = {'recall': 0.40, 'roc_auc': 0.35, 'precision': 0.15, 'f1': 0.10}

    champion_score   = sum(champ_m[m]  * w for m, w in WEIGHTS.items())
    challenger_score = sum(chall_m[m]  * w for m, w in WEIGHTS.items())
    score_delta      = round(challenger_score - champion_score, 4)

    # ── Phân tích từng chiều ─────────────────────────────────────────────────
    analysis = []
    risks    = []

    if deltas['recall'] > 0.02:
        analysis.append(f"Recall cải thiện đáng kể (+{deltas['recall']:.3f}) — ít miss anomaly hơn")
    elif deltas['recall'] < -0.02:
        risks.append(f"Recall giảm {deltas['recall']:.3f} — nguy cơ miss thêm anomaly")
    else:
        analysis.append(f"Recall ổn định (Δ={deltas['recall']:+.3f})")

    if deltas['roc_auc'] > 0.01:
        analysis.append(f"ROC-AUC tốt hơn (+{deltas['roc_auc']:.3f}) — khả năng phân biệt anomaly tốt hơn")
    elif deltas['roc_auc'] < -0.01:
        risks.append(f"ROC-AUC giảm nhẹ ({deltas['roc_auc']:.3f})")

    if deltas['precision'] < -0.05:
        risks.append(f"Precision giảm {deltas['precision']:.3f} — có thể tăng false alarm")
    elif deltas['precision'] > 0.02:
        analysis.append(f"Precision cải thiện (+{deltas['precision']:.3f}) — ít false alarm hơn")

    # ── Quyết định khuyến nghị ────────────────────────────────────────────────
    critical_recall_drop = deltas['recall'] < -0.03
    significant_improvement = score_delta > 0.01

    if critical_recall_drop:
        verdict    = 'GIỮ CHAMPION'
        confidence = 'CAO'
        reason     = (
            f"Recall giảm {deltas['recall']:.3f} — trong bài toán air anomaly detection, "
            "đây là rủi ro không chấp nhận được. Miss anomaly nguy hiểm hơn false alarm."
        )
    elif significant_improvement and not risks:
        verdict    = 'PROMOTE CHALLENGER'
        confidence = 'CAO'
        reason     = (
            f"Challenger cải thiện điểm tổng hợp +{score_delta:.3f} với không có rủi ro đáng kể."
        )
    elif score_delta > 0 and len(risks) <= 1:
        verdict    = 'PROMOTE CHALLENGER'
        confidence = 'TRUNG BÌNH'
        reason     = (
            f"Challenger nhỉnh hơn một chút (Δ={score_delta:+.4f}) với {len(risks)} rủi ro nhỏ. "
            "Nên xem xét thêm context drift."
        )
    elif score_delta < -0.005:
        verdict    = 'GIỮ CHAMPION'
        confidence = 'CAO'
        reason     = f"Champion vẫn tốt hơn về điểm tổng hợp (Δ={score_delta:+.4f})."
    else:
        verdict    = 'XEM XÉT THÊM'
        confidence = 'THẤP'
        reason     = (
            f"Hiệu suất gần tương đương (Δ={score_delta:+.4f}). "
            "Nên xem xét drift context và metadata trước khi quyết định."
        )

    recommendation = {
        'verdict':          verdict,
        'confidence':       confidence,
        'reason':           reason,
        'weighted_score': {
            'champion':   round(champion_score,   4),
            'challenger': round(challenger_score, 4),
            'delta':      round(score_delta,       4),
        },
        'weights_used':    WEIGHTS,
        'analysis_points': analysis,
        'risk_points':     risks,
        'domain_note': (
            "Trọng số recall=0.40 vì trong IoT air quality monitoring, "
            "false negative (miss anomaly) nguy hiểm hơn false positive (false alarm)."
        ),
    }

    logger.info(f"\n🎯 RECOMMENDATION: {verdict} (confidence: {confidence})")
    logger.info(f"   Reason: {reason}")
    logger.info(f"   Analysis: {analysis}")
    logger.info(f"   Risks: {risks}")

    context['ti'].xcom_push(key='recommendation', value=recommendation)
    return recommendation


def load_drift_context(**context):
    """
    Load drift report gần nhất từ S3 để đưa vào model comparison report.
    Cung cấp context: tại sao challenger được tạo ra.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Load Drift Context")
    logger.info("=" * 60)

    s3_hook    = S3Hook(aws_conn_id='aws_default')
    latest_key = "drift/reports/latest/drift_report_latest.json"

    try:
        obj          = s3_hook.get_key(latest_key, bucket_name=S3_BUCKET)
        drift_report = json.loads(obj.get()['Body'].read().decode('utf-8'))
        logger.info(f"✅ Loaded drift report từ: {latest_key}")
        logger.info(f"   Overall drift: {drift_report.get('overall_drift')}")
        logger.info(f"   PSI violations: {drift_report.get('psi_violations')}")
        context['ti'].xcom_push(key='drift_context', value=drift_report)
        return drift_report
    except Exception as e:
        logger.warning(f"⚠️ Không load được drift report: {e} — sẽ bỏ qua section này")
        context['ti'].xcom_push(key='drift_context', value=None)
        return None


def generate_html_report(**context):
    """
    Tạo HTML report đầy đủ để human review.
    Report được thiết kế để mở trực tiếp trên browser — không cần login.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Generate HTML Report")
    logger.info("=" * 60)

    champion_info   = context['ti'].xcom_pull(task_ids='load_models',    key='champion_info')
    challenger_info = context['ti'].xcom_pull(task_ids='load_models',    key='challenger_info')
    eval_results    = context['ti'].xcom_pull(task_ids='run_evaluation', key='eval_results')
    recommendation  = context['ti'].xcom_pull(task_ids='gen_recommendation', key='recommendation')
    drift_context   = context['ti'].xcom_pull(task_ids='load_drift',    key='drift_context')
    eval_anomaly_rate = context['ti'].xcom_pull(task_ids='prepare_eval', key='eval_anomaly_rate')

    champ_m  = eval_results['champion']['metrics']
    chall_m  = eval_results['challenger']['metrics']
    deltas   = eval_results['deltas']
    champ_cm = eval_results['champion']['confusion_matrix']
    chall_cm = eval_results['challenger']['confusion_matrix']

    verdict_color = {
        'PROMOTE CHALLENGER': '#1a7f4b',
        'GIỮ CHAMPION':       '#c0392b',
        'XEM XÉT THÊM':       '#d68910',
    }.get(recommendation['verdict'], '#555')

    confidence_color = {
        'CAO':     '#1a7f4b',
        'TRUNG BÌNH': '#d68910',
        'THẤP':    '#c0392b',
    }.get(recommendation['confidence'], '#555')

    def metric_row(label, champ_val, chall_val, delta, highlight=False):
        """Tạo HTML row cho metrics table."""
        delta_class = 'positive' if delta > 0 else ('negative' if delta < 0 else 'neutral')
        delta_icon  = '▲' if delta > 0 else ('▼' if delta < 0 else '—')
        bg = ' style="background:#f0f8ff;"' if highlight else ''
        return f"""
        <tr{bg}>
          <td class="metric-name">{label}</td>
          <td class="metric-val">{champ_val:.4f}</td>
          <td class="metric-val">{chall_val:.4f}</td>
          <td class="metric-val delta {delta_class}">{delta_icon} {abs(delta):.4f}</td>
        </tr>"""

    def confusion_matrix_html(cm, title, color):
        """Tạo HTML confusion matrix."""
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        return f"""
        <div class="cm-container">
          <div class="cm-title" style="color:{color}">{title}</div>
          <table class="cm-table">
            <tr><th></th><th>Pred Normal</th><th>Pred Anomaly</th></tr>
            <tr>
              <th>True Normal</th>
              <td class="cm-tn">{tn:,}<br><small>{tn/total:.1%}</small></td>
              <td class="cm-fp">{fp:,}<br><small>{fp/total:.1%}</small></td>
            </tr>
            <tr>
              <th>True Anomaly</th>
              <td class="cm-fn">{fn:,}<br><small>{fn/total:.1%}</small></td>
              <td class="cm-tp">{tp:,}<br><small>{tp/total:.1%}</small></td>
            </tr>
          </table>
        </div>"""

    # ── Drift context section ─────────────────────────────────────────────────
    drift_section = ""
    if drift_context:
        feat_rows = ""
        for feat, result in drift_context.get('features', {}).items():
            psi_icon = '🔴' if result.get('psi_flag') else '🟢'
            ks_icon  = '🔴' if result.get('ks_flag')  else '🟢'
            feat_rows += f"""
            <tr>
              <td>{feat}</td>
              <td>{result.get('psi', 0):.4f} {psi_icon}</td>
              <td>{result.get('kl_divergence', 0):.4f}</td>
              <td>{result.get('ks_pvalue', 0):.4f} {ks_icon}</td>
              <td>{result.get('mean_shift', 0):+.3f}</td>
            </tr>"""

        drift_section = f"""
        <div class="section">
          <h2>📊 Drift Context — Lý do Challenger được tạo</h2>
          <div class="drift-summary">
            <span class="badge {'badge-red' if drift_context.get('overall_drift') else 'badge-green'}">
              Overall Drift: {'DETECTED' if drift_context.get('overall_drift') else 'NONE'}
            </span>
            <span>KL mean = {drift_context.get('kl_mean', 0):.4f}</span>
            <span>PSI violations: {drift_context.get('psi_violations', [])}</span>
            <span>Production window: {drift_context.get('production_window_days', 7)} ngày</span>
          </div>
          <table class="data-table">
            <thead><tr><th>Feature</th><th>PSI</th><th>KL Div</th><th>KS p-value</th><th>Mean Shift</th></tr></thead>
            <tbody>{feat_rows}</tbody>
          </table>
        </div>"""

    # ── Feature importance section ─────────────────────────────────────────────
    importance_rows = ""
    all_features = set(eval_results['champion']['importance'].keys()) | \
                   set(eval_results['challenger']['importance'].keys())
    for feat in sorted(all_features):
        champ_imp = eval_results['champion']['importance'].get(feat, 0)
        chall_imp = eval_results['challenger']['importance'].get(feat, 0)
        importance_rows += f"""
        <tr>
          <td>{feat}</td>
          <td>{champ_imp:.4f}</td>
          <td>{chall_imp:.4f}</td>
        </tr>"""

    importance_section = f"""
        <div class="section">
          <h2>🔍 Feature Importance (permutation-based)</h2>
          <p class="note">Giá trị cao = feature đóng góp nhiều vào quyết định anomaly</p>
          <table class="data-table">
            <thead><tr><th>Feature</th><th>Champion</th><th>Challenger</th></tr></thead>
            <tbody>{importance_rows if importance_rows else '<tr><td colspan="3">Không tính được</td></tr>'}</tbody>
          </table>
        </div>""" if importance_rows else ""

    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Comparison Report — {now_str}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f6fa; color: #2d3436; line-height: 1.6; }}
    .header {{ background: linear-gradient(135deg, #1e3a5f, #2980b9);
               color: white; padding: 32px 40px; }}
    .header h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; }}
    .header .meta {{ font-size: 13px; opacity: 0.85; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
    .section {{ background: white; border-radius: 10px; padding: 28px 32px;
                margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    h2 {{ font-size: 17px; font-weight: 600; margin-bottom: 16px;
          padding-bottom: 10px; border-bottom: 1px solid #ecf0f1; }}
    .recommendation-box {{ border-left: 5px solid {verdict_color};
                            background: #fafbfc; border-radius: 6px;
                            padding: 20px 24px; margin-bottom: 16px; }}
    .verdict {{ font-size: 22px; font-weight: 700; color: {verdict_color};
                margin-bottom: 8px; }}
    .confidence {{ font-size: 13px; color: {confidence_color};
                   font-weight: 600; margin-bottom: 12px; }}
    .reason {{ font-size: 14px; color: #555; margin-bottom: 12px; }}
    .bullet-list {{ list-style: none; padding: 0; }}
    .bullet-list li {{ font-size: 13px; padding: 3px 0; }}
    .bullet-list li:before {{ content: '• '; color: #7f8c8d; }}
    .risk-list li:before {{ content: '⚠ '; color: #e67e22; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; }}
    .metrics-table th {{ background: #f8f9fa; text-align: center;
                         padding: 10px 14px; font-size: 13px;
                         border-bottom: 2px solid #dee2e6; }}
    .metric-name {{ font-size: 13px; padding: 9px 14px; font-weight: 500; }}
    .metric-val  {{ text-align: center; padding: 9px 14px; font-size: 14px;
                    font-family: monospace; }}
    .metrics-table tr:hover {{ background: #f8f9fa; }}
    .delta.positive {{ color: #1a7f4b; font-weight: 600; }}
    .delta.negative {{ color: #c0392b; font-weight: 600; }}
    .delta.neutral  {{ color: #7f8c8d; }}
    .cm-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px; }}
    .cm-container {{ flex: 1; min-width: 280px; }}
    .cm-title {{ font-weight: 600; font-size: 14px; margin-bottom: 10px; }}
    .cm-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .cm-table th {{ background: #f8f9fa; padding: 8px 12px;
                    text-align: center; font-weight: 500;
                    border: 1px solid #dee2e6; }}
    .cm-table td {{ padding: 10px 12px; text-align: center;
                    border: 1px solid #dee2e6; }}
    .cm-tn {{ background: #eafaf1; color: #1a7f4b; font-weight: 600; }}
    .cm-tp {{ background: #eafaf1; color: #1a7f4b; font-weight: 600; }}
    .cm-fp {{ background: #fef9e7; color: #d68910; }}
    .cm-fn {{ background: #fdedec; color: #c0392b; font-weight: 600; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table th {{ background: #f8f9fa; padding: 9px 14px;
                      text-align: left; font-weight: 500;
                      border-bottom: 2px solid #dee2e6; }}
    .data-table td {{ padding: 8px 14px; border-bottom: 1px solid #ecf0f1; }}
    .data-table tr:hover {{ background: #f8f9fa; }}
    .model-meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .meta-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px 20px; }}
    .meta-card h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 10px; }}
    .meta-row {{ display: flex; justify-content: space-between;
                 font-size: 12px; padding: 3px 0;
                 border-bottom: 1px solid #ecf0f1; }}
    .meta-label {{ color: #7f8c8d; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
              font-size: 12px; font-weight: 600; margin-right: 8px; }}
    .badge-red   {{ background: #fdedec; color: #c0392b; }}
    .badge-green {{ background: #eafaf1; color: #1a7f4b; }}
    .drift-summary {{ display: flex; gap: 16px; flex-wrap: wrap;
                      align-items: center; margin-bottom: 14px;
                      font-size: 13px; color: #555; }}
    .note {{ font-size: 12px; color: #7f8c8d; margin-bottom: 12px; }}
    .weighted-score-row {{ display: flex; gap: 24px; margin-top: 12px;
                            flex-wrap: wrap; }}
    .score-box {{ flex: 1; min-width: 160px; text-align: center;
                  background: #f8f9fa; border-radius: 8px; padding: 14px; }}
    .score-label {{ font-size: 12px; color: #7f8c8d; margin-bottom: 4px; }}
    .score-val {{ font-size: 22px; font-weight: 700; font-family: monospace; }}
    .footer {{ text-align: center; font-size: 12px; color: #95a5a6;
               padding: 24px 0 40px; }}
  </style>
</head>
<body>
<div class="header">
  <h1>🤖 Model Comparison Report — Champion vs Challenger</h1>
  <div class="meta">
    Generated: {now_str} &nbsp;|&nbsp;
    Champion: {REGISTERED_MODEL_NAME} v{champion_info['version']} &nbsp;|&nbsp;
    Challenger: {REGISTERED_MODEL_NAME} v{challenger_info['version']}
  </div>
</div>

<div class="container">

  <!-- RECOMMENDATION -->
  <div class="section">
    <h2>🎯 Recommendation (để tham khảo — quyết định cuối thuộc về human)</h2>
    <div class="recommendation-box">
      <div class="verdict">{recommendation['verdict']}</div>
      <div class="confidence">Confidence: {recommendation['confidence']}</div>
      <div class="reason">{recommendation['reason']}</div>
      <div class="weighted-score-row">
        <div class="score-box">
          <div class="score-label">Champion score</div>
          <div class="score-val" style="color:#2980b9">{recommendation['weighted_score']['champion']:.4f}</div>
        </div>
        <div class="score-box">
          <div class="score-label">Challenger score</div>
          <div class="score-val" style="color:#27ae60">{recommendation['weighted_score']['challenger']:.4f}</div>
        </div>
        <div class="score-box">
          <div class="score-label">Delta</div>
          <div class="score-val" style="color:{verdict_color}">{recommendation['weighted_score']['delta']:+.4f}</div>
        </div>
      </div>
    </div>
    {"<p><strong>Điểm tích cực:</strong></p><ul class='bullet-list'>" + "".join(f"<li>{a}</li>" for a in recommendation['analysis_points']) + "</ul>" if recommendation['analysis_points'] else ""}
    {"<p style='margin-top:10px'><strong>Rủi ro:</strong></p><ul class='bullet-list risk-list'>" + "".join(f"<li>{r}</li>" for r in recommendation['risk_points']) + "</ul>" if recommendation['risk_points'] else ""}
    <p class="note" style="margin-top:12px">{recommendation['domain_note']}</p>
    <p class="note">Weights: recall={recommendation['weights_used']['recall']}, roc_auc={recommendation['weights_used']['roc_auc']}, precision={recommendation['weights_used']['precision']}, f1={recommendation['weights_used']['f1']}</p>
  </div>

  <!-- PERFORMANCE METRICS -->
  <div class="section">
    <h2>📊 Performance Metrics — Eval trên shared test set (7 ngày gần nhất)</h2>
    <p class="note">
      Eval set: {eval_results['champion']['n_eval']:,} records &nbsp;|&nbsp;
      Anomaly rate thực tế: {eval_anomaly_rate:.2%}
    </p>
    <table class="metrics-table">
      <thead>
        <tr>
          <th style="text-align:left">Metric</th>
          <th>Champion (v{champion_info['version']})</th>
          <th>Challenger (v{challenger_info['version']})</th>
          <th>Delta (Challenger − Champion)</th>
        </tr>
      </thead>
      <tbody>
        {metric_row('ROC-AUC ⭐',  champ_m['roc_auc'],   chall_m['roc_auc'],   deltas['roc_auc'],   highlight=True)}
        {metric_row('Recall ⭐⭐',  champ_m['recall'],    chall_m['recall'],    deltas['recall'],    highlight=True)}
        {metric_row('Precision',    champ_m['precision'], chall_m['precision'], deltas['precision'])}
        {metric_row('F1 Score',     champ_m['f1'],        chall_m['f1'],        deltas['f1'])}
        {metric_row('Accuracy',     champ_m['accuracy'],  chall_m['accuracy'],  deltas['accuracy'])}
        {metric_row('Predicted anomaly rate',
                    champ_m['anomaly_rate_predicted'],
                    chall_m['anomaly_rate_predicted'],
                    chall_m['anomaly_rate_predicted'] - champ_m['anomaly_rate_predicted'])}
      </tbody>
    </table>
    <p class="note" style="margin-top:10px">⭐ = metric quan trọng | ⭐⭐ = metric quan trọng nhất (air anomaly domain)</p>
  </div>

  <!-- CONFUSION MATRIX -->
  <div class="section">
    <h2>🔢 Confusion Matrix</h2>
    <div class="cm-row">
      {confusion_matrix_html(champ_cm, f'Champion v{champion_info["version"]}', '#2980b9')}
      {confusion_matrix_html(chall_cm, f'Challenger v{challenger_info["version"]}', '#27ae60')}
    </div>
    <p class="note" style="margin-top:14px">
      TN = True Normal | FP = False Alarm | FN = Missed Anomaly ⚠️ | TP = Correct Anomaly Detected
    </p>
  </div>

  <!-- ANOMALY SCORE DISTRIBUTION -->
  <div class="section">
    <h2>📉 Anomaly Score Distribution</h2>
    <table class="data-table">
      <thead><tr><th>Stat</th><th>Champion v{champion_info['version']}</th><th>Challenger v{challenger_info['version']}</th></tr></thead>
      <tbody>
        {''.join(f"<tr><td>{k}</td><td>{eval_results['champion']['score_stats'][k]}</td><td>{eval_results['challenger']['score_stats'][k]}</td></tr>" for k in ['mean','std','p50','p95','p99'])}
      </tbody>
    </table>
    <p class="note" style="margin-top:8px">Score cao = model coi là anomaly. p95/p99 cao hơn = model nhạy hơn với outliers cực đoan.</p>
  </div>

  {importance_section}

  {drift_section}

  <!-- MODEL METADATA -->
  <div class="section">
    <h2>📦 Model Metadata</h2>
    <div class="model-meta-grid">
      <div class="meta-card">
        <h3 style="color:#2980b9">🏆 Champion — v{champion_info['version']}</h3>
        {''.join(f'<div class="meta-row"><span class="meta-label">{k}</span><span>{v}</span></div>' for k,v in [('Stage', champion_info.get('stage','')), ('Training date', champion_info.get('training_date','')), ('Run ID', champion_info.get('run_id','')[:16]+'...' if champion_info.get('run_id') else ''), ('n_estimators', champion_info.get('logged_params',{}).get('n_estimators','')), ('contamination', champion_info.get('logged_params',{}).get('contamination','')), ('split_method', champion_info.get('logged_params',{}).get('split_method',''))])}
        <div class="meta-row"><span class="meta-label">Description</span><span style="font-size:11px">{champion_info.get('description','')[:120]}</span></div>
      </div>
      <div class="meta-card">
        <h3 style="color:#27ae60">🆕 Challenger — v{challenger_info['version']}</h3>
        {''.join(f'<div class="meta-row"><span class="meta-label">{k}</span><span>{v}</span></div>' for k,v in [('Stage', challenger_info.get('stage','')), ('Training date', challenger_info.get('training_date','')), ('Run ID', challenger_info.get('run_id','')[:16]+'...' if challenger_info.get('run_id') else ''), ('n_estimators', challenger_info.get('logged_params',{}).get('n_estimators','')), ('contamination', challenger_info.get('logged_params',{}).get('contamination','')), ('split_method', challenger_info.get('logged_params',{}).get('split_method',''))])}
        <div class="meta-row"><span class="meta-label">Description</span><span style="font-size:11px">{challenger_info.get('description','')[:120]}</span></div>
      </div>
    </div>
  </div>

</div>
<div class="footer">
  IoT BME680 MLOps — Model Comparison Report &nbsp;|&nbsp; Generated {now_str}
  <br>Quyết định cuối cùng thuộc về human reviewer. Report này chỉ mang tính tham khảo.
</div>
</body>
</html>"""

    # Upload HTML report lên S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    now     = datetime.utcnow()

    report_key = (
        f"{S3_REPORT_PREFIX}"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/"
        f"model_comparison_v{champion_info['version']}_vs_v{challenger_info['version']}"
        f"_{now.strftime('%Y%m%d_%H%M%S')}.html"
    )

    s3_hook.load_bytes(
        bytes_data=html.encode('utf-8'),
        key=report_key,
        bucket_name=S3_BUCKET,
        replace=True,
        acl_policy='bucket-owner-full-control',
    )
    logger.info(f"💾 HTML report saved: s3://{S3_BUCKET}/{report_key}")

    # Cũng lưu JSON summary để audit
    summary = {
        'generated_at':    now_str,
        'champion':        {'version': champion_info['version'], 'metrics': champ_m},
        'challenger':      {'version': challenger_info['version'], 'metrics': chall_m},
        'deltas':          deltas,
        'recommendation':  recommendation,
        'report_s3_key':   report_key,
    }
    summary_key = report_key.replace('.html', '_summary.json')
    s3_hook.load_bytes(
        bytes_data=json.dumps(summary, indent=2, default=str).encode('utf-8'),
        key=summary_key,
        bucket_name=S3_BUCKET,
        replace=True,
    )

    context['ti'].xcom_push(key='report_s3_key', value=report_key)
    return report_key


def send_review_notification(**context):
    """
    Gửi notification kèm link report để human review.
    Tạo presigned URL (24h) để mở report trực tiếp không cần login AWS.
    """
    logger.info("=" * 60)
    logger.info("STEP 7: Send Review Notification")
    logger.info("=" * 60)

    champion_info   = context['ti'].xcom_pull(task_ids='load_models',       key='champion_info')
    challenger_info = context['ti'].xcom_pull(task_ids='load_models',       key='challenger_info')
    recommendation  = context['ti'].xcom_pull(task_ids='gen_recommendation', key='recommendation')
    report_key      = context['ti'].xcom_pull(task_ids='gen_report',        key='report_s3_key')
    eval_results    = context['ti'].xcom_pull(task_ids='run_evaluation',    key='eval_results')

    champ_m = eval_results['champion']['metrics']
    chall_m = eval_results['challenger']['metrics']
    deltas  = eval_results['deltas']

    # Tạo presigned URL (48h) — human không cần credentials AWS
    s3_hook = S3Hook(aws_conn_id='aws_default')
    try:
        s3_client = s3_hook.get_conn()
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': report_key},
            ExpiresIn=172800,  # 48 hours
        )
    except Exception as e:
        logger.warning(f"⚠️ Không tạo được presigned URL: {e}")
        presigned_url = f"s3://{S3_BUCKET}/{report_key}"

    verdict    = recommendation['verdict']
    confidence = recommendation['confidence']

    message = f"""
🤖 MODEL COMPARISON REPORT — Human Review Required

📋 Tóm tắt:
   Champion : {REGISTERED_MODEL_NAME} v{champion_info['version']} ({champion_info.get('training_date', '')})
   Challenger: {REGISTERED_MODEL_NAME} v{challenger_info['version']} ({challenger_info.get('training_date', '')})

📊 Key Metrics Comparison:
   {'Metric':<20} {'Champion':>10} {'Challenger':>12} {'Delta':>8}
   {'─'*54}
   {'ROC-AUC':<20} {champ_m['roc_auc']:>10.4f} {chall_m['roc_auc']:>12.4f} {deltas['roc_auc']:>+8.4f}
   {'Recall ⭐':<20} {champ_m['recall']:>10.4f} {chall_m['recall']:>12.4f} {deltas['recall']:>+8.4f}
   {'Precision':<20} {champ_m['precision']:>10.4f} {chall_m['precision']:>12.4f} {deltas['precision']:>+8.4f}
   {'F1':<20} {champ_m['f1']:>10.4f} {chall_m['f1']:>12.4f} {deltas['f1']:>+8.4f}

🎯 AI Recommendation: {verdict} (confidence: {confidence})
   {recommendation['reason']}

⚡ ACTION REQUIRED: Bạn cần quyết định promote hay giữ Champion
   1. Mở report để xem chi tiết đầy đủ
   2. Nếu promote → transition Challenger v{challenger_info['version']} lên Production trong MLflow
   3. Nếu giữ → archive Challenger v{challenger_info['version']}

🔗 Full Report (valid 48h): {presigned_url}
"""
    logger.info(message)

    # TODO: uncomment khi có Slack webhook
    # import requests
    # requests.post(SLACK_WEBHOOK_URL, json={
    #     "text": message,
    #     "blocks": [
    #         {"type": "section", "text": {"type": "mrkdwn", "text": f"*{verdict}*"}},
    #         {"type": "actions", "elements": [
    #             {"type": "button", "text": {"type": "plain_text", "text": "View Report"},
    #              "url": presigned_url}
    #         ]}
    #     ]
    # })

    return presigned_url


# ==================== DAG ====================

default_args = {
    'owner':             'iot-ml-team',
    'depends_on_past':   False,
    'email':             ['iot-ml@company.com'],
    'email_on_failure':  True,
    'email_on_retry':    False,
    'retries':           1,
    'retry_delay':       timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

with DAG(
    dag_id='iot_ml_model_comparison',
    description='Champion vs Challenger evaluation & HTML report for human review',
    schedule=None,   # Chỉ trigger từ training pipeline hoặc thủ công
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['ml', 'model-comparison', 'champion-challenger', 'iot', 'human-review'],
    doc_md="""
## Model Comparison DAG

**Trigger**: Tự động sau `iot_ml_training_pipeline` hoặc manual.

**Output**: HTML report trên S3 + presigned URL gửi notification.

**Flow:**
1. Load Champion (Production) + Challenger (Staging) từ MLflow
2. Prepare shared eval dataset (Silver 7 ngày, same test set cho cả 2)
3. Run evaluation: ROC-AUC, Precision, Recall, F1, Confusion Matrix
4. Generate recommendation (recall-weighted, tham khảo)
5. Load drift context từ latest drift report
6. Generate HTML report → S3
7. Send notification với presigned URL (48h)

**Human action sau khi nhận report:**
- Promote: MLflow UI → transition Challenger → Production
- Giữ: Archive Challenger trong MLflow
"""
) as dag:

    t_models     = PythonOperator(task_id='load_models',        python_callable=load_models_from_registry)
    t_eval_data  = PythonOperator(task_id='prepare_eval',       python_callable=prepare_evaluation_dataset)
    t_eval       = PythonOperator(task_id='run_evaluation',     python_callable=run_evaluation)
    t_recommend  = PythonOperator(task_id='gen_recommendation', python_callable=generate_recommendation)
    t_drift      = PythonOperator(task_id='load_drift',         python_callable=load_drift_context)
    t_report     = PythonOperator(task_id='gen_report',         python_callable=generate_html_report)
    t_notify     = PythonOperator(task_id='send_notification',  python_callable=send_review_notification)

    # ── DAG flow ──────────────────────────────────────────────────────────────
    # load_models + prepare_eval song song → run_evaluation
    # gen_recommendation + load_drift song song → gen_report (cần cả 2)
    [t_models, t_eval_data] >> t_eval >> [t_recommend, t_drift] >> t_report >> t_notify
