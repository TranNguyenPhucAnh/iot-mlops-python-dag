# =============================================
# Airflow MLflow Image - Optimized for Jenkins DinD
# =============================================

ARG AIRFLOW_VERSION=3.0.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

USER root

# Cài đặt các thư viện hệ thống cần thiết cho C-extensions và AWS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements.txt vào container
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt

# Cài đặt Python dependencies
# Sử dụng --upgrade và --constraint để đảm bảo không phá vỡ Airflow core
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        -r /tmp/requirements.txt \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"

# Jenkins sẽ chạy cái này khi build để đảm bảo image "sạch"
RUN python <<EOF
import mlflow, boto3, psycopg2
print(f'✅ MLflow version: {mlflow.__version__}')
print(f'✅ Boto3 version: {boto3.__version__}')
print('🎉 Verification Successful!')
EOF

# Metadata cho Image
LABEL maintainer="phucanhatt@gmail.com" \
      org.opencontainers.image.title="Airflow MLflow IoT" \
      org.opencontainers.image.description="Custom Airflow image with MLflow for IoT MLOps"
