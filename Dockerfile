# ============================================
# Airflow MLflow Image
# Base: Apache Airflow 3.0.2
# Features: MLflow tracking, AWS integrations, IoT sensors
# ============================================

ARG AIRFLOW_VERSION=3.0.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

# Build arguments for labels
ARG BUILD_DATE
ARG VCS_REF
ARG AIRFLOW_VERSION

# OCI Standard Labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="IoT Platform Team" \
      org.opencontainers.image.url="https://github.com/TranNguyenPhucAnh/iot-mlops-python-dag" \
      org.opencontainers.image.version="${AIRFLOW_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="Airflow MLflow IoT" \
      org.opencontainers.image.description="Apache Airflow with MLflow tracking and IoT integrations" \
      maintainer="phucanhatt@gmail.com"

# ============================================
# System Dependencies
# ============================================
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Build tools for Python packages with C extensions
        gcc \
        g++ \
        python3-dev \
        libpq-dev \
        # IoT sensor libraries (if needed for edge processing)
        i2c-tools \
        # Utilities
        curl \
        vim \
        && \
    # Cleanup
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================
# Python Dependencies
# ============================================
USER airflow

# Copy requirements first for better layer caching
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt

# Install Python packages with constraints
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt" \
    -r /tmp/requirements.txt && \
    # Cleanup
    rm -rf /home/airflow/.cache/pip

# ============================================
# Verification
# ============================================
RUN python -c "
import sys
print('Python version:', sys.version)

# Verify critical imports
import airflow
print('✅ Airflow version:', airflow.__version__)

import mlflow
print('✅ MLflow version:', mlflow.__version__)

import boto3
print('✅ Boto3 version:', boto3.__version__)

import psycopg2
print('✅ Psycopg2 installed')

from airflow.providers.amazon.aws.hooks.sqs import SqsHook
print('✅ AWS providers installed')

try:
    import bme680
    print('✅ BME680 sensor library available')
except ImportError:
    print('ℹ️ BME680 not installed (optional)')

print('🎉 All dependencies verified!')
"

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import airflow, mlflow, boto3; print('healthy')"

# ============================================
# Metadata
# ============================================
ENV AIRFLOW_VERSION=${AIRFLOW_VERSION}
ENV IMAGE_BUILD_DATE=${BUILD_DATE}
ENV IMAGE_VCS_REF=${VCS_REF}
