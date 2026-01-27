FROM apache/airflow:slim-latest

USER root

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Verify installations
RUN python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')" && \
    python -c "import boto3; print(f'Boto3 version: {boto3.__version__}')"
