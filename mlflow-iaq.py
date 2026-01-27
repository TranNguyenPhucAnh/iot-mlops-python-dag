import mlflow
import os

# Đây là endpoint nội bộ của bạn
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.mlflow.svc.cluster.local:5000"

with mlflow.start_run():
    mlflow.log_param("sensor_type", "BME680")
    mlflow.log_metric("temperature", 28.5)
    # Thử log một file nhỏ để test S3
    with open("test.txt", "w") as f:
        f.write("Hello MLOps")
    mlflow.log_artifact("test.txt")
