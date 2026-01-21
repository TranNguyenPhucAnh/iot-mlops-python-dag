FROM apache/airflow:3.0.2
USER root
RUN apt-get update && apt-get install -y gcc python3-dev
USER airflow
RUN pip install bme680 smbus2
