from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    return "Hello từ hệ thống IoT MLOps của Phúc Anh"

with DAG(
    dag_id='hello_world_dag',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:

    task_hello = PythonOperator(
        task_id='print_hello_task',
        python_callable=print_hello
    )
