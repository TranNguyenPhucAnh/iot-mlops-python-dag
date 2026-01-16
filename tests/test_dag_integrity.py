import pytest
from airflow.models import DagBag

def test_dag_loaded_with_no_errors():
    dag_bag = DagBag(dag_folder=".", include_examples=False)
    # Kiểm tra xem có lỗi import không (ví dụ lỗi schedule_interval bạn gặp hôm trước)
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"

def test_dag_ids_present():
    dag_bag = DagBag(dag_folder=".", include_examples=False)
    # Kiểm tra xem có đúng cái DAG iot_sqs_to_s3_test không
    assert "iot_sqs_to_s3_test" in dag_bag.dag_ids
