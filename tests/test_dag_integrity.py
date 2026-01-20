import pytest
from airflow.models import DagBag

def test_iot_sqs_to_s3_pipeline():
    dag_bag = DagBag(include_examples=False)
    dag_bag.process_file("iot_data_pipeline.py")  # ✅ Chỉ 1 file
    
    assert len(dag_bag.import_errors) == 0
    
    dag = dag_bag.get_dag("iot_sqs_to_s3_test")
    assert dag is not None
    
    # Test tasks
    tasks = dag.task_dict
    assert "write_test_to_s3" in tasks
