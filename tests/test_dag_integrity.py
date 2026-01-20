import pytest
from airflow.models import DagBag

def test_iot_sqs_to_s3_pipeline():
    # ✅ No DB connection
    dag_bag = DagBag(
        dag_folder=".", 
        include_examples=False,
        database=None,  # Skip DB
        load_ops_from_past=True  # Parse operators
    )
    
    # Chỉ load file cụ thể
    dag_bag.process_file("iot_data_pipeline.py")
    
    assert len(dag_bag.import_errors) == 0
    
    dag = dag_bag.get_dag("iot_sqs_to_s3_test")
    assert dag is not None
    
    tasks = dag.task_dict
    assert "write_test_to_s3" in tasks
