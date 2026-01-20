import os
import pytest
from airflow.models import DagBag

def test_iot_sqs_to_s3_pipeline():
    # ✅ Mock DB URI: in-memory SQLite (no file)
    os.environ['AIRFLOW__CORE__SQL_ALCHEMY_CONN'] = 'sqlite:///:memory:'
    
    try:
        dag_bag = DagBag(
            dag_folder=".", 
            include_examples=False,
            safe_mode=False  # Parse full operators
        )
        dag_bag.process_file("iot_data_pipeline.py")
        
        assert len(dag_bag.import_errors) == 0
        assert "iot_sqs_to_s3_test" in dag_bag.dag_ids
        
        dag = dag_bag.get_dag("iot_sqs_to_s3_test")
        tasks = dag.task_dict
        assert "write_test_to_s3" in tasks
        
    finally:
        # ✅ Cleanup env
        del os.environ['AIRFLOW__CORE__SQL_ALCHEMY_CONN']
