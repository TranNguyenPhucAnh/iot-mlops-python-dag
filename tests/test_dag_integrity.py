"""
DAG Integrity Tests
Run: pytest tests/test_dag_integrity.py -v
"""

import pytest
from airflow.models import DagBag
import os

DAG_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'dags')

@pytest.fixture(scope="session")
def dagbag():
    """Load all DAGs"""
    return DagBag(dag_folder=DAG_FOLDER, include_examples=False)

def test_no_import_errors(dagbag):
    """Test that all DAGs load without errors"""
    assert not dagbag.import_errors, \
        f"DAG import errors: {dagbag.import_errors}"

def test_required_dags_exist(dagbag):
    """Test that required DAGs are present"""
    required_dags = ["hello_world_dag", "debug_s3_write_only", "iot_sqs_to_s3_test", "mlflow_connection_test_dag", "iot_bme680_ingestion_pipeline_v3"]
    
    for dag_id in required_dags:
        assert dag_id in dagbag.dags, \
            f"Required DAG '{dag_id}' not found"

def test_dag_tags(dagbag):
    """Test that DAGs have appropriate tags"""
    for dag_id, dag in dagbag.dags.items():
        assert dag.tags, \
            f"DAG '{dag_id}' should have tags"

def test_dag_owners(dagbag):
    """Test that DAGs have owners"""
    for dag_id, dag in dagbag.dags.items():
        assert dag.default_args.get('owner'), \
            f"DAG '{dag_id}' should have an owner"

def test_no_duplicate_dag_ids(dagbag):
    """Test that there are no duplicate DAG IDs"""
    dag_ids = [dag.dag_id for dag in dagbag.dags.values()]
    assert len(dag_ids) == len(set(dag_ids)), \
        "Duplicate DAG IDs found"

def test_task_count(dagbag):
    """Test that DAGs have tasks"""
    for dag_id, dag in dagbag.dags.items():
        assert len(dag.tasks) > 0, \
            f"DAG '{dag_id}' has no tasks"

@pytest.mark.parametrize("dag_id", ["hello_world_dag", "debug_s3_write_only", "iot_sqs_to_s3_test", "mlflow_connection_test_dag", "iot_bme680_ingestion_pipeline_v3"])
def test_specific_dag_structure(dagbag, dag_id):
    """Test specific DAG structure"""
    dag = dagbag.get_dag(dag_id)
    assert dag is not None
    
    # Test has tasks
    assert len(dag.tasks) > 0
    
    # Test has schedule
    assert dag.schedule_interval is not None or dag.timetable is not None
    
    # Test catchup is disabled for real-time pipelines
    if 'iot' in dag_id:
        assert dag.catchup == False
