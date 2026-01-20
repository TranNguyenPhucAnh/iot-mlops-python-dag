import pytest
from unittest.mock import patch, MagicMock
from airflow.models import DagBag

def test_write_test_to_s3_execution(dagbag):
    dag = dagbag.get_dag("iot-data-s3-test")
    task = dag.get_task("write_test_to_s3")
    
    # Mock S3 client
    with patch('airflow.providers.amazon.aws.hooks.s3.S3Hook') as mock_s3:
        mock_hook = MagicMock()
        mock_s3.return_value = mock_hook
        
        # Dry-run task (không thực upload)
        result = task.execute({})
        
        # Verify S3 hook gọi đúng
        mock_s3.assert_called_once()
        mock_hook.load_string.assert_called() # Verify upload
