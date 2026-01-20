import pytest
import ast

def test_dag_syntax():    
    with open("iot_data_pipeline.py", "r") as f:
        tree = ast.parse(f.read())
    
    # Check DAG imports, task definitions
    assert any(node.id == "DAG" for node in ast.walk(tree))
    print("DAG syntax OK")
