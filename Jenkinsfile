pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.11-slim
    command: ["cat"]
    tty: true
  - name: docker
    image: docker:24.0.5-dind
    securityContext:
      privileged: true
'''
        }
    }

    environment {
        AWS_REGION = "ap-southeast-1"
        ECR_REPO = "408279620390.dkr.ecr.ap-southeast-1.amazonaws.com/iot-bme680"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Unit Test DAGs') {
            steps {
                container('python') {
                    sh '''
                    pip install apache-airflow==3.0.2 # Hoặc version bạn đang dùng
                    pip install pytest
                    # Chạy pytest từ thư mục gốc, tránh quét trúng file pytest
                    export PYTHONPATH=$PYTHONPATH:.
                    # Lệnh test: Kiểm tra xem các file DAG có lỗi cú pháp hay import lỗi không
                    python -m pytest tests/test_dag_integrity.py
                    '''
                }
            }
        }

        stage('Build & Push Docker Image (Optional)') {
            when {
                expression { return env.BRANCH_NAME == 'main' }
            }
            steps {
                container('docker') {
                    script {
                        echo "Phần này sẽ dùng để build Airflow Worker custom nếu bạn thêm thư viện mới"
                        // sh "docker build -t ${ECR_REPO}:latest ."
                    }
                }
            }
        }
    }
}
