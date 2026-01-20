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
                    pip install -r requirements.txt \
                        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.12.txt"
                    pytest tests/test_dag_integrity.py -v
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
