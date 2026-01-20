pipeline {
    agent {
        kubernetes {
            yaml '''
# Kubernetes Pod Template: Định nghĩa Pod structure cho Jenkins Kubernetes agent (có khi cài đặt Jenkins Kubernetes Plugin)
apiVersion: v1
kind: Pod
spec:
  containers:
  # ✅ JNLP container (Jenkins agent)
  - name: jnlp
    image: jenkins/inbound-agent:latest
    tty: true
  # ✅ Python container cho pytest
  - name: python
    image: python:3.12-slim
    command: ["/bin/sh"]
    tty: true
    volumeMounts:
    - name: pip-cache
      mountPath: /root/.cache/pip
  # ✅ Docker DIND chạy command docker build, docker push trong pipeline (privileged đúng cách)
  - name: docker
    image: docker:24.0.5-dind
    securityContext:
      privileged: true  # ✅ Container chạy FULL Docker daemon
    tty: true
    volumeMounts:
    - name: docker-sock  # Optional: host Docker
      mountPath: /var/run/docker.sock
  volumes:
  - name: pip-cache
    emptyDir: {}
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
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
