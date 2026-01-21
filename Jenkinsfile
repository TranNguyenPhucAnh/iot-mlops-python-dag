pipeline {
    agent {
        kubernetes {
            yaml '''
# Kubernetes Pod Template: Định nghĩa Pod structure cho Jenkins Kubernetes agent (có khi cài đặt Jenkins Kubernetes Plugin)
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-sa
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
    args:
      - --host=unix:///var/run/docker.sock
    // volumeMounts:
    // - name: docker-sock  # Optional: host Docker
    //   mountPath: /var/run/docker.sock
    volumeMounts:
      - name: docker-graph-storage
        mountPath: /var/lib/docker
  volumes:
  - name: pip-cache
    emptyDir: {}
  - name: docker-graph-storage
    emptyDir: {}
  #
  // - name: docker-sock
  //   hostPath:
  //     path: /var/run/docker.sock
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

    stage('Build & Push Docker Image') {
            steps {
                container('docker') {
                    script {
                        sh "aws sts get-caller-identity" // Lệnh này sẽ in ra Role mà Pod đang thực sự dùng
                        // 1. Cài đặt AWS CLI nhanh để thực hiện Login (nếu image docker:dind chưa có)
                        sh "apk add --no-cache aws-cli"

                        // 2. Login vào AWS ECR dùng IAM Role (IRSA) đã gắn cho jenkins-sa
                        sh "aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}"

                        // 3. Build image với tag là ID của commit để dễ truy vết (Rollback khi cần)
                        def imageTag = "commit-${env.GIT_COMMIT.take(7)}"
                        echo "Đang build image: ${ECR_REPO}:${imageTag}"
                        
                        sh "docker build -t ${ECR_REPO}:${imageTag} -t ${ECR_REPO}:latest ."

                        // 4. Push lên ECR
                        sh "docker push ${ECR_REPO}:${imageTag}"
                        sh "docker push ${ECR_REPO}:latest"
                        
                        echo "Đã push thành công image lên ECR: ${ECR_REPO}:${imageTag}"
                    }
                }
            }
        }
    }
}
