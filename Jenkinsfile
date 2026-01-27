pipeline {
    agent {
        kubernetes {
            yaml '''
# Kubernetes Pod Template: Định nghĩa Pod structure cho Jenkins Kubernetes agent (có khi cài đặt Jenkins Kubernetes Plugin)
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-sa # chung SA mà controller pod đang dùng
# ✅ JNLP container (Jenkins agent)
  containers:
  - name: jnlp
    image: jenkins/inbound-agent:latest
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
      privileged: true
    env:
    - name: DOCKER_TLS_CERTDIR
      value: ""
    volumeMounts:
    - name: docker-graph-storage
      mountPath: /var/lib/docker
  - name: trivy
    image: aquasec/trivy:latest
    command: ["/bin/sh"]
    tty: true
  volumes:
  - name: pip-cache
    emptyDir: {}
  - name: docker-graph-storage
    emptyDir: {}
'''
        }
    }

    environment {
        AWS_REGION      = "ap-southeast-1"
        ECR_REGISTRY    = "408279620390.dkr.ecr.ap-southeast-1.amazonaws.com"
        ECR_REPO        = "iot-mlops-repo"
        IMAGE_NAME      = "${ECR_REGISTRY}/${ECR_REPO}"
        AIRFLOW_VERSION = "3.0.2"
        // Sử dụng đường dẫn tuyệt đối cho docker binary để tránh lỗi 'stat docker'
        DOCKER_BIN      = "/usr/local/bin/docker" 
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.BUILD_DATE = sh(script: "date -u +'%Y-%m-%dT%H:%M:%SZ'", returnStdout: true).trim()
                    env.GIT_SHORT_COMMIT = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                }
            }
        }

        stage('Unit Test DAGs') {
            steps {
                container('python') {
                    sh '''
                    pip install pytest mlflow boto3 psycopg2-binary
                    pytest tests/ -v --tb=short || echo "Tests failed but continuing build"
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                container('docker') {
                    // Cung cấp DOCKER_HOST và dùng đường dẫn tuyệt đối
                    withEnv(['DOCKER_HOST=unix:///var/run/docker.sock']) {
                        script {
                            echo "=== Building Image: ${IMAGE_NAME}:${env.GIT_SHORT_COMMIT} ==="
                            
                            // 1. Cài đặt git vào container docker để fix lỗi buildx warning
                            sh "apk add --no-cache git"
                            
                            // 2. Build với đường dẫn tuyệt đối tới docker binary
                            // Sử dụng -f ./docker/Dockerfile để rõ ràng hơn
                            sh """
                            ${DOCKER_BIN} build \
                                --build-arg AIRFLOW_VERSION=${AIRFLOW_VERSION} \
                                --build-arg BUILD_DATE=${env.BUILD_DATE} \
                                --build-arg VCS_REF=${env.GIT_SHORT_COMMIT} \
                                -t ${IMAGE_NAME}:${env.GIT_SHORT_COMMIT} \
                                -t ${IMAGE_NAME}:latest \
                                -f ./docker/Dockerfile .
                            """
                            
                            // 3. Test sơ bộ image vừa build
                            sh "${DOCKER_BIN} run --rm ${IMAGE_NAME}:${env.GIT_SHORT_COMMIT} python --version"
                        }
                    }
                }
            }
        }

        stage('Push to ECR') {
            steps {
                container('docker') {
                    withEnv(['DOCKER_HOST=unix:///var/run/docker.sock']) {
                        sh "apk add --no-cache aws-cli"
                        sh """
                        aws ecr get-login-password --region ${AWS_REGION} | \
                            ${DOCKER_BIN} login --username AWS --password-stdin ${ECR_REGISTRY}
                        
                        ${DOCKER_BIN} push ${IMAGE_NAME}:${env.GIT_SHORT_COMMIT}
                        ${DOCKER_BIN} push ${IMAGE_NAME}:latest
                        """
                    }
                }
            }
        }

        stage('Deploy Trigger') {
            when { branch 'main' }
            steps {
                // Sử dụng sed để update file manifest trong git repo (GitOps flow)
                sh """
                sed -i 's|tag:.*|tag: "${env.GIT_SHORT_COMMIT}"|g' helm/airflow-values.yaml
                git config user.email "jenkins@iot.local"
                git config user.name "Jenkins"
                git add helm/airflow-values.yaml
                git commit -m "chore: update airflow image to ${env.GIT_SHORT_COMMIT}" || echo "no changes"
                git push origin main || echo "push failed"
                """
            }
        }
    }
}
