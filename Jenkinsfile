// Jenkinsfile - Optimized Airflow MLflow Image Build
pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-sa  # chung SA mà controller pod đang dùng
# ✅ JNLP container (Jenkins agent)
  containers:
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
      privileged: true
    tty: true
    args:
      - --host=unix:///var/run/docker.sock
    volumeMounts:
      - name: docker-graph-storage
        mountPath: /var/lib/docker
  
  volumes:
  - name: pip-cache
    emptyDir: {}
  - name: docker-graph-storage
    emptyDir: {}
'''
        }
    }
    
    environment {
        AWS_REGION = "ap-southeast-1"
        ECR_REGISTRY = "408279620390.dkr.ecr.ap-southeast-1.amazonaws.com"
        ECR_REPO = "airflow-mlflow"
        IMAGE_NAME = "${ECR_REGISTRY}/${ECR_REPO}"
        
        // Versioning
        AIRFLOW_VERSION = "3.0.2"
        BUILD_DATE = sh(script: "date -u +'%Y-%m-%dT%H:%M:%SZ'", returnStdout: true).trim()
        GIT_SHORT_COMMIT = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    // Store commit info for tagging
                    env.GIT_COMMIT_MSG = sh(
                        script: "git log -1 --pretty=%B",
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Validate Dependencies') {
            steps {
                container('python') {
                    sh '''
                    echo "=== Validating requirements.txt ==="
                    cat requirements.txt
                    
                    echo ""
                    echo "=== Checking for security vulnerabilities ==="
                    pip install safety
                    safety check --file requirements.txt || echo "⚠️ Found vulnerabilities (non-blocking)"
                    '''
                }
            }
        }
        
        stage('Unit Test DAGs') {
            steps {
                container('python') {
                    sh '''
                    echo "=== Installing dependencies ==="
                    pip install -r requirements.txt \
                        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"
                    
                    echo ""
                    echo "=== Running DAG integrity tests ==="
                    pytest tests/ -v --tb=short
                    '''
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                container('docker') {
                    script {
                        echo "=== Building Docker image ==="
                        
                        // Image tags
                        def tags = [
                            "${IMAGE_NAME}:${GIT_SHORT_COMMIT}",
                            "${IMAGE_NAME}:${AIRFLOW_VERSION}-${GIT_SHORT_COMMIT}",
                            "${IMAGE_NAME}:latest"
                        ]
                        
                        // Build with cache and labels
                        sh """
                        docker build \
                            --build-arg AIRFLOW_VERSION=${AIRFLOW_VERSION} \
                            --build-arg BUILD_DATE='${BUILD_DATE}' \
                            --build-arg VCS_REF=${GIT_SHORT_COMMIT} \
                            --label "org.opencontainers.image.created=${BUILD_DATE}" \
                            --label "org.opencontainers.image.version=${AIRFLOW_VERSION}-${GIT_SHORT_COMMIT}" \
                            --label "org.opencontainers.image.revision=${env.GIT_COMMIT}" \
                            --label "org.opencontainers.image.source=${env.GIT_URL}" \
                            ${tags.collect { "-t $it" }.join(' ')} \
                            -f docker/Dockerfile \
                            .
                        """
                        
                        // Test image
                        echo "=== Testing built image ==="
                        sh """
                        docker run --rm ${IMAGE_NAME}:${GIT_SHORT_COMMIT} \
                            python -c "
import sys
import mlflow
import boto3
import psycopg2
from airflow.providers.amazon.aws.hooks.sqs import SqsHook

print('✅ Python version:', sys.version)
print('✅ MLflow version:', mlflow.__version__)
print('✅ Boto3 version:', boto3.__version__)
print('✅ Psycopg2 installed')
print('✅ Airflow AWS providers installed')
print('🎉 All dependencies validated!')
"
                        """
                        
                        // Store tags for next stages
                        env.IMAGE_TAGS = tags.join(' ')
                    }
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                container('trivy') {
                    script {
                        echo "=== Scanning image for vulnerabilities ==="
                        
                        // Scan with Trivy
                        sh """
                        trivy image \
                            --severity HIGH,CRITICAL \
                            --exit-code 0 \
                            --no-progress \
                            --format table \
                            ${IMAGE_NAME}:${GIT_SHORT_COMMIT}
                        """
                        
                        // Generate JSON report
                        sh """
                        trivy image \
                            --severity HIGH,CRITICAL \
                            --format json \
                            --output trivy-report.json \
                            ${IMAGE_NAME}:${GIT_SHORT_COMMIT}
                        """
                        
                        // Archive report
                        archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
                    }
                }
            }
        }
        
        stage('Push to ECR') {
            steps {
                container('docker') {
                    script {
                        echo "=== Logging into AWS ECR ==="
                        
                        // Install AWS CLI
                        sh "apk add --no-cache aws-cli"
                        
                        // Verify IAM role
                        sh """
                        echo '=== Verifying AWS IAM Role ==='
                        aws sts get-caller-identity
                        """
                        
                        // ECR login
                        sh """
                        aws ecr get-login-password --region ${AWS_REGION} | \
                            docker login --username AWS --password-stdin ${ECR_REGISTRY}
                        """
                        
                        // Push all tags
                        echo "=== Pushing images to ECR ==="
                        env.IMAGE_TAGS.split(' ').each { tag ->
                            echo "Pushing: ${tag}"
                            sh "docker push ${tag}"
                        }
                        
                        echo """
=== ✅ Build & Push Successful ===
Images pushed:
${env.IMAGE_TAGS.split(' ').collect { "  - $it" }.join('\n')}

Latest image:
  ${IMAGE_NAME}:latest
  
Commit-tagged image:
  ${IMAGE_NAME}:${GIT_SHORT_COMMIT}
  
Versioned image:
  ${IMAGE_NAME}:${AIRFLOW_VERSION}-${GIT_SHORT_COMMIT}
"""
                    }
                }
            }
        }
        
        stage('Update Helm Values') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo "=== Updating Helm values with new image ==="
                    
                    // Update values file with new image tag
                    sh """
                    sed -i 's|tag:.*|tag: "${GIT_SHORT_COMMIT}"|g' helm/airflow-values.yaml
                    
                    git config user.email "jenkins@iot-platform.local"
                    git config user.name "Jenkins CI"
                    git add helm/airflow-values.yaml
                    git commit -m "chore: Update Airflow image to ${GIT_SHORT_COMMIT}" || echo "No changes"
                    git push origin main || echo "Push skipped"
                    """
                }
            }
        }
        
        stage('Trigger Airflow Deployment') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo "=== Triggering ArgoCD sync (optional) ==="
                    
                    // Option 1: ArgoCD CLI
                    sh """
                    # Install ArgoCD CLI if needed
                    # argocd app sync airflow --force
                    echo "ArgoCD sync would trigger here"
                    """
                    
                    // Option 2: kubectl rollout restart
                    sh """
                    kubectl set image deployment/airflow-worker \
                        airflow=${IMAGE_NAME}:${GIT_SHORT_COMMIT} \
                        -n airflow || echo "Manual deployment needed"
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo """
🎉 ============================================
   BUILD SUCCESS
============================================
Branch:  ${env.BRANCH_NAME}
Commit:  ${GIT_SHORT_COMMIT}
Message: ${env.GIT_COMMIT_MSG}

Image: ${IMAGE_NAME}:${GIT_SHORT_COMMIT}

Next steps:
1. Verify image in ECR console
2. Update Helm values if needed
3. Deploy to Airflow cluster

============================================
"""
        }
        
        failure {
            echo """
❌ ============================================
   BUILD FAILED
============================================
Branch:  ${env.BRANCH_NAME}
Commit:  ${GIT_SHORT_COMMIT}

Check logs above for details.
============================================
"""
        }
        
        always {
            // Cleanup
            container('docker') {
                sh '''
                echo "=== Cleaning up Docker images ==="
                docker image prune -f || true
                '''
            }
        }
    }
}
