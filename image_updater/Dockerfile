FROM argoprojlabs/argocd-image-updater:latest

USER root

# AWS CLI static (no pip)
RUN apk add --no-cache curl unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && aws --version \
    && rm -rf aws awscliv2.zip

# Tạo symlink /manager → binary thật
RUN ln -sf /usr/local/bin/argocd-image-updater /manager \
    && chmod 755 /manager

USER 1001
