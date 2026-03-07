FROM argoprojlabs/argocd-image-updater:latest
USER root

# aws-cli từ Alpine community repo — không cần curl/unzip, không bị conflict
RUN apk add --no-cache aws-cli \
    && aws --version

RUN ln -sf /usr/local/bin/argocd-image-updater /manager \
    && chmod 755 /manager

USER 1001
