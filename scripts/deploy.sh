#!/bin/bash

# 1. Set up GPU droplet
doctl compute droplet create model-server \
    --image docker-20.10.21-focal-amd64 \
    --size gd-2vcpu-8gb \
    --region nyc1 \
    --ssh-keys your-ssh-key-id

# 2. Install NVIDIA drivers and Docker
ssh root@droplet-ip "bash -s" << 'EOF'
# Install NVIDIA drivers
apt-get update
apt-get install -y nvidia-driver-525

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker
EOF 