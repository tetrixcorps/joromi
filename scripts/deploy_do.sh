#!/bin/bash

# Configuration
DROPLET_NAME="ml-gpu-server"
DROPLET_SIZE="s-4vcpu-8gb-amd"  # Updated to match your GPU specs
REGION="nyc3"  # Updated to NYC3
SPACE_NAME="joromi"
DOCKER_IMAGE="model-server"
DROPLET_IP="162.243.13.112"  # Your existing droplet IP

# Spaces Configuration
export SPACES_ACCESS_KEY="DO00AXG8EXAA4RXJNWX3"
export SPACES_SECRET_KEY="dop_v1_959c69e49e737c7b89055263668465a316843d8401f688869e3b51dd205cd2bb"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment to Digital Ocean...${NC}"

# 1. Configure Spaces access
echo "Configuring Spaces access..."
cat > ~/.s3cfg << EOF
[default]
access_key = ${SPACES_ACCESS_KEY}
secret_key = ${SPACES_SECRET_KEY}
host_base = nyc3.digitaloceanspaces.com
host_bucket = %(bucket)s.nyc3.digitaloceanspaces.com
EOF

# 2. Install NVIDIA drivers and Docker
echo "Installing NVIDIA drivers and Docker..."
ssh -o StrictHostKeyChecking=no root@${DROPLET_IP} bash -c '
    # Update system
    apt-get update && apt-get upgrade -y

    # Install NVIDIA drivers
    apt-get install -y nvidia-driver-525
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
'

# 3. Copy application files
echo "Copying application files..."
rsync -avz --exclude 'venv' --exclude '__pycache__' \
    --exclude '.git' ./ root@${DROPLET_IP}:/root/app/

# 4. Update cache manager configuration
ssh root@${DROPLET_IP} "cat > /root/app/config/spaces_config.py" << EOF
SPACES_CONFIG = {
    'endpoint_url': 'https://nyc3.digitaloceanspaces.com',
    'access_key': '${SPACES_ACCESS_KEY}',
    'secret_key': '${SPACES_SECRET_KEY}',
    'space_name': '${SPACE_NAME}',
    'region': '${REGION}'
}
EOF

# 5. Build and run Docker container
echo "Building and running Docker container..."
ssh root@${DROPLET_IP} bash -c '
    cd /root/app
    
    # Create persistent volume for model cache
    docker volume create model-cache
    
    # Build Docker image
    docker build -t '$DOCKER_IMAGE' .
    
    # Stop existing container if running
    docker stop model-server || true
    docker rm model-server || true
    
    # Run new container with environment variables
    docker run -d \
        --name model-server \
        --gpus all \
        -p 8000:8000 \
        -v model-cache:/app/model_cache \
        -e SPACES_ACCESS_KEY='${SPACES_ACCESS_KEY}' \
        -e SPACES_SECRET_KEY='${SPACES_SECRET_KEY}' \
        -e SPACE_NAME='${SPACE_NAME}' \
        --restart unless-stopped \
        '$DOCKER_IMAGE'
'

# 6. Verify GPU access
echo "Verifying GPU access..."
ssh root@${DROPLET_IP} "docker exec model-server nvidia-smi"

# 7. Print deployment info
echo -e "${GREEN}Deployment completed!${NC}"
echo "Server IP: ${DROPLET_IP}"
echo "API endpoint: http://${DROPLET_IP}:8000"
echo "Monitor GPU usage: ssh root@${DROPLET_IP} 'nvidia-smi'"

# Create instructions file
cat << EOF > deployment_info.txt
Deployment Information:
----------------------
Server IP: ${DROPLET_IP}
API Endpoint: http://${DROPLET_IP}:8000
Space Name: ${SPACE_NAME}
Region: ${REGION}

Useful Commands:
---------------
1. SSH into server:
   ssh root@${DROPLET_IP}

2. View GPU status:
   nvidia-smi

3. View container logs:
   docker logs -f model-server

4. Enter container:
   docker exec -it model-server bash

5. Monitor metrics:
   http://${DROPLET_IP}:8000/metrics
EOF

echo -e "${GREEN}Deployment information saved to deployment_info.txt${NC}"

# Add monitoring section
echo "Starting deployment monitoring..."
python3 - << EOF
from utils.deployment_monitor import DeploymentMonitor
from utils.model_preloader import ModelPreloader
from models.model_loader import ModelManager
import asyncio
import torch

async def main():
    # Initialize monitor
    monitor = DeploymentMonitor(host="${DROPLET_IP}", port=8000)
    
    # Monitor deployment
    if not monitor.monitor_deployment():
        print("Deployment monitoring failed")
        exit(1)
    
    # Initialize model manager
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_manager = ModelManager(device)
    
    # Initialize and run preloader
    preloader = ModelPreloader(model_manager)
    await preloader.preload_models()
    
    # Check loading status
    status = preloader.get_loading_status()
    for model_name, loaded in status.items():
        print(f"Model {model_name}: {'Loaded' if loaded else 'Failed'}")

asyncio.run(main())
EOF

# Continue with deployment if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment and model preloading completed successfully!${NC}"
else
    echo -e "${RED}Deployment or model preloading failed!${NC}"
    exit 1
fi 