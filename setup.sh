#!/bin/bash

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo "doctl is not installed. Please install it first."
    exit 1
fi

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "terraform is not installed. Please install it first."
    exit 1
fi

# Get DO token
echo "Please enter your DigitalOcean API token:"
read -s DO_TOKEN
echo

# Authenticate with DigitalOcean
doctl auth init -t "$DO_TOKEN"

# Get SSH key fingerprints
SSH_KEYS=$(doctl compute ssh-key list --format FingerPrint --no-header)

if [ -z "$SSH_KEYS" ]; then
    # No existing SSH keys found, check for local key
    if [ -f ~/.ssh/id_rsa.pub ]; then
        echo "No SSH keys found in DigitalOcean, using local SSH key."
        cat > terraform.tfvars << EOF
do_token = "$DO_TOKEN"
ssh_key_path = "~/.ssh/id_rsa.pub"
environment = "dev"
space_name = "models"
EOF
    else
        echo "Error: No SSH keys found in DigitalOcean and no local SSH key found."
        echo "Please either add SSH keys to DigitalOcean or create a local SSH key pair."
        exit 1
    fi
else
    # Use existing DigitalOcean SSH keys
    cat > terraform.tfvars << EOF
do_token = "$DO_TOKEN"
ssh_keys = [$(echo $SSH_KEYS | sed 's/ /,/g' | sed 's/[^,]*/"&"/g')]
environment = "dev"
space_name = "models"
EOF
fi

# Initialize Terraform
terraform init

echo "Setup complete! You can now run 'terraform plan' to see the execution plan."

# Install dependencies
pip install -r requirements.txt

# Setup Git LFS
git lfs install

# Download models
python scripts/model_manager.py

# Create necessary directories
mkdir -p logs
mkdir -p model_cache
mkdir -p frontend/data

# Set up environment variables
cp .env.example .env 