resource "digitalocean_droplet" "gpu" {
  name     = var.name
  region   = var.region
  size     = var.size
  image    = "ubuntu-22-04-x64"
  ssh_keys = local.ssh_key_ids

  user_data = <<-EOF
              #!/bin/bash
              # Update and install basic dependencies
              apt-get update && apt-get install -y \
                nvidia-driver-525 \
                python3-pip \
                python3-dev \
                build-essential \
                git

              # Install Docker
              curl -fsSL https://get.docker.com -o get-docker.sh
              sh get-docker.sh

              # Install NVIDIA Container Toolkit
              distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
              curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
              curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
              apt-get update && apt-get install -y nvidia-container-toolkit
              systemctl restart docker

              # Install Python ML dependencies
              pip3 install --no-cache-dir \
                torch \
                torchvision \
                torchaudio \
                transformers \
                decord \
                librosa \
                numpy \
                pillow \
                requests

              # Create ML workspace directory
              mkdir -p /opt/ml
              chmod 777 /opt/ml
              EOF

  tags = [var.environment]
}

# Firewall rules for GPU droplet
resource "digitalocean_firewall" "gpu" {
  name = "${var.name}-firewall"

  droplet_ids = [digitalocean_droplet.gpu.id]

  inbound_rule {
    protocol         = "tcp"
    port_range      = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range      = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range      = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range           = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
} 