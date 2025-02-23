resource "aws_instance" "gpu" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id
  key_name      = var.key_name

  root_block_device {
    volume_size = 100  # Large storage for models
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              # Install CUDA and GPU drivers
              apt-get update && apt-get install -y nvidia-driver-525
              # Install Docker
              curl -fsSL https://get.docker.com -o get-docker.sh
              sh get-docker.sh
              # Install Python dependencies
              apt-get install -y python3-pip
              pip3 install torch torchvision torchaudio
              EOF

  tags = {
    Name        = "${var.environment}-gpu-instance"
    Environment = var.environment
  }
}

# Security group for GPU instance
resource "aws_security_group" "gpu_instance" {
  name        = "${var.environment}-gpu-instance-sg"
  description = "Security group for GPU instance"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
} 