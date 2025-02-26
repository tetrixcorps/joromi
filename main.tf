terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Kubernetes Cluster
module "kubernetes" {
  source = "./modules/kubernetes"
  
  cluster_name       = local.k8s_cluster_name
  region            = var.do_region
  kubernetes_version = var.kubernetes_version
  node_pools        = local.node_pools
  environment       = var.environment
  tags              = local.common_tags
}

# GPU Droplet for Model Inference
module "gpu_droplet" {
  source = "./modules/gpu_droplet"
  
  name        = local.gpu_droplet_name
  region      = var.do_region
  size        = var.gpu_droplet_size
  ssh_keys    = var.ssh_keys
  environment = var.environment
  tags        = local.gpu_droplet_tags
}

# Spaces (Object Storage)
module "storage" {
  source = "./modules/storage"
  
  space_name  = local.spaces_bucket_name
  region      = var.do_region
  environment = var.environment
  tags        = local.common_tags
}

# VPC and Network Configuration
module "vpc" {
  source = "./modules/vpc"
  
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  environment         = var.environment
}

# API Gateway and Lambda (if needed)
module "api" {
  source = "./modules/api"
  
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnets
} 