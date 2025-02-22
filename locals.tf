locals {
  # Project naming and tagging
  name_prefix = "${var.environment}-ml"
  common_tags = {
    Environment = var.environment
    Project     = "ml-infrastructure"
    ManagedBy   = "terraform"
  }

  # Kubernetes configurations
  k8s_cluster_name = "${local.name_prefix}-cluster"
  node_pools = {
    general = {
      size       = var.node_pool_size
      node_count = 2
      auto_scale = true
      min_nodes  = 1
      max_nodes  = 3
    }
    gpu = {
      size       = var.gpu_droplet_size
      node_count = 1
      auto_scale = true
      min_nodes  = 1
      max_nodes  = 2
      labels = {
        service = "ml-workload"
        gpu     = "true"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }

  # Storage configurations
  spaces_bucket_name = "${local.name_prefix}-${var.space_name}"
  spaces_domain     = "${var.do_region}.digitaloceanspaces.com"

  # GPU Droplet configurations
  gpu_droplet_name = "${local.name_prefix}-gpu"
  gpu_droplet_tags = concat([var.environment], var.additional_tags)
} 