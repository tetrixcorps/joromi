resource "digitalocean_kubernetes_cluster" "cluster" {
  name    = var.cluster_name
  region  = var.region
  version = var.kubernetes_version

  node_pool {
    name       = "${var.environment}-node-pool"
    size       = var.node_pool_size
    auto_scale = true
    min_nodes  = 1
    max_nodes  = 3
  }

  tags = [var.environment]
}

# GPU node pool for ML workloads
resource "digitalocean_kubernetes_node_pool" "gpu_nodes" {
  cluster_id = digitalocean_kubernetes_cluster.cluster.id
  name       = "${var.environment}-gpu-pool"
  size       = "g-2vcpu-8gb"  # GPU instance
  auto_scale = true
  min_nodes  = 1
  max_nodes  = 2

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NoSchedule"
  }

  labels = {
    service  = "ml-workload"
    gpu      = "true"
  }
} 