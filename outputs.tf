output "kubernetes_cluster_id" {
  value       = module.kubernetes.cluster_id
  description = "ID of the Kubernetes cluster"
}

output "kubeconfig" {
  value       = module.kubernetes.kubeconfig
  sensitive   = true
  description = "Kubernetes configuration file"
}

output "spaces_bucket_endpoint" {
  value       = module.storage.bucket_endpoint
  description = "Endpoint for the Spaces bucket"
}

output "gpu_droplet_ip" {
  value       = module.gpu_droplet.droplet_ip
  description = "IP address of the GPU droplet"
} 