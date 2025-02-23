output "cluster_id" {
  value = digitalocean_kubernetes_cluster.cluster.id
}

output "kubeconfig" {
  value = digitalocean_kubernetes_cluster.cluster.kube_config[0].raw_config
}

output "endpoint" {
  value = digitalocean_kubernetes_cluster.cluster.endpoint
} 