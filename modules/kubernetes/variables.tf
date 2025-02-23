variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
}

variable "node_pool_size" {
  description = "Size of the node pool"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "image_registry" {
  description = "Container registry for service images"
  type        = string
}

variable "image_versions" {
  description = "Versions for service images"
  type = object({
    api_gateway = string
    ml_service  = string
    websocket   = string
  })
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
}

variable "elasticsearch_storage_size" {
  description = "Storage size for Elasticsearch"
  type        = string
  default     = "10Gi"
} 