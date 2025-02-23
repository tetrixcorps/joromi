# Create Kubernetes manifests using the kubernetes_manifest resource
resource "kubernetes_manifest" "ml_namespace" {
  manifest = {
    apiVersion = "v1"
    kind       = "Namespace"
    metadata = {
      name = "${var.environment}-ml"
      labels = {
        name = "${var.environment}-ml"
      }
    }
  }
}

# Create ConfigMap for service configuration
resource "kubernetes_config_map" "ml_config" {
  metadata {
    name      = "ml-service-config"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  data = {
    "service-config.yaml" = <<-EOT
      api:
        port: 8000
        timeout: 30s
        max_request_size: "10MB"
      
      model_servers:
        text:
          protocol: "REST"
          port: 8001
          max_batch_size: 32
        vision:
          protocol: "REST"
          port: 8002
          max_batch_size: 16
        speech:
          protocol: "WebSocket"
          port: 8003
          max_audio_duration: "30s"
      
      messaging:
        protocol: "WebSocket"
        port: 8004
        heartbeat_interval: "5s"
      
      monitoring:
        metrics_port: 9090
        health_check_interval: "10s"
    EOT
  }
}

# Create Service for API Gateway
resource "kubernetes_service" "api_gateway" {
  metadata {
    name      = "api-gateway"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  spec {
    selector = {
      app = "api-gateway"
    }
    
    port {
      port        = 80
      target_port = 8000
    }

    type = "LoadBalancer"
  }
}

# Create Deployments for each service
resource "kubernetes_deployment" "api_gateway" {
  metadata {
    name      = "api-gateway"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "api-gateway"
      }
    }

    template {
      metadata {
        labels = {
          app = "api-gateway"
        }
      }

      spec {
        container {
          name  = "api-gateway"
          image = "your-registry/api-gateway:latest"

          env {
            name = "CONFIG_PATH"
            value = "/config/service-config.yaml"
          }

          volume_mount {
            name       = "config-volume"
            mount_path = "/config"
          }

          resources {
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
            requests = {
              cpu    = "250m"
              memory = "256Mi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds       = 10
          }
        }

        volume {
          name = "config-volume"
          config_map {
            name = kubernetes_config_map.ml_config.metadata[0].name
          }
        }
      }
    }
  }
}

# Create ML Model Service Deployment
resource "kubernetes_deployment" "ml_service" {
  metadata {
    name      = "ml-service"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "ml-service"
      }
    }

    template {
      metadata {
        labels = {
          app = "ml-service"
        }
      }

      spec {
        container {
          name  = "ml-service"
          image = "your-registry/ml-service:latest"

          env {
            name = "MODEL_PATH"
            value = "/models"
          }

          env {
            name  = "LOG_LEVEL"
            value = "INFO"
          }

          env {
            name  = "LOG_FORMAT"
            value = "json"
          }

          volume_mount {
            name       = "model-storage"
            mount_path = "/models"
          }

          volume_mount {
            name       = "log"
            mount_path = "/var/log/ml-service"
          }

          resources {
            limits = {
              cpu    = "2000m"
              memory = "4Gi"
              "nvidia.com/gpu" = 1
            }
            requests = {
              cpu    = "1000m"
              memory = "2Gi"
            }
          }

          port {
            container_port = 8001
          }
        }

        volume {
          name = "model-storage"
          persistent_volume_claim {
            claim_name = "model-storage-pvc"
          }
        }

        volume {
          name = "log"
          empty_dir {}
        }

        node_selector = {
          "nvidia.com/gpu" = "true"
        }
      }
    }
  }
}

# Create WebSocket Service for Real-time Communication
resource "kubernetes_deployment" "websocket_service" {
  metadata {
    name      = "websocket-service"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "websocket-service"
      }
    }

    template {
      metadata {
        labels = {
          app = "websocket-service"
        }
      }

      spec {
        container {
          name  = "websocket-service"
          image = "your-registry/websocket-service:latest"

          env {
            name = "WS_PORT"
            value = "8004"
          }

          resources {
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
            requests = {
              cpu    = "250m"
              memory = "256Mi"
            }
          }

          port {
            container_port = 8004
          }
        }
      }
    }
  }
}

# Create Monitoring Service
resource "kubernetes_deployment" "monitoring" {
  metadata {
    name      = "monitoring"
    namespace = kubernetes_manifest.ml_namespace.manifest.metadata.name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "monitoring"
      }
    }

    template {
      metadata {
        labels = {
          app = "monitoring"
        }
      }

      spec {
        container {
          name  = "prometheus"
          image = "prom/prometheus:latest"

          volume_mount {
            name       = "prometheus-config"
            mount_path = "/etc/prometheus"
          }

          port {
            container_port = 9090
          }
        }

        volume {
          name = "prometheus-config"
          config_map {
            name = "prometheus-config"
          }
        }
      }
    }
  }
} 