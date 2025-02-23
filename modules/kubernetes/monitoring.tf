# Create monitoring namespace
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "${var.environment}-monitoring"
    labels = {
      name = "${var.environment}-monitoring"
    }
  }
}

# Elasticsearch StatefulSet
resource "kubernetes_stateful_set" "elasticsearch" {
  metadata {
    name      = "elasticsearch"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  spec {
    service_name = "elasticsearch"
    replicas     = 1

    selector {
      match_labels = {
        app = "elasticsearch"
      }
    }

    template {
      metadata {
        labels = {
          app = "elasticsearch"
        }
      }

      spec {
        container {
          name  = "elasticsearch"
          image = "docker.elastic.co/elasticsearch/elasticsearch:8.10.2"

          env {
            name  = "discovery.type"
            value = "single-node"
          }

          env {
            name  = "ES_JAVA_OPTS"
            value = "-Xms512m -Xmx512m"
          }

          resources {
            limits = {
              cpu    = "1000m"
              memory = "2Gi"
            }
            requests = {
              cpu    = "500m"
              memory = "1Gi"
            }
          }

          port {
            container_port = 9200
            name          = "http"
          }

          volume_mount {
            name       = "elasticsearch-data"
            mount_path = "/usr/share/elasticsearch/data"
          }
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "elasticsearch-data"
      }
      spec {
        access_modes = ["ReadWriteOnce"]
        resources {
          requests = {
            storage = "10Gi"
          }
        }
      }
    }
  }
}

# Kibana Deployment
resource "kubernetes_deployment" "kibana" {
  metadata {
    name      = "kibana"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "kibana"
      }
    }

    template {
      metadata {
        labels = {
          app = "kibana"
        }
      }

      spec {
        container {
          name  = "kibana"
          image = "docker.elastic.co/kibana/kibana:8.10.2"

          env {
            name  = "ELASTICSEARCH_HOSTS"
            value = "http://elasticsearch:9200"
          }

          resources {
            limits = {
              cpu    = "500m"
              memory = "1Gi"
            }
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
          }

          port {
            container_port = 5601
          }
        }
      }
    }
  }
}

# Logstash ConfigMap
resource "kubernetes_config_map" "logstash_config" {
  metadata {
    name      = "logstash-config"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "logstash.conf" = <<-EOT
      input {
        beats {
          port => 5044
        }
      }
      
      filter {
        grok {
          # Use %% for escaping % in Terraform string templates
          match => { "message" => "%%{TIMESTAMP_ISO8601:timestamp} %%{LOGLEVEL:log_level} %%{GREEDYDATA:message}" }
        }
        
        date {
          match => [ "timestamp", "ISO8601" ]
        }
      }
      
      output {
        elasticsearch {
          hosts => ["elasticsearch-master:9200"]
          index => "logstash-%%{+YYYY.MM.dd}"
        }
      }
    EOT
  }
}

# Logstash Deployment
resource "kubernetes_deployment" "logstash" {
  metadata {
    name      = "logstash"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "logstash"
      }
    }

    template {
      metadata {
        labels = {
          app = "logstash"
        }
      }

      spec {
        container {
          name  = "logstash"
          image = "docker.elastic.co/logstash/logstash:8.10.2"

          resources {
            limits = {
              cpu    = "500m"
              memory = "1Gi"
            }
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
          }

          port {
            container_port = 5044
            name          = "beats"
          }

          volume_mount {
            name       = "config"
            mount_path = "/usr/share/logstash/config"
          }

          volume_mount {
            name       = "pipeline"
            mount_path = "/usr/share/logstash/pipeline"
          }
        }

        volume {
          name = "config"
          config_map {
            name = kubernetes_config_map.logstash_config.metadata[0].name
            items {
              key  = "logstash.conf"
              path = "logstash.conf"
            }
          }
        }

        volume {
          name = "pipeline"
          config_map {
            name = kubernetes_config_map.logstash_config.metadata[0].name
            items {
              key  = "logstash.conf"
              path = "logstash.conf"
            }
          }
        }
      }
    }
  }
}

# Filebeat DaemonSet for log collection
resource "kubernetes_daemon_set" "filebeat" {
  metadata {
    name      = "filebeat"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  spec {
    selector {
      match_labels = {
        app = "filebeat"
      }
    }

    template {
      metadata {
        labels = {
          app = "filebeat"
        }
      }

      spec {
        container {
          name  = "filebeat"
          image = "docker.elastic.co/beats/filebeat:8.10.2"

          args = [
            "-c", "/etc/filebeat.yml",
            "-e",
          ]

          env {
            name = "NODE_NAME"
            value_from {
              field_ref {
                field_path = "spec.nodeName"
              }
            }
          }

          resources {
            limits = {
              cpu    = "200m"
              memory = "256Mi"
            }
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
          }

          volume_mount {
            name       = "config"
            mount_path = "/etc/filebeat.yml"
            sub_path   = "filebeat.yml"
          }

          volume_mount {
            name       = "docker-sock"
            mount_path = "/var/run/docker.sock"
          }

          volume_mount {
            name       = "varlog"
            mount_path = "/var/log"
          }
        }

        volume {
          name = "config"
          config_map {
            name = "filebeat-config"
          }
        }

        volume {
          name = "docker-sock"
          host_path {
            path = "/var/run/docker.sock"
          }
        }

        volume {
          name = "varlog"
          host_path {
            path = "/var/log"
          }
        }
      }
    }
  }
}

# Grafana Deployment
resource "kubernetes_deployment" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "grafana"
      }
    }

    template {
      metadata {
        labels = {
          app = "grafana"
        }
      }

      spec {
        container {
          name  = "grafana"
          image = "grafana/grafana:latest"

          env {
            name  = "GF_SECURITY_ADMIN_PASSWORD"
            value = var.grafana_admin_password
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
            container_port = 3000
          }

          volume_mount {
            name       = "grafana-storage"
            mount_path = "/var/lib/grafana"
          }
        }

        volume {
          name = "grafana-storage"
          persistent_volume_claim {
            claim_name = "grafana-pvc"
          }
        }
      }
    }
  }
} 