# Test namespace
resource "kubernetes_namespace" "testing" {
  metadata {
    name = "${var.environment}-testing"
    labels = {
      name = "${var.environment}-testing"
    }
  }
}

# Test ConfigMap for test configurations
resource "kubernetes_config_map" "test_config" {
  metadata {
    name      = "test-config"
    namespace = kubernetes_namespace.testing.metadata[0].name
  }

  data = {
    "test-cases.yaml" = <<-EOT
      banking_queries:
        - query: "How do I block my credit card?"
          expected_type: "CARD_SERVICES"
        - query: "What's my account balance?"
          expected_type: "ACCOUNT_INQUIRY"
        - query: "I noticed suspicious activity"
          expected_type: "SECURITY"

      speech_recognition:
        - audio_file: "test-audio-1.wav"
          expected_text: "Hello, this is a test"
        - audio_file: "test-audio-2.wav"
          expected_text: "Testing speech recognition"

      text_generation:
        - prompt: "Explain quantum computing"
          max_length: 100
          temperature: 0.7
        - prompt: "Write a story about AI"
          max_length: 200
          temperature: 0.8
    EOT
  }
}

# Test Job for ML Services
resource "kubernetes_cron_job" "integration_tests" {
  metadata {
    name      = "ml-integration-tests"
    namespace = kubernetes_namespace.testing.metadata[0].name
  }

  spec {
    schedule = "0 */6 * * *"  # Run every 6 hours
    concurrency_policy = "Replace"

    job_template {
      metadata {
        labels = {
          app = "ml-integration-tests"
        }
      }

      spec {
        template {
          metadata {
            labels = {
              app = "ml-integration-tests"
            }
          }

          spec {
            container {
              name  = "test-runner"
              image = "${var.image_registry}/ml-test-runner:${var.image_versions.test_runner}"

              env {
                name  = "TEST_CONFIG_PATH"
                value = "/config/test-cases.yaml"
              }

              env {
                name  = "LOG_LEVEL"
                value = "INFO"
              }

              volume_mount {
                name       = "test-config"
                mount_path = "/config"
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
            }

            volume {
              name = "test-config"
              config_map {
                name = kubernetes_config_map.test_config.metadata[0].name
              }
            }

            restart_policy = "Never"
          }
        }
      }
    }
  }
} 