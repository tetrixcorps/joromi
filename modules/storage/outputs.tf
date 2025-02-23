output "bucket_endpoint" {
  value = digitalocean_spaces_bucket.model_storage.bucket_domain_name
}

output "cdn_endpoint" {
  value = digitalocean_cdn.model_storage_cdn.endpoint
} 