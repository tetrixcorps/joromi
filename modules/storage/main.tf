resource "digitalocean_spaces_bucket" "model_storage" {
  name   = var.space_name
  region = var.region
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    enabled = true

    noncurrent_version_expiration {
      days = 30
    }
  }
}

# CDN for the Spaces bucket
resource "digitalocean_cdn" "model_storage_cdn" {
  origin = digitalocean_spaces_bucket.model_storage.bucket_domain_name
  ttl    = 3600
}

# Create Spaces access keys
resource "digitalocean_spaces_bucket_policy" "model_storage" {
  bucket = digitalocean_spaces_bucket.model_storage.name
  region = digitalocean_spaces_bucket.model_storage.region

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = [
          "${digitalocean_spaces_bucket.model_storage.urn}/*"
        ]
      }
    ]
  })
} 