terraform {
  backend "s3" {
    endpoint                    = "nyc3.digitaloceanspaces.com"
    region                     = "us-east-1"  # Required for compatibility
    bucket                     = "terraform-state-ml-infrastructure"
    key                        = "terraform.tfstate"
    skip_credentials_validation = true
    skip_metadata_api_check    = true
    force_path_style           = true
  }
} 