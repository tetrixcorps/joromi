variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "do_region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc1"
  validation {
    condition     = contains(["nyc1", "nyc3", "sfo3", "ams3", "sgp1", "lon1", "fra1", "tor1", "blr1"], var.do_region)
    error_message = "Must be a valid DigitalOcean region with GPU support."
  }
}

variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.27"
}

variable "node_pool_size" {
  description = "Size of the node pool"
  type        = string
  default     = "s-4vcpu-8gb"
}

variable "gpu_droplet_size" {
  description = "Size of GPU droplet"
  type        = string
  default     = "g-2vcpu-8gb"
  validation {
    condition     = can(regex("^g-", var.gpu_droplet_size))
    error_message = "Must be a valid GPU droplet size (starts with 'g-')."
  }
}

variable "ssh_keys" {
  description = "List of SSH key fingerprints to add to the droplet"
  type        = list(string)
  default     = []
}

variable "ssh_key_path" {
  description = "Path to SSH public key file to be added to DigitalOcean"
  type        = string
  default     = ""

  validation {
    condition     = var.ssh_key_path == "" || can(regex("^~?/.+\\.pub$", var.ssh_key_path))
    error_message = "SSH key path must be empty or a valid path to a .pub file."
  }
}

variable "space_name" {
  description = "Name of the Spaces bucket (will be prefixed with environment)"
  type        = string
}

variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = list(string)
  default     = []
}

# Add validation to ensure either ssh_keys or ssh_key_path is provided
variable "has_ssh_access" {
  type        = bool
  default     = true
  validation {
    condition     = var.has_ssh_access
    error_message = "Either ssh_keys or ssh_key_path must be provided."
  }
} 