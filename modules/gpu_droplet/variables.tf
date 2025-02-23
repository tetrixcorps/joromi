variable "name" {
  description = "Name of the GPU droplet"
  type        = string
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
}

variable "size" {
  description = "Size of the droplet"
  type        = string
}

variable "ssh_keys" {
  description = "List of SSH key IDs"
  type        = list(string)
}

variable "environment" {
  description = "Environment name"
  type        = string
} 