# Read local SSH key file if provided
data "local_file" "ssh_key" {
  count    = var.ssh_key_path != "" && length(var.ssh_keys) == 0 ? 1 : 0
  filename = pathexpand(var.ssh_key_path)
}

# Create SSH key in DigitalOcean if local key is provided
resource "digitalocean_ssh_key" "default" {
  count      = var.ssh_key_path != "" && length(var.ssh_keys) == 0 ? 1 : 0
  name       = "${local.name_prefix}-ssh-key"
  public_key = data.local_file.ssh_key[0].content
}

locals {
  # Use existing SSH keys if provided, otherwise use newly created one
  ssh_key_ids = length(var.ssh_keys) > 0 ? var.ssh_keys : (
    var.ssh_key_path != "" ? [digitalocean_ssh_key.default[0].fingerprint] : []
  )
} 