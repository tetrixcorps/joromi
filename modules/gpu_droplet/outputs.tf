output "droplet_ip" {
  value = digitalocean_droplet.gpu.ipv4_address
}

output "droplet_id" {
  value = digitalocean_droplet.gpu.id
} 