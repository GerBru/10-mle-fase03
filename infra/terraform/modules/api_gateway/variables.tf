variable "name_prefix" {
  type        = string
  description = "Prefixo de naming dos recursos."
}

variable "alb_dns_name" {
  type        = string
  description = "DNS publico do ALB para roteamento HTTP proxy."
}
