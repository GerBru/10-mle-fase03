variable "name_prefix" {
  type        = string
  description = "Prefixo dos recursos."
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR da VPC."
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDRs das subnets publicas."
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones correspondentes as subnets."
}

variable "alb_ingress_cidr" {
  type        = string
  description = "CIDR permitido no ALB."
}

variable "container_port" {
  type        = number
  description = "Porta do container da API."
}
