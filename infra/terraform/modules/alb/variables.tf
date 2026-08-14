variable "name_prefix" {
  type        = string
  description = "Prefixo de naming dos recursos."
}

variable "vpc_id" {
  type        = string
  description = "ID da VPC."
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnets onde o ALB sera criado."
}

variable "alb_sg_id" {
  type        = string
  description = "Security group do ALB."
}

variable "target_port" {
  type        = number
  description = "Porta do container no target group."
}

variable "health_check_path" {
  type        = string
  description = "Path de healthcheck."
}
