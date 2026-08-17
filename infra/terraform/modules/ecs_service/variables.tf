variable "name_prefix" {
  type        = string
  description = "Prefixo de naming dos recursos."
}

variable "aws_region" {
  type        = string
  description = "Regiao AWS para logs do ECS."
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnets para executar as tasks do ECS."
}

variable "ecs_security_group_id" {
  type        = string
  description = "Security group do ECS service."
}

variable "target_group_arn" {
  type        = string
  description = "Target group ARN do ALB."
}

variable "container_port" {
  type        = number
  description = "Porta exposta pelo container."
}

variable "desired_count" {
  type        = number
  description = "Quantidade de tasks desejada."
}

variable "task_cpu" {
  type        = number
  description = "CPU da task Fargate."
}

variable "task_memory" {
  type        = number
  description = "Memoria da task Fargate em MiB."
}

variable "container_image" {
  type        = string
  description = "Imagem do container."
}

variable "log_group_retention_in_days" {
  type        = number
  description = "Retencao de logs em dias."
  default     = 14
}

variable "environment_variables" {
  type        = map(string)
  description = "Variaveis de ambiente nao sensiveis da aplicacao."
  default     = {}
}

variable "secret_environment" {
  type        = map(string)
  description = "Mapeamento nome_da_variavel => valueFrom ARN no formato ECS."
  default     = {}
}

variable "secret_arns_for_execution" {
  type        = list(string)
  description = "Lista de ARNs de secrets permitidos para o execution role."
  default     = []
}

variable "existing_task_execution_role_arn" {
  type        = string
  description = "ARN de execution role existente para reutilizar. Se vazio, o Terraform cria uma nova."
  default     = ""
}

variable "existing_task_role_arn" {
  type        = string
  description = "ARN de task role existente para reutilizar. Se vazio, o Terraform cria uma nova."
  default     = ""
}
