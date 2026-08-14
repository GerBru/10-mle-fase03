# Configurações globais da conta/região e identificação de ambiente.
variable "aws_region" {
  type        = string
  description = "Regiao AWS para criar os recursos."
  default     = "sa-east-1"
}

variable "project_name" {
  type        = string
  description = "Nome base do projeto para prefixar recursos."
  default     = "churn-prediction"
}

variable "environment" {
  type        = string
  description = "Ambiente de deploy (ex.: dev, staging, prod)."
  default     = "dev"
}

variable "tags" {
  type        = map(string)
  description = "Tags adicionais para os recursos."
  default     = {}
}

# Rede e exposição da API.
variable "vpc_cidr" {
  type        = string
  description = "CIDR da VPC."
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDRs para as subnets publicas."
  default     = ["10.42.1.0/24", "10.42.2.0/24"]
}

variable "container_port" {
  type        = number
  description = "Porta exposta pelo container da API."
  default     = 8000
}

variable "desired_count" {
  type        = number
  description = "Quantidade desejada de tarefas no ECS Service."
  default     = 1
}

variable "task_cpu" {
  type        = number
  description = "CPU da task ECS (unidades Fargate)."
  default     = 256
}

variable "task_memory" {
  type        = number
  description = "Memoria da task ECS em MiB."
  default     = 512
}

variable "container_image" {
  type        = string
  description = "Imagem Docker da API (ex.: <account>.dkr.ecr.<region>.amazonaws.com/repo:tag)."
  default     = ""
}

variable "grafana_image" {
  type        = string
  description = "Imagem Docker custom do Grafana (com provisioning embutido). Se vazio, usa o repo ECR dedicado do Grafana com a tag latest."
  default     = ""
}

variable "jwt_expire_minutes" {
  type        = number
  description = "TTL do token JWT em minutos."
  default     = 60
}

variable "models_dir" {
  type        = string
  description = "Diretorio dos artefatos de modelo no container."
  default     = "models"
}

variable "log_level" {
  type        = string
  description = "Nivel de log da aplicacao."
  default     = "INFO"
}

variable "rate_limit_requests" {
  type        = number
  description = "Quantidade de requests permitida por janela."
  default     = 100
}

variable "rate_limit_window" {
  type        = number
  description = "Janela do rate limit em segundos."
  default     = 60
}

variable "health_check_path" {
  type        = string
  description = "Path de health check no ALB target group."
  default     = "/ready"
}

variable "alb_ingress_cidr" {
  type        = string
  description = "CIDR autorizado a acessar o ALB."
  default     = "0.0.0.0/0"
}

# Dimensionamento e imagem da API principal.
variable "api_log_retention_in_days" {
  type        = number
  description = "Retencao de logs da API no CloudWatch."
  default     = 7
}

# Dimensionamento e retenção dos serviços de observabilidade.
variable "observability_desired_count" {
  type        = number
  description = "Quantidade desejada de tasks para MLflow/Prometheus/Grafana."
  default     = 1
}

variable "observability_task_cpu" {
  type        = number
  description = "CPU por task de observabilidade (unidades Fargate)."
  default     = 256
}

variable "observability_task_memory" {
  type        = number
  description = "Memoria por task de observabilidade em MiB."
  default     = 512
}

variable "observability_log_retention_in_days" {
  type        = number
  description = "Retencao de logs dos servicos de observabilidade no CloudWatch."
  default     = 7
}

# ARNs de roles opcionais para ambientes com IAM restrito (sem iam:CreateRole).
variable "ecs_task_execution_role_arn" {
  type        = string
  description = "ARN de execution role existente para a API ECS. Se vazio, o Terraform cria uma nova."
  default     = ""
}

variable "ecs_task_role_arn" {
  type        = string
  description = "ARN de task role existente para a API ECS. Se vazio, o Terraform cria uma nova."
  default     = ""
}

variable "observability_task_execution_role_arn" {
  type        = string
  description = "ARN de execution role existente para MLflow/Prometheus/Grafana. Se vazio, o Terraform cria uma nova."
  default     = ""
}

variable "observability_task_role_arn" {
  type        = string
  description = "ARN de task role existente para MLflow/Prometheus/Grafana. Se vazio, o Terraform cria uma nova."
  default     = ""
}

