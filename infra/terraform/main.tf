# Descobre AZs disponíveis para distribuir as subnets públicas.
data "aws_availability_zones" "available" {
  state = "available"
}

# Convenções globais e variáveis de ambiente da aplicação.
locals {
  name_prefix = "${var.project_name}-${var.environment}"

  app_environment = {
    API_HOST            = "0.0.0.0"
    API_PORT            = tostring(var.container_port)
    MODELS_DIR          = var.models_dir
    LOG_LEVEL           = var.log_level
    JWT_EXPIRE_MINUTES  = tostring(var.jwt_expire_minutes)
    RATE_LIMIT_REQUESTS = tostring(var.rate_limit_requests)
    RATE_LIMIT_WINDOW   = tostring(var.rate_limit_window)
  }
}

# Camada de rede base (VPC, subnets, roteamento e security groups).
module "network" {
  source = "./modules/network"

  name_prefix         = local.name_prefix
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidrs = var.public_subnet_cidrs
  availability_zones  = slice(data.aws_availability_zones.available.names, 0, length(var.public_subnet_cidrs))
  alb_ingress_cidr    = var.alb_ingress_cidr
  container_port      = var.container_port
}

# Registro de imagens Docker da aplicação.
module "ecr" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
}

# Repositório ECR dedicado à imagem custom do Grafana (provisioning embutido).
module "ecr_grafana" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
  repo_suffix = "grafana"
}

# Exposição HTTP pública com balanceamento para os serviços ECS.
module "alb" {
  source = "./modules/alb"

  name_prefix       = local.name_prefix
  vpc_id            = module.network.vpc_id
  subnet_ids        = module.network.public_subnet_ids
  alb_sg_id         = module.network.alb_security_group_id
  target_port       = var.container_port
  health_check_path = var.health_check_path
}

# Camada de entrada pública via API Gateway apontando para o ALB.
module "api_gateway" {
  source = "./modules/api_gateway"

  name_prefix  = local.name_prefix
  alb_dns_name = module.alb.alb_dns_name

  depends_on = [
    module.alb
  ]
}

# Serviço principal da API em ECS/Fargate.
module "ecs_service" {
  source = "./modules/ecs_service"

  name_prefix = local.name_prefix
  aws_region  = var.aws_region

  subnet_ids                       = module.network.public_subnet_ids
  ecs_security_group_id            = module.network.ecs_security_group_id
  target_group_arn                 = module.alb.target_group_arn
  container_port                   = var.container_port
  desired_count                    = var.desired_count
  task_cpu                         = var.task_cpu
  task_memory                      = var.task_memory
  container_image                  = var.container_image != "" ? var.container_image : "${module.ecr.repository_url}:latest"
  log_group_retention_in_days      = var.api_log_retention_in_days
  existing_task_execution_role_arn = var.ecs_task_execution_role_arn
  existing_task_role_arn           = var.ecs_task_role_arn

  environment_variables = local.app_environment

  depends_on = [
    module.alb
  ]
}
