# Convenções locais do serviço principal da API.
locals {
  app_name                   = "${var.name_prefix}-api"
  secrets_policy_resources   = length(var.secret_arns_for_execution) > 0 ? var.secret_arns_for_execution : ["*"]
  create_task_execution_role = var.existing_task_execution_role_arn == ""
  create_task_role           = var.existing_task_role_arn == ""
}

# Log group central da API no CloudWatch.
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.app_name}"
  retention_in_days = var.log_group_retention_in_days
}

# Cluster ECS dedicado ao ambiente (reutilizado pela observabilidade).
resource "aws_ecs_cluster" "this" {
  name = "${var.name_prefix}-cluster"
}

# Role base para tasks ECS.
data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Role de execução da task (pull de imagem, envio de logs, etc.).
resource "aws_iam_role" "task_execution" {
  count = local.create_task_execution_role ? 1 : 0

  name               = "${var.name_prefix}-ecs-exec-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

# Permissão gerenciada padrão necessária para execução de task ECS/Fargate.
resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  count = local.create_task_execution_role ? 1 : 0

  role       = aws_iam_role.task_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Policy opcional para leitura de secrets em runtime.
data "aws_iam_policy_document" "task_execution_secrets" {
  statement {
    sid       = "ReadAppSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = local.secrets_policy_resources
  }
}

# Anexa a policy opcional de secrets quando houver ARNs configurados.
resource "aws_iam_role_policy" "task_execution_secrets" {
  count = length(var.secret_arns_for_execution) > 0 && local.create_task_execution_role ? 1 : 0

  name   = "${var.name_prefix}-ecs-exec-secrets"
  role   = aws_iam_role.task_execution[0].id
  policy = data.aws_iam_policy_document.task_execution_secrets.json
}

# Task role da aplicação (permissões de negócio, quando necessário).
resource "aws_iam_role" "task" {
  count = local.create_task_role ? 1 : 0

  name               = "${var.name_prefix}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

# Task definition da API com logs, env vars e integração com ALB.
resource "aws_ecs_task_definition" "this" {
  family                   = local.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = local.create_task_execution_role ? aws_iam_role.task_execution[0].arn : var.existing_task_execution_role_arn
  task_role_arn            = local.create_task_role ? aws_iam_role.task[0].arn : var.existing_task_role_arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = [
        for name, value in var.environment_variables : {
          name  = name
          value = value
        }
      ]
      secrets = [
        for name, arn_ref in var.secret_environment : {
          name      = name
          valueFrom = arn_ref
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  depends_on = [
    aws_iam_role_policy_attachment.task_execution_managed
  ]
}

# Serviço ECS/Fargate principal da API.
resource "aws_ecs_service" "this" {
  name            = "${var.name_prefix}-service"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "api"
    container_port   = var.container_port
  }

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.ecs_security_group_id]
    assign_public_ip = true
  }

  # Evita drift por revisões de task disparadas fora do Terraform.
  lifecycle {
    ignore_changes = [task_definition]
  }
}
