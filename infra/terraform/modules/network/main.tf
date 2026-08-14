# VPC principal com DNS habilitado para permitir descoberta interna dos serviços.
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.name_prefix}-vpc"
  }
}

# Gateway de internet para liberar saída/entrada das subnets públicas.
resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name_prefix}-igw"
  }
}

# Subnets públicas distribuídas em múltiplas AZs para alta disponibilidade.
resource "aws_subnet" "public" {
  for_each = {
    for idx, cidr in var.public_subnet_cidrs : idx => {
      cidr = cidr
      az   = var.availability_zones[idx]
    }
  }

  vpc_id                  = aws_vpc.this.id
  cidr_block              = each.value.cidr
  availability_zone       = each.value.az
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.name_prefix}-public-${each.key}"
    Tier = "public"
  }
}

# Tabela de rota pública com saída default para a internet.
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = {
    Name = "${var.name_prefix}-public-rt"
  }
}

# Associação de cada subnet pública à tabela de rotas pública.
resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

# Security group do ALB: recebe tráfego público HTTP.
resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb-sg"
  description = "Permite acesso HTTP ao ALB"
  vpc_id      = aws_vpc.this.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.alb_ingress_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-alb-sg"
  }
}

# Security group do ECS: aceita tráfego apenas do SG do ALB.
# IMPORTANTE: nao usar bloco `ingress` inline aqui. Outros arquivos (ex.: observability.tf)
# adicionam regras de ingress via `aws_security_group_rule`. Misturar regras inline com
# regras standalone faz o provider AWS sobrescrever/remover as standalone a cada apply,
# derrubando a conectividade do ALB ate as portas de MLflow/Prometheus/Grafana.
resource "aws_security_group" "ecs" {
  name        = "${var.name_prefix}-ecs-sg"
  description = "Permite trafego do ALB para tarefas ECS"
  vpc_id      = aws_vpc.this.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-ecs-sg"
  }
}

# Regra de ingress da API como recurso standalone, evitando conflito com as regras
# de observabilidade definidas em observability.tf.
resource "aws_security_group_rule" "ecs_ingress_api_from_alb" {
  type                     = "ingress"
  from_port                = var.container_port
  to_port                  = var.container_port
  protocol                 = "tcp"
  security_group_id        = aws_security_group.ecs.id
  source_security_group_id = aws_security_group.alb.id
  description              = "Permite ALB acessar a API no ECS"
}
