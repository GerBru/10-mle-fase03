# Provider AWS único do projeto com tags padrão para governança/custo.
# Sem 'profile' explícito: usa a cadeia de credenciais padrão do AWS CLI/SDK
# (profile default ou variaveis de ambiente AWS_*). Responsabilidade de quem faz o deploy.
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(
      {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "terraform"
      },
      var.tags
    )
  }
}
