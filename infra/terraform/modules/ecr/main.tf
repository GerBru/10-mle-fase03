# Repositório ECR (tag mutável para facilitar ciclo de deploy).
resource "aws_ecr_repository" "api" {
  name                 = "${var.name_prefix}-${var.repo_suffix}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy para conter custo de storage removendo imagens antigas.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Manter somente as 5 imagens mais recentes"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
