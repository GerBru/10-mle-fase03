output "repository_url" {
  description = "URL do repositorio ECR."
  value       = aws_ecr_repository.api.repository_url
}

output "repository_name" {
  description = "Nome do repositorio ECR."
  value       = aws_ecr_repository.api.name
}
