output "cluster_name" {
  description = "Nome do cluster ECS."
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "Nome do ECS service."
  value       = aws_ecs_service.this.name
}

output "task_definition_arn" {
  description = "ARN da task definition ativa."
  value       = aws_ecs_task_definition.this.arn
}

output "log_group_name" {
  description = "Nome do log group da API."
  value       = aws_cloudwatch_log_group.api.name
}
