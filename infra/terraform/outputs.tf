# Endpoints públicos para consumo da API e testes rápidos.
output "alb_dns_name" {
  description = "DNS publico do Application Load Balancer."
  value       = module.alb.alb_dns_name
}

output "api_gateway_url" {
  description = "URL publica do API Gateway que encaminha para o ALB."
  value       = module.api_gateway.api_endpoint
}

output "api_docs_url" {
  description = "URL de testes da documentacao Swagger."
  value       = "${module.api_gateway.api_endpoint}/docs"
}

output "api_health_url" {
  description = "URL de health check da API."
  value       = "${module.api_gateway.api_endpoint}/health"
}

output "api_ready_url" {
  description = "URL de readiness da API."
  value       = "${module.api_gateway.api_endpoint}/ready"
}

# Endpoints públicos dos serviços de observabilidade.
output "mlflow_url" {
  description = "URL publica do MLflow via API Gateway."
  value       = "${module.api_gateway.api_endpoint}/mlflow"
}

output "prometheus_url" {
  description = "URL publica do Prometheus via API Gateway."
  value       = "${module.api_gateway.api_endpoint}/prometheus"
}

output "grafana_url" {
  description = "URL publica do Grafana via API Gateway."
  value       = "${module.api_gateway.api_endpoint}/grafana"
}

# Saídas de suporte para operação/deploy.
output "ecr_repository_url" {
  description = "URL do repositorio ECR da aplicacao."
  value       = module.ecr.repository_url
}

output "ecr_grafana_repository_url" {
  description = "URL do repositorio ECR da imagem custom do Grafana."
  value       = module.ecr_grafana.repository_url
}

output "ecs_cluster_name" {
  description = "Nome do cluster ECS."
  value       = module.ecs_service.cluster_name
}

output "ecs_service_name" {
  description = "Nome do servico ECS."
  value       = module.ecs_service.service_name
}
