output "api_id" {
  description = "ID do API Gateway HTTP."
  value       = aws_apigatewayv2_api.this.id
}

output "api_endpoint" {
  description = "Endpoint publico do API Gateway."
  value       = aws_apigatewayv2_api.this.api_endpoint
}

output "api_execution_arn" {
  description = "Execution ARN do API Gateway."
  value       = aws_apigatewayv2_api.this.execution_arn
}

output "default_stage_name" {
  description = "Nome do stage default do API Gateway."
  value       = aws_apigatewayv2_stage.this.name
}
