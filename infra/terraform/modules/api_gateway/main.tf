# HTTP API pública usada como fachada única para o ALB.
resource "aws_apigatewayv2_api" "this" {
  name          = "${var.name_prefix}-http-api"
  protocol_type = "HTTP"
}

# Integração proxy para encaminhar todas as rotas ao ALB.
resource "aws_apigatewayv2_integration" "alb_proxy" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  payload_format_version = "1.0"
  integration_uri        = "http://${var.alb_dns_name}"
}

# Rota default captura qualquer path e repassa para a integração.
resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.this.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.alb_proxy.id}"
}

# Stage $default com deploy automático para aplicar mudanças sem etapa manual.
resource "aws_apigatewayv2_stage" "this" {
  api_id      = aws_apigatewayv2_api.this.id
  name        = "$default"
  auto_deploy = true
}
