output "alb_arn" {
  description = "ARN do ALB."
  value       = aws_lb.this.arn
}

output "alb_dns_name" {
  description = "DNS publico do ALB."
  value       = aws_lb.this.dns_name
}

output "target_group_arn" {
  description = "ARN do target group."
  value       = aws_lb_target_group.api.arn
}

output "listener_arn" {
  description = "ARN do listener HTTP."
  value       = aws_lb_listener.http.arn
}
