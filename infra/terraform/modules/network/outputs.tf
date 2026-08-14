output "vpc_id" {
  description = "ID da VPC."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "IDs das subnets publicas."
  value       = [for subnet in aws_subnet.public : subnet.id]
}

output "alb_security_group_id" {
  description = "Security group do ALB."
  value       = aws_security_group.alb.id
}

output "ecs_security_group_id" {
  description = "Security group do ECS."
  value       = aws_security_group.ecs.id
}
