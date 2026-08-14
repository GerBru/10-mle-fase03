# Terraform AWS - Churn API

Infraestrutura minima para executar a API em AWS com:

- VPC + subnets publicas
- Security groups para ALB e ECS
- ECR para armazenar a imagem da aplicacao
- ECS Fargate + CloudWatch Logs
- API Gateway HTTP na frente do ALB para exposicao publica da API
- Servicos de observabilidade no ECS (MLflow, Prometheus, Grafana)

Padrao de custo aplicado:

- API: `256 CPU / 512 MiB`
- Observabilidade: `desired_count = 0` (sem tasks rodando por padrao)
- Retencao de logs: 7 dias

## Estrutura

```
infra/terraform/
├── main.tf
├── observability.tf
├── provider.tf
├── variables.tf
├── outputs.tf
├── terraform.tfvars.example
├── environments/
│   └── dev/
│       └── terraform.tfvars.example
└── modules/
    ├── alb/
    ├── api_gateway/
    ├── ecr/
    ├── ecs_service/
    └── network/
```

## Credenciais AWS (sem hardcode)

O provider AWS foi configurado para funcionar com:

1. `AWS_PROFILE` (recomendado), ou
2. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` e opcionalmente `AWS_SESSION_TOKEN`.

Nao adicione credenciais no codigo Terraform.

## Documentacao completa

O guia completo de provisionamento, deploy e validacao esta em:

- `docs/aws_terraform_deploy.md`

O procedimento de apagar toda a infra e recriar também esta documentado nesse guia.