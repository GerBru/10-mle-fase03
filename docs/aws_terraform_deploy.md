# Deploy AWS com Terraform

Este documento centraliza a documentação de infraestrutura em nuvem para o projeto.

Escopo atual:

- ECS Fargate para a API
- API Gateway HTTP para exposição pública da API
- ALB como backend da API dentro da stack
- MLflow, Prometheus e Grafana em ECS (roteados por path no ALB/API Gateway)
- ECR como registry oficial
- VPC, subnets e security groups
- CloudWatch Logs

## Onde este documento se encaixa no projeto

- `README.md`: visão geral e acesso rápido.
- `docs/aws_terraform_deploy.md` (este arquivo): guia completo de provisionamento.
- `infra/terraform/README.md`: referência técnica da estrutura de código Terraform.
- `docs/monitoring_plan.md`: estratégia de observabilidade e alertas (camada operacional/modelo).

## Pre-requisitos

- Conta AWS ativa.
- AWS CLI configurado localmente.
- Terraform >= 1.6.
- Docker instalado para build/push da imagem.

## Credenciais AWS

Este projeto **não fixa profile** no Terraform. Tanto o Terraform quanto o AWS CLI e o Docker usam a **cadeia de credenciais padrão da AWS** (profile `default` ou variáveis de ambiente `AWS_*`).

Escolha **uma** das opções abaixo antes de rodar qualquer comando:

- **Profile `default`** (configurado via `aws configure`): nada a fazer, é usado automaticamente.
- **Outro profile**: exporte na sessão para que Terraform, CLI e Docker usem o mesmo:

```bash
export AWS_PROFILE=<seu-profile>
```

- **Variáveis de ambiente** (ex.: credenciais temporárias de lab):

```bash
export AWS_ACCESS_KEY_ID="<sua-access-key>"
export AWS_SECRET_ACCESS_KEY="<sua-secret-key>"
export AWS_SESSION_TOKEN="<token-opcional>"
export AWS_REGION="us-east-1"
```

> Valide SEMPRE que está na conta certa antes de `terraform apply` ou `docker push`. O `docker login` e o `terraform` precisam apontar para a mesma conta:
>
> ```bash
> aws sts get-caller-identity
> ```
>
> Se retornar `InvalidClientTokenId`, a credencial/token está inválida ou expirada e precisa ser renovada.

Não versione credenciais, `terraform.tfvars` reais nem arquivos de state.

### Troca de conta AWS (local state)

Ao trocar de conta mantendo backend local, não reutilize o state antigo.

```bash
cd infra/terraform
rm -rf .terraform .terraform.lock.hcl terraform.tfstate terraform.tfstate.backup tfplan
terraform init -reconfigure
```

Depois, ajuste o ID da conta AWS em `container_image` no `terraform.tfvars` e execute `terraform plan`.

## Fluxo de provisionamento

### Execução rápida (copiar e colar)

Escolha o fluxo conforme o seu ambiente:

- **Fluxo A (conta com permissão IAM para criar roles)**: Terraform cria tudo.
- **Fluxo B (lab/conta com IAM restrito)**: Terraform reutiliza roles existentes via ARN.

#### Fluxo A - conta com permissão IAM

> Os blocos abaixo usam a cadeia de credenciais padrão da AWS (seção **Credenciais AWS**). Se usar um profile específico, exporte `AWS_PROFILE` antes; caso contrário, o `default` é usado.

```bash
# 1) Preparar e validar credencial (usa o profile default / AWS_PROFILE se exportado)
export AWS_REGION=us-east-1
cd infra/terraform
aws sts get-caller-identity   # confirme que é a conta correta antes de continuar

# 2) Primeiro apply sem tasks da API (evita erro de pull antes da imagem existir)
cp -n terraform.tfvars.example terraform.tfvars
# ajuste o terraform.tfvars e deixe desired_count = 0
terraform init -reconfigure
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan

# 3) Login ECR + obter repo (ainda em infra/terraform)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(terraform output -raw ecr_repository_url | cut -d'.' -f4)
ECR_REPO=$(terraform output -raw ecr_repository_url)
echo "ECR_REPO=${ECR_REPO}"
[ -z "${ECR_REPO}" ] && echo "Erro: ecr_repository_url vazio. Rode terraform apply antes." && exit 1
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 4) Build/push da imagem da API (na raiz do projeto)
cd ../..
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f Dockerfile \
  -t "${ECR_REPO}:latest" \
  --push .

# 4b) Build/push da imagem custom do Grafana (provisioning embutido: datasource + dashboard)
# Usa o repositório ECR dedicado do Grafana (output ecr_grafana_repository_url).
GRAFANA_ECR_REPO=$(cd infra/terraform && terraform output -raw ecr_grafana_repository_url)
echo "GRAFANA_ECR_REPO=${GRAFANA_ECR_REPO}"
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f monitoring/grafana/Dockerfile \
  -t "${GRAFANA_ECR_REPO}:latest" \
  --push monitoring/grafana

# 5) Ativar tasks da API e reaplicar infra
cd infra/terraform
# ajuste desired_count = 1 no terraform.tfvars
terraform plan -out tfplan
terraform apply tfplan
```

> O Grafana no ECS usa uma imagem custom (repo ECR dedicado, tag `:latest`) porque o Fargate não monta volumes locais — o provisioning (`monitoring/grafana/provisioning`) precisa estar embutido na imagem. A datasource do Prometheus é resolvida pela variável `PROMETHEUS_URL`, injetada pelo Terraform como `${api_gateway}/prometheus`. Para usar uma imagem própria, defina `grafana_image` no `terraform.tfvars`.

#### Fluxo B - IAM restrito (sem iam:CreateRole)

Mesmo fluxo acima, com um passo extra no `terraform.tfvars` antes do primeiro `plan`:

```hcl
ecs_task_execution_role_arn           = "arn:aws:iam::<account-id>:role/<ecs-exec-role-existente>"
ecs_task_role_arn                     = "arn:aws:iam::<account-id>:role/<ecs-task-role-existente>"
observability_task_execution_role_arn = "arn:aws:iam::<account-id>:role/<obs-exec-role-existente>"
observability_task_role_arn           = "arn:aws:iam::<account-id>:role/<obs-task-role-existente>"
```

Se esses campos estiverem preenchidos, o Terraform não tenta criar novas IAM Roles para ECS/API/observabilidade.

### 1) Preparar variáveis

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Preencha ao menos:

- `container_image`

> O profile/credenciais da AWS NÃO ficam no `terraform.tfvars`. Use o profile `default` ou exporte `AWS_PROFILE`/credenciais na sessão (seção **Credenciais AWS**).

Importante:

- MLflow, Prometheus e Grafana usam imagens públicas e podem subir sem imagem própria no ECR.
- A API de inferência depende da imagem indicada em `container_image` (ou `${ecr_repository_url}:latest`).
- Se a imagem da API ainda não existir no ECR, o service pode ser criado, mas as tasks da API podem falhar até o push da imagem.

Configuração padrão otimizada para menor custo:

- API em Fargate com `task_cpu=256` e `task_memory=512`.
- Observabilidade com `observability_desired_count=0` (MLflow/Prometheus/Grafana criados, mas sem tasks rodando).
- Retenção de logs reduzida para 7 dias.

Quando habilitar observabilidade (`observability_desired_count=1`), recomenda-se:

- `observability_task_cpu = 512`
- `observability_task_memory = 2048`

O `mlflow` foi configurado com `--workers 1` para reduzir consumo de memória e evitar reinícios por OOM durante o health check do ALB.

### 2) Criar infraestrutura

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan
```

Explicação dos comandos:


| Comando                      | O que faz                                                                             | Quando usar                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `terraform init`             | Inicializa o diretório Terraform, baixa providers/módulos e prepara o ambiente local. | Na primeira execução do projeto ou quando houver mudança de providers/módulos/backend. |
| `terraform fmt -recursive`   | Formata os arquivos `.tf` no padrão oficial do Terraform, inclusive em subpastas.     | Sempre antes de validar ou versionar alterações de IaC.                                |
| `terraform validate`         | Valida sintaxe e consistência da configuração, sem criar recursos na AWS.             | Depois de alterar arquivos Terraform e antes do `plan`.                                |
| `terraform plan -out tfplan` | Gera o plano de mudanças e salva no arquivo `tfplan`.                                 | Para revisar exatamente o que será criado/alterado/removido.                           |
| `terraform apply tfplan`     | Aplica exatamente o plano salvo em `tfplan`, sem recalcular mudanças.                 | Depois de revisar o plano e aprovar a execução.                                        |


### Ordem recomendada (conta nova)

Para evitar erro de pull da imagem da API no primeiro deploy:

1. Em `terraform.tfvars`, use `desired_count = 0`.
2. Rode `terraform apply` para criar a infraestrutura base (incluindo ECR).
3. Faça o build/push da imagem da API para o ECR (seção abaixo).
4. Altere `desired_count = 1` e rode novo `terraform apply`.

### Erros comuns de autenticação/autorização

- `InvalidClientTokenId`: credenciais inválidas/expiradas (renove Access Key/Secret/Session Token).
- `UnauthorizedOperation` (ex.: `ec2:DescribeAvailabilityZones`): credencial válida, mas sem permissão IAM na conta atual.

Importante: permissões IAM base devem ser ajustadas diretamente na conta AWS (console/policy/role). Terraform só consegue gerenciar IAM se a credencial atual já tiver privilégios para isso.

#### Caso específico: erro no `data "aws_availability_zones" "available"`

Se o `terraform plan` parar no `main.tf` com erro de `ec2:DescribeAvailabilityZones`, isso significa que o usuário/role atual não tem permissão de leitura de Availability Zones.

Como resolver na AWS (Console IAM):

1. Acesse **IAM > Policies > Create policy**.
2. Crie uma policy com permissão mínima para leitura de AZ:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeAvailabilityZones"
      ],
      "Resource": "*"
    }
  ]
}
```

1. Anexe essa policy na role/usuário que você usa no Terraform (ex.: role assumida pelo lab).
2. Valide novamente no terminal:

```bash
aws ec2 describe-availability-zones --region us-east-1
```

Se esse comando funcionar, o Terraform deve passar dessa etapa.

#### Workaround para ambientes com IAM restrito (sem `iam:CreateRole`)

Se a sua role não puder criar IAM Roles, reutilize roles já existentes preenchendo no `terraform.tfvars`:

```hcl
ecs_task_execution_role_arn           = "arn:aws:iam::<account-id>:role/<ecs-exec-role-existente>"
ecs_task_role_arn                     = "arn:aws:iam::<account-id>:role/<ecs-task-role-existente>"
observability_task_execution_role_arn = "arn:aws:iam::<account-id>:role/<obs-exec-role-existente>"
observability_task_role_arn           = "arn:aws:iam::<account-id>:role/<obs-task-role-existente>"
```

Com esses campos preenchidos, o Terraform não tenta criar novas roles para ECS/API/observabilidade.

Outputs principais:

- `api_gateway_url`
- `api_docs_url`
- `api_health_url`
- `api_ready_url`
- `mlflow_url`
- `prometheus_url`
- `grafana_url`
- `alb_dns_name`
- `ecr_repository_url` (repo da imagem da API)
- `ecr_grafana_repository_url` (repo da imagem custom do Grafana)
- `ecs_cluster_name`
- `ecs_service_name`

## Publicar imagens no ECR

Este projeto sobe **duas** imagens para o ECR:

1. **API de inferência** — repo do output `ecr_repository_url`, build a partir do `Dockerfile` na raiz.
2. **Grafana customizado** — repo do output `ecr_grafana_repository_url`, build a partir de `monitoring/grafana/Dockerfile` (provisioning embutido).

### Login no ECR (vale para as duas imagens)

Com a infraestrutura criada, faça o login uma vez. Execute este bloco dentro de `infra/terraform`:

```bash
export AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Conta autenticada: ${AWS_ACCOUNT_ID} (confirme que é a conta esperada)"
ECR_REPO=$(terraform output -raw ecr_repository_url)
echo "ECR_REPO=${ECR_REPO}"
[ -z "${ECR_REPO}" ] && echo "Erro: ecr_repository_url vazio. Rode terraform apply antes." && exit 1

aws ecr get-login-password --region "${AWS_REGION}" | \
docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
cd ../..
```

> O `docker login` autentica no **registry** (`<account>.dkr.ecr.<region>.amazonaws.com`), que é o mesmo para os dois repositórios. Logo, um único login serve tanto para a imagem da API quanto para a do Grafana.

### Build e push da imagem da API (por sistema operacional)

#### macOS Apple Silicon (M1/M2/M3 - arm64)

Use build direcionado para `linux/amd64` (arquitetura esperada no ECS/Fargate neste projeto):

> Execute estes comandos na raiz do projeto (diretório que contém o `Dockerfile`).

```bash
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f Dockerfile \
  -t "${ECR_REPO}:latest" \
  --push .
```

Flags de otimização:

- `--provenance=false --sbom=false`: evita gerar manifests de attestation extras (push mais rápido).
- O builder `docker-container` já mantém **cache local** entre builds na mesma máquina, então apenas as camadas que mudaram são re-enviadas no push seguinte.

> Atenção ao cache de registry: `--cache-to type=registry,...,mode=max` sobe uma **segunda cópia** de todas as camadas para o ECR (no log aparece como `exporting cache to registry`, que pode dobrar o tempo total). Só vale a pena em **CI/múltiplas máquinas**, onde não há cache local. Numa máquina única, evite. Se precisar em CI, prefira `mode=min` ou cache inline:
>
> ```bash
> # CI: cache inline (sem segundo upload completo)
> docker buildx build \
>   --platform linux/amd64 \
>   --provenance=false --sbom=false \
>   --cache-from "type=registry,ref=${ECR_REPO}:buildcache" \
>   --cache-to   "type=inline" \
>   -f Dockerfile -t "${ECR_REPO}:latest" --push .
> ```

> Imagem otimizada (2 cortes em `pyproject.toml`):
>
> 1. **torch CPU-only** em Linux (sem bibliotecas CUDA da NVIDIA) via `[tool.uv.sources]` + índice `pytorch-cpu` — reduz ~2GB.
> 2. **Libs de treino/EDA fora do runtime**: `mlflow`, `matplotlib`, `seaborn`, `imbalanced-learn`, `openpyxl` e `pandera` ficam no extra `train` (`[project.optional-dependencies].train`). A API não as importa, e o `Dockerfile` usa `uv sync --no-dev` (sem `--extra train`), então elas não entram na imagem.
>
> Nenhum dos cortes afeta o ambiente local: para treinar/EDA use `uv sync --extra dev --extra train`.

#### Linux/Windows x86_64 (amd64)

Build direto costuma ser suficiente:

```bash
docker build -f Dockerfile -t "${ECR_REPO}:latest" .
docker push "${ECR_REPO}:latest"
```

Se atualizar a imagem/tag, rode novo `terraform apply` ajustando `container_image`.

### Observação importante sobre arquitetura da imagem

Se a imagem for enviada somente como `arm64` (comum em Mac Apple Silicon), o servico pode falhar no ECS/Fargate com:

- `CannotPullContainerError`
- `image Manifest does not contain descriptor matching platform 'linux/amd64'`

Nessa situação, normalmente o log stream aparece no CloudWatch, mas sem eventos (`storedBytes = 0`) porque o container não chega a iniciar.

### Build e push da imagem do Grafana

O Grafana no ECS/Fargate usa uma **imagem custom** porque o Fargate não monta volumes locais — o provisioning (datasource do Prometheus + dashboards em `monitoring/grafana/provisioning`) precisa estar **embutido na imagem**. Por isso existe um repositório ECR dedicado, exposto no output `ecr_grafana_repository_url`.

> **Pré-requisitos:** ter feito o **login no ECR** (seção acima). O login é o mesmo registry, então serve para as duas imagens.
>
> **Contexto do build:** diferente da API (contexto `.`), aqui o contexto é a pasta `monitoring/grafana`, porque o `Dockerfile` faz `COPY provisioning/ ...` relativo a ela. Rode a partir da **raiz do projeto**.

```bash
export AWS_REGION=us-east-1

# Repo ECR dedicado do Grafana (terraform output via subshell, sem precisar mudar de pasta)
GRAFANA_ECR_REPO=$(cd infra/terraform && terraform output -raw ecr_grafana_repository_url)
echo "GRAFANA_ECR_REPO=${GRAFANA_ECR_REPO}"
[ -z "${GRAFANA_ECR_REPO}" ] && echo "Erro: ecr_grafana_repository_url vazio. Rode terraform apply antes." && exit 1

# Build + push (contexto = monitoring/grafana)
docker buildx create --use --name churn-amd64-builder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -f monitoring/grafana/Dockerfile \
  -t "${GRAFANA_ECR_REPO}:latest" \
  --push monitoring/grafana
```

> No `linux/amd64` (x86_64) você pode usar `docker build`/`docker push` no lugar do `buildx`, igual à imagem da API. Em Mac Apple Silicon mantenha o `buildx --platform linux/amd64` para evitar o erro de manifest descrito acima.

Detalhes úteis:

- A datasource do Prometheus é resolvida em runtime pela variável `PROMETHEUS_URL`, injetada pelo Terraform como `${api_gateway}/prometheus`. Você não precisa editar nada na imagem para isso.
- Por padrão o ECS usa `${ecr_grafana_repository_url}:latest`. Para fixar outra imagem, defina `grafana_image` no `terraform.tfvars`.
- Após o push, se o service do Grafana já estiver rodando (`observability_desired_count >= 1`), force o redeploy para puxar a nova imagem `:latest`. O service segue o padrão `${project_name}-${environment}-grafana-service` (ex.: `churn-prediction-dev-grafana-service`):

```bash
CLUSTER=$(cd infra/terraform && terraform output -raw ecs_cluster_name)
aws ecs update-service \
  --cluster "${CLUSTER}" \
  --service "churn-prediction-dev-grafana-service" \
  --force-new-deployment \
  --region "${AWS_REGION}"
```

## Reset completo da infraestrutura (apagar e recriar)

Use este fluxo quando quiser limpar toda a infra e provisionar do zero.

### 1) Destruir tudo que o Terraform criou

```bash
cd infra/terraform
terraform init
terraform destroy
```

Se você acabou de habilitar `force_delete = true` no ECR, rode antes:

```bash
terraform plan -out tfplan
terraform apply tfplan
```

Isso atualiza o estado da infraestrutura para que o `destroy` consiga remover repositório ECR com imagens.

Opcional para execução não interativa:

```bash
terraform destroy -auto-approve
```

### 2) Limpar artefatos locais (opcional)

```bash
rm -rf .terraform .terraform.lock.hcl terraform.tfstate terraform.tfstate.backup tfplan
```

Observações:

- Não remova `terraform.tfvars` se quiser manter seus parâmetros.
- Esse passo remove apenas arquivos locais; os recursos AWS já devem ter sido apagados no `destroy`.
- O módulo ECR usa `force_delete = true`, então o repositório é apagado mesmo com imagens.

### 3) Provisionar novamente

```bash
terraform init
terraform fmt -recursive
terraform validate
terraform plan -out tfplan
terraform apply tfplan
```

### 4) Reenviar imagens para o ECR recriado

Use exatamente o mesmo procedimento da seção **Publicar imagens no ECR** (confirme a conta com `aws sts get-caller-identity` antes).

Resumo:

1. Confirme que está na conta certa (profile `default` ou `AWS_PROFILE` exportado).
2. Faça login no ECR (vale para as duas imagens).
3. Build e push da imagem da **API** (`ecr_repository_url`).
4. Build e push da imagem do **Grafana** (`ecr_grafana_repository_url`).

## Validação pós-deploy

- `terraform output -raw api_gateway_url`: endpoint principal para consumir a API.
- `terraform output -raw api_docs_url`: endpoint de testes via Swagger.
- `terraform output -raw api_health_url`: endpoint de health check.
- `terraform output -raw api_ready_url`: endpoint de readiness.
- `terraform output -raw mlflow_url`: endpoint do MLflow.
- `terraform output -raw prometheus_url`: endpoint do Prometheus.
- `terraform output -raw grafana_url`: endpoint do Grafana (usuário/senha padrão: `admin` / `admin123`).
- `GET /health`: disponibilidade básica.
- `GET /ready`: readiness da API com artefatos carregados.
- Verificar logs no CloudWatch do grupo `/ecs/<project>-<env>-api`.

## Notas sobre Free Tier

Mesmo com tuning de custo, alguns recursos desta arquitetura normalmente geram cobrança:

- ALB (Application Load Balancer)
- API Gateway HTTP
- ECS Fargate (quando `desired_count > 0`)

Para testes com menor custo, mantenha:

- `observability_desired_count = 0` (padrão)
- `task_cpu = 256` e `task_memory = 512` (padrão)
- `desired_count = 1` apenas durante validações

