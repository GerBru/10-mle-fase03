# 🔮 Churn Prediction — FIAP Tech Challenge Fase 2

> Modelo preditivo de churn para operadora de telecomunicações.
> Pipeline end-to-end reprodutível: DVC → Baselines → Model Registry → MLP (PyTorch) → API (FastAPI) → Deploy (Docker + CI/CD).

**Fase 2** adiciona fundamentos de MLOps sobre a base da Fase 1: versionamento de dados
e pipeline reprodutível com **DVC**, e promoção de modelos com o **MLflow Model Registry**.
Toda a estrutura da Fase 1 (API, monitoramento, MLP PyTorch, IaC) foi preservada.

Python
PyTorch
FastAPI
MLflow
Fairlearn
Ruff
CI
License

---

## 📋 Contexto

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Este projeto
constrói um sistema preditivo de churn do zero até o modelo servido via API, aplicando
boas práticas de Machine Learning Engineering:

- **Versionamento de dados:** DVC — dataset e artefatos fora do git, pipeline em estágios (`preprocess` → `train`)
- **Modelo principal:** Rede neural MLP treinada com PyTorch
- **Baselines:** DummyClassifier, LogisticRegression, RandomForest (com RandomizedSearchCV), GradientBoosting
- **Rastreamento:** MLflow (parâmetros, métricas, artefatos)
- **Promoção de modelos:** MLflow Model Registry — melhor baseline sklearn promovido com alias `@champion`
- **Serving:** API REST com FastAPI + Pydantic (single e batch inference)
- **Segurança:** JWT + API Key + Rate Limiting + CORS
- **Monitoramento:** Drift detection com KS test + PSI; análise de fairness com Fairlearn MetricFrame
- **Deploy:** Docker multi-stage + CI/CD via GitHub Actions + AWS ECS Fargate

---

## 🗂️ Estrutura do Projeto

```
10-machine-learning-engineering/
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI — rotas, middlewares, /metrics e health checks
│   │   ├── metrics.py           # Métricas Prometheus (Counter, Histogram, Gauge)
│   │   ├── model_loader.py      # ModelRepository protocol + LocalModelRepository
│   │   └── prediction_service.py # PredictionService + RiskClassifier (Strategy)
│   │   ├── schemas.py           # Schemas Pydantic (entrada e saída)
│   │   ├── security.py          # JWT, API Key, repositório de usuários e rate limiting
│   ├── data/
│   │   ├── preprocessing.py     # load_data(), clean_data(), split_data(), pipelines
│   │   └── schema.py            # Validação Pandera do dataset
│   │   ├── transformers.py      # OutlierClipper, TotalChargesImputer, BinaryEncoder
│   ├── features/
│   │   └── engineering.py       # Feature engineering
│   ├── models/
│   │   ├── baseline.py          # DummyClassifier, LogReg, RF, GBT + train_baseline()
│   │   ├── evaluation.py        # evaluate_model(), compute_metrics()
│   │   ├── mlp.py               # ChurnMLP (PyTorch) + EarlyStopping + MLPTrainer
│   │   └── registry.py          # Promoção no MLflow Model Registry (alias @champion)
│   ├── pipeline/
│   │   └── preprocess.py        # Estágio DVC: carga → validação → split → features
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detection.py   # KS test + PSI para monitoramento pós-deploy
│   │   └── fairness.py          # Fairlearn MetricFrame + mf.difference() por grupo sensível
│   ├── training/
│   │   └── train.py             # Pipeline de treino com MLflow (5 etapas compostas)
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuração centralizada (pydantic-settings)
│       └── logger.py            # Logging estruturado (loguru)
├── tests/
│   ├── test_api.py              # Testes da API FastAPI (com JWT)
│   ├── test_config.py           # Separação entre segredo e configuração (APP_ENV)
│   ├── test_fairness.py         # Testes de fairness (Fairlearn MetricFrame)
│   ├── test_patterns_adherence.py  # Aderência aos padrões documentados
│   ├── test_model.py            # Testes do MLP PyTorch
│   ├── test_preprocessing.py    # Testes de pré-processamento
│   ├── test_registry.py         # Testes do Model Registry (backend SQLite isolado)
│   ├── test_schema.py           # Validação do schema do dataset (Pandera)
│   └── test_smoke.py            # Smoke tests: pipeline e MLP
├── notebooks/
│   ├── 01_eda_baselines.ipynb   # EDA + baselines
│   ├── eda.ipynb                # EDA exploratório (dataset Telco)
│   └── modeling.ipynb           # Experimentos de modelagem
├── data/
│   ├── raw/                     # dataset original (versionado via DVC)
│   └── processed/               # splits.joblib (saída do estágio preprocess)
├── models/                      # artefatos gerenciados pelo DVC (fora do git)
│   ├── best_baseline.joblib     # melhor baseline sklearn (promovido no Registry)
│   ├── mlp_model.pt             # checkpoint PyTorch
│   ├── model_config.json        # input_dim e metadados do MLP
│   ├── preprocessor.joblib      # pipeline sklearn de pré-processamento
│   ├── reference_stats.npz      # distribuição de treino (referência p/ drift)
│   └── results.json             # métricas rastreadas pelo DVC
├── monitoring/                  # stack de observabilidade local (docker-compose)
│   ├── prometheus.yml           # scrape da API (/metrics)
│   └── grafana/
│       ├── Dockerfile           # imagem custom para ECS (provisioning embutido)
│       └── provisioning/
│           ├── datasources/prometheus.yml
│           └── dashboards/
│               ├── dashboard.yml
│               └── churn-api-overview.json
├── infra/
│   └── terraform/               # IaC AWS (ECS Fargate, ECR, ALB, API Gateway)
│       ├── main.tf              # VPC, ECS API, ECR, ALB, API Gateway
│       ├── observability.tf     # MLflow, Prometheus e Grafana em ECS
│       ├── outputs.tf           # URLs (API, MLflow, Prometheus, Grafana, ECR)
│       ├── provider.tf          # provider AWS + tags padrão
│       ├── variables.tf         # variáveis de ambiente e sizing
│       ├── versions.tf
│       ├── terraform.tfvars.example
│       ├── README.md
│       ├── environments/dev/terraform.tfvars.example
│       └── modules/
│           ├── network/           # VPC, subnets, security groups
│           ├── ecr/               # repositórios de imagem (API + Grafana)
│           ├── ecs_service/       # cluster e service da API
│           ├── alb/               # Application Load Balancer
│           └── api_gateway/       # HTTP API Gateway
├── docs/
│   ├── aws_terraform_deploy.md  # Guia de deploy AWS com Terraform
│   ├── deploy_architecture.md   # Arquitetura de deploy
│   └── ml_canvas.md             # ML Canvas do projeto
│   ├── model_card.md            # Model Card: performance, limitações e vieses
│   ├── monitoring_plan.md       # Plano de monitoramento
│   ├── refactoring_report.md    # Relatório de refatoração SOLID
│   ├── technical_overview.md    # Documentação técnica end-to-end + roteiro STAR
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI: lint + testes em todo PR
│       └── cd.yml               # CD: build e push Docker para GHCR
├── dvc.yaml                     # estágios do pipeline (preprocess → train)
├── dvc.lock                     # hashes de deps, outs e params de cada estágio
├── params.yaml                  # seed e hiperparâmetros centralizados (rastreados pelo DVC)
├── pyproject.toml               # dependências + config de ferramentas (single source of truth)
├── Makefile                     # atalhos (install, lint, test, train, fairness, run)
├── Dockerfile                   # imagem multi-stage para produção
├── docker-compose.yml           # stack local: API + MLflow + Prometheus + Grafana
├── .env.example                 # template de variáveis de ambiente
└── README.md
```

---

## 📚 Documentação

A documentação completa do projeto está organizada em `docs/` e no módulo de infraestrutura. Comece pela visão técnica e navegue conforme a necessidade:


| Documento                                                      | Conteúdo                                                                                | Quando consultar                                              |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `[docs/technical_overview.md](docs/technical_overview.md)`     | Visão técnica end-to-end (dados → features → modelos → API) + roteiro STAR para o vídeo | Para entender o funcionamento interno de cada módulo          |
| `[docs/model_card.md](docs/model_card.md)`                     | Model Card: arquitetura, métricas, vieses, limitações e cenários de falha               | Para avaliar performance, fairness e uso pretendido do modelo |
| `[docs/monitoring_plan.md](docs/monitoring_plan.md)`           | Plano de monitoramento: drift, fairness, alertas e playbook de incidentes               | Para operar o modelo em produção                              |
| `[docs/refactoring_report.md](docs/refactoring_report.md)`     | Relatório de refatoração SOLID + Design Patterns + proposta de microsserviços           | Para entender as decisões de arquitetura de software          |
| `[docs/aws_terraform_deploy.md](docs/aws_terraform_deploy.md)` | Guia completo de deploy na AWS com Terraform (credenciais, ECR, ECS, troubleshooting)   | Para provisionar e publicar a infraestrutura                  |
| `[infra/terraform/README.md](infra/terraform/README.md)`       | Visão geral da IaC (estrutura dos módulos e credenciais)                                | Para navegar o código Terraform                               |


---

## 🚀 Setup Rápido

### Pré-requisitos

- Python **3.12.2** (versão exata definida em `.python-version`; 3.13+ não suportado pelo torch 2.2.x)
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes
- Git
- Make (opcional, mas recomendado)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (necessário apenas para `docker compose up`)

### Instalação do uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### Instalação do projeto

```bash
git clone git@github.com:bbryttos/10-machine-learning-engineering.git
cd 10-machine-learning-engineering

# Cria o ambiente com a versão exata do Python (lê .python-version automaticamente)
uv venv --python 3.12.2

# Instala todas as dependências (runtime + treino/EDA + dev)
# --extra train: MLflow, matplotlib, pandera, openpyxl (usados no treino, não vão para a imagem Docker)
# --extra dev:   pytest, ruff, fairlearn, jupyter (ferramentas de desenvolvimento)
# Nota: `make install` instala apenas --extra dev. Para rodar make train, use o comando abaixo.
uv sync --extra dev --extra train

# Configura as variáveis de ambiente
cp .env.example .env
```

Para desenvolvimento local não é preciso preencher nada: `APP_ENV=development`
(padrão) usa segredos de placeholder e a aplicação sobe normalmente. Para rodar
via Docker, veja **Segredos e ambientes** abaixo.

### Primeira execução (ordem recomendada)

O treino usa `MLFLOW_TRACKING_URI=http://localhost:5001`.  
Por isso, suba o MLflow antes de rodar o pipeline de treino:

```bash
# Terminal 1: inicia o servidor de tracking
make mlflow-ui

# Terminal 2: executa o treinamento
make train
```

### Sem uv (alternativa com pip)

```bash
# Certifique-se de usar Python 3.12.2
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,train]"
```

### Valida a instalação

```bash
uv run python -c "from src.utils.config import settings; print('Seed:', settings.seed)"
# Saída esperada: Seed: 42
```

### Segredos e ambientes

`APP_ENV` decide se a aplicação aceita os segredos de desenvolvimento:

| Valor | Comportamento | Uso |
| --- | --- | --- |
| `development` (padrão) | Aceita os placeholders versionados (`dev-insecure-*`). Sobe sem `.env`. | Local, testes, CI |
| `production` | Recusa subir se `JWT_SECRET_KEY` ou `API_KEY` continuarem no padrão. | Docker, deploy |

Os placeholders são públicos por definição — estão no repositório. Servem apenas
para que o projeto rode sem configuração; não são segredos.

O `docker-compose.yml` define `APP_ENV=production`, então subir os containers
exige um `.env` na raiz com os dois valores preenchidos:

```bash
cp .env.example .env
printf 'JWT_SECRET_KEY=%s\nAPI_KEY=%s\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 16)" >> .env

docker compose config >/dev/null && echo "compose OK"
```

Sem isso, o compose falha antes do build, com a variável faltante nomeada. Para
verificar a proteção sem alterar seu `.env`:

```bash
APP_ENV=production \
JWT_SECRET_KEY=dev-insecure-jwt-secret-change-me \
API_KEY=dev-insecure-api-key-change-me \
uv run python -c "from src.utils.config import settings"
# ValidationError: APP_ENV=production exige segredos reais. JWT_SECRET_KEY não foi definida...
```

### Dataset

O dataset é versionado com DVC — o git rastreia apenas o ponteiro
`data/raw/Telco_customer_churn.csv.dvc`. Para recuperar o arquivo:

```bash
uv run dvc pull
```

> **Limitação conhecida:** o remote configurado é local (`../dvc-storage-fase2`),
> um caminho relativo que só existe na máquina de quem versionou os dados.
> `dvc pull` falhará para os demais integrantes e no runner de CI. Enquanto o
> remote não migrar para um storage compartilhado (ex.: S3), baixe o dataset
> [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
> manualmente para `data/raw/Telco_customer_churn.csv` e rode `uv run dvc status`
> para confirmar que o hash confere.

Sem o dataset, seis testes de `test_schema.py` são pulados silenciosamente
(`pytest -rs` mostra o motivo).

---

## ⚙️ Comandos


| Comando          | Descrição                                                                     |
| ---------------- | ----------------------------------------------------------------------------- |
| `make install`   | Instala dependências de desenvolvimento (`--extra dev`)                       |
| `make mlflow-ui` | Abre o MLflow UI em `http://localhost:5001`*                                  |
| `make train`     | Treina baselines + MLP, loga no MLflow, salva artefatos (requer MLflow ativo) |
| `uv run dvc repro` | Reproduz o pipeline completo (`preprocess` → `train`), pulando estágios sem mudança |
| `uv run dvc status` | Mostra quais estágios estão desatualizados                                 |
| `make run`       | Sobe a API FastAPI em `http://localhost:8000`                                 |
| `make test`      | Roda todos os testes com pytest                                               |
| `make lint`      | Verifica estilo com ruff                                                      |
| `make clean`     | Remove caches e artefatos temporários                                         |


> *A porta 5000 é reservada pelo AirPlay Receiver no macOS Monterey+.
> Para usar a porta 5000: **System Settings → AirDrop & Handoff → AirPlay Receiver → desligar**.
> Usuários Linux/Windows podem usar a porta 5000 normalmente.

---

## 🔁 Pipeline reprodutível (DVC)

O treino é dividido em dois estágios declarados em `dvc.yaml`. O DVC rastreia
dependências, saídas e parâmetros de cada um, e pula estágios cujas entradas
não mudaram.


| Estágio      | Comando                                | Saídas                                                        |
| ------------ | -------------------------------------- | ------------------------------------------------------------- |
| `preprocess` | `uv run python -m src.pipeline.preprocess` | `data/processed/splits.joblib`, `models/preprocessor.joblib`, `models/reference_stats.npz` |
| `train`      | `uv run python -m src.training.train`  | `models/best_baseline.joblib`, `mlp_model.pt`, `model_config.json` |


Hiperparâmetros e seed vivem em `params.yaml`, não mais hardcoded no código:

```yaml
train:
  random_state: 42
  mlp:
    hidden_dims: [128, 64, 32]
    dropout_rate: 0.3
    ...
```

```bash
# Terminal 1: MLflow precisa estar ativo — o estágio train registra runs
make mlflow-ui

# Terminal 2
uv run dvc repro          # roda apenas o que mudou
uv run dvc repro -f train # força a reexecução do treino
uv run dvc metrics show   # exibe models/results.json
```

> `models/results.json` é declarado como métrica com `cache: false`, então fica
> versionado no git e pode ser comparado entre commits com `dvc metrics diff`.

---

## 🏆 MLflow Model Registry

Ao final do treino, o **melhor baseline sklearn por F1** é logado com assinatura
inferida e promovido no Model Registry como `churn-classifier`, recebendo o alias
`@champion`.

A promoção usa **alias**, não *stages* — depreciados desde o MLflow 2.9. Uma tag
`stage=production` é gravada na versão para manter a legibilidade do estágio na UI
sem invocar a API depreciada.

```python
from src.models.registry import load_champion

model = load_champion()          # models:/churn-classifier@champion
model.predict(df_raw)            # pipeline autocontido: aceita dados crus
```

Ou diretamente pelo MLflow:

```python
import mlflow

model = mlflow.sklearn.load_model("models:/churn-classifier@champion")
```

O modelo registrado é um `Pipeline` sklearn completo — pré-processamento e
classificador —, portanto prediz a partir do CSV original sem depender do
`preprocessor.joblib`.

Cada execução do treino cria uma nova versão e move o alias para ela. Se o
Registry estiver indisponível (servidor fora do ar, backend sem banco), o módulo
loga o erro e o treino segue normalmente — a promoção é opcional, o pipeline não
quebra.

> O Registry exige backend com banco de dados. O projeto usa
> `MLFLOW_TRACKING_URI=http://localhost:5001` com SQLite; um file store puro
> não suporta `register_model`.

Consulte no MLflow UI: **Models → churn-classifier**.

---

## 🏗️ Arquitetura

```
data/raw/Telco_customer_churn.csv (DVC)
         │
    load_data() + clean_data()     # renomeia colunas, imputa, normaliza
         │
    build_full_pipeline()          # FeatureEngineer → ColumnTransformer
         │                         # num: StandardScaler / cat: OneHotEncoder
    train/val/test split
    (estratificado, seed de params.yaml)
         │
    data/processed/splits.joblib      # fim do estágio DVC `preprocess`
         │                            # início do estágio DVC `train`
    ┌────┴────────────────────────────┐
    │  Baselines (sklearn)             │  Dummy, LogReg, RF, GBT
    │  evaluation.py                   │  evaluate_model(), compute_metrics()
    │  MLflow nested runs              │  params, métricas, artefatos
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │  registry.py                     │  melhor baseline por F1
    │  log_model + register_model      │  assinatura + input_example
    │  set_registered_model_alias      │  churn-classifier@champion
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────┐
    │   ChurnMLP (PyTorch) │   [input(59) → 128 → 64 → 32 → 1]
    │   BCEWithLogitsLoss  │   BatchNorm + Dropout(0.3) + EarlyStopping
    │   + pos_weight       │   Adam + ReduceLROnPlateau
    └────┬────────────────┘
         │
      MLflow tracking (params, métricas, artefatos)
         │
    FastAPI (app.py — controller)
    ├── security.py          # JWT + API Key + InMemoryUserRepository
    ├── metrics.py           # Prometheus: 8 métricas
    ├── model_loader.py      # ModelRepository → LocalModelRepository
    ├── prediction_service.py # PredictionService + RiskClassifier
    ├── /predict             # predição individual (JWT)
    ├── /predict-apikey      # predição individual (API Key)
    └── /predict-batch       # predição em lote até 1000 (JWT)
         │
    Docker (multi-stage)     # python:3.12-slim + uv + usuário não-root
         │
    GitHub Actions CI/CD     # lint + testes + build + push GHCR
```

---

## 🧩 Padrões de Projeto

Os padrões abaixo **não foram introduzidos para cumprir requisito** — descrevem o
desenho já existente no código. Cada linha aponta o arquivo e o símbolo onde o
padrão pode ser verificado.

### Padrões clássicos (GoF)

| Padrão | Onde | O que resolve |
| --- | --- | --- |
| **Strategy** | `src/models/baseline.py` → `build_baselines()` | Família de classificadores intercambiáveis atrás da mesma interface `fit/predict`. Adicionar ou remover um baseline não altera `src/training/train.py`. |
| **Strategy** (parametrizado) | `src/api/prediction_service.py` → `RiskClassifier` | Thresholds de risco injetáveis via construtor. A regra de negócio fica isolada da inferência e testável sem carregar modelo. |
| **Factory Method** | `src/data/preprocessing.py` → `build_full_pipeline()`, `_build_preprocessor()` | Construção do pipeline centralizada em um ponto. Treino e serving obtêm exatamente a mesma topologia de transformações. |
| **Adapter** | `src/features/engineering.py` → `FeatureEngineerTransformer` | Adapta `add_features()` — função pura sobre `DataFrame` — ao protocolo `fit/transform` do scikit-learn, tornando-a plugável em `Pipeline` sem duplicar a lógica. |
| **Composite** | `sklearn.Pipeline` + estágios de `dvc.yaml` | Uma etapa isolada e uma sequência de etapas expõem a mesma interface. O `Pipeline` do champion contém pré-processamento **e** classificador, e é tratado como um único estimador. |
| **Repository** | `src/api/model_loader.py` → `ModelRepository` (Protocol) / `LocalModelRepository` | Abstrai a origem dos artefatos de modelo. A API conhece apenas o Protocol. |
| **Dependency Injection / DIP** | `src/api/app.py` → `lifespan()` e `Depends(_require_model)`; `model_loader.build_model_repository()` | A escolha da implementação concreta vive numa factory, fora do consumidor — trocar a origem dos artefatos (por exemplo, para o Model Registry) não altera `app.py`. O FastAPI resolve `PredictionService` por injeção em cada requisição. |
| **Facade** | `src/api/prediction_service.py` → `PredictionService` | Um único `predict()` encapsula transformação, inferência e classificação de risco. Os endpoints não conhecem nenhuma dessas etapas. |

### Padrões de Machine Learning

Referência: *Machine Learning Design Patterns* (Lakshmanan, Robinson & Munn, O'Reilly).

| Padrão | Onde | O que resolve |
| --- | --- | --- |
| **Heuristic Benchmark** | `DummyClassifier` em `build_baselines()` | Estabelece o piso de referência (AUC-ROC 0.5163) contra o qual todo ganho dos modelos reais é medido. Sem ele, "0.85 de AUC" é um número sem escala. |
| **Transform** | Pipelines autocontidos em `build_baselines()`; `models/preprocessor.joblib` reaplicado em `PredictionService.predict()` | A transformação é persistida junto do modelo e reaplicada idêntica em serving, eliminando *training-serving skew*. O modelo do Registry prediz diretamente sobre dados crus. |
| **Workflow Pipeline** | `dvc.yaml` (`preprocess` → `train`), `dvc.lock`, `params.yaml` | Etapas versionadas, cacheadas e reexecutáveis isoladamente. Alterar um hiperparâmetro em `params.yaml` invalida apenas o estágio afetado. |
| **Model Versioning** | `src/models/registry.py` → `log_and_register_champion()`, `load_champion()` | Versões imutáveis no Model Registry com o alias `@champion` como ponteiro móvel. Consumidores resolvem `models:/churn-classifier@champion` sem conhecer o número da versão. |
| **Rebalancing** | `class_weight="balanced"` (LogReg, RF) e `pos_weight` no `BCEWithLogitsLoss` — `src/models/mlp.py` → `train_mlp()` | Compensa o desbalanceamento da classe minoritária (churn) no cálculo da loss, em vez de reamostrar os dados. Consequência assumida: recall alto e precision menor, adequado ao caso de uso de retenção. |
| **Checkpoints** | `src/models/mlp.py` → `EarlyStopping.best_state`, restaurado ao fim de `train_mlp()` | Os pesos salvos são os da melhor época por validation loss, não os da última. Evita persistir um modelo já em sobreajuste. |
| **Continued Model Evaluation** | `src/monitoring/drift_detection.py` (KS-test, PSI); referência persistida em `models/reference_stats.npz` por `preprocess.save_reference_stats_for_drift()`; thresholds e ações em `docs/monitoring_plan.md` | A distribuição de treino é versionada como artefato do pipeline, permitindo comparar dados de produção contra a referência que originou o modelo. Fairness é verificada em `tests/test_fairness.py` (fairlearn). |
| **Stateless Serving Function** | `src/api/app.py` → `lifespan()` | Preprocessor e modelo são carregados uma vez na inicialização e o caminho de inferência não guarda estado entre requisições. Ressalva assumida: rate limiting (`security.py` → `_request_history`) e o repositório de usuários permanecem em memória do processo, o que limita a escala horizontal enquanto não forem externalizados. |

### Padrões avaliados e deliberadamente fora de escopo

| Padrão | Decisão |
| --- | --- |
| **Repeatable Splitting** | A reprodutibilidade é garantida em dois níveis: seed fixo em `params.yaml` e hash do dataset fixado no `dvc.lock`, o que reproduz o split exato a partir do artefato, não apenas do gerador aleatório. A variante canônica (hash sobre chave estável) pressupõe dados incrementais e uma chave preservada — o `customer_id` é descartado em `clean_data()` por não ser feature. |
| **Feature Store** | Não há reuso de features entre projetos nem exigência de serving em baixa latência a partir de um store. A serialização do preprocessor atende ao requisito de consistência com custo operacional muito menor. |

---

## 🔐 Autenticação da API

### JWT (usuários autenticados)

```bash
# 1. Login
curl -X POST "http://localhost:8000/auth/login?username=admin&password=admin123"
# Retorna: { "access_token": "eyJ...", "token_type": "bearer" }

# 2. Predição com token
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{...payload...}'
```

**Usuários disponíveis para teste:**


| Usuário | Senha      | Papel |
| ------- | ---------- | ----- |
| `admin` | `admin123` | admin |
| `user`  | `user123`  | user  |


> Em produção: banco de dados com senhas hasheadas e `JWT_SECRET_KEY` gerado com `openssl rand -hex 32`.

### API Key (comunicação entre serviços)

```bash
curl -X POST http://localhost:8000/predict-apikey \
  -H "X-API-Key: $API_KEY" \  # em desenvolvimento: dev-insecure-api-key-change-me
  -H "Content-Type: application/json" \
  -d '{...payload...}'
```

---

## 🌐 Endpoints da API


| Método | Endpoint          | Auth    | Descrição                                     |
| ------ | ----------------- | ------- | --------------------------------------------- |
| GET    | `/health`         | Público | Status da API e modelo                        |
| GET    | `/ready`          | Público | Readiness check (503 se modelo não carregado) |
| POST   | `/auth/login`     | Público | Login e geração de token JWT                  |
| GET    | `/auth/me`        | JWT     | Dados do usuário autenticado                  |
| POST   | `/predict`        | JWT     | Predição para um cliente                      |
| POST   | `/predict-apikey` | API Key | Predição para um cliente (serviços)           |
| POST   | `/predict-batch`  | JWT     | Predição em lote (até 1000 clientes)          |


### Exemplo de requisição e resposta

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "senior_citizen": 0, "tenure": 12, "monthly_charges": 65.5,
    "total_charges": 786.0, "gender": "Male", "partner": "Yes",
    "dependents": "No", "phone_service": "Yes", "multiple_lines": "No",
    "internet_service": "Fiber optic", "online_security": "No",
    "online_backup": "Yes", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check"
  }'
```

```json
{
  "churn_probability": 0.7422,
  "prediction": 1,
  "risk_level": "high"
}
```

### Exemplo de requisição batch

O body de `/predict-batch` é um **array JSON direto** (não um objeto com chave):

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '[
    {"senior_citizen": 0, "tenure": 12, "monthly_charges": 65.5, "total_charges": 786.0,
     "gender": "Male", "partner": "Yes", "dependents": "No", "phone_service": "Yes",
     "multiple_lines": "No", "internet_service": "Fiber optic", "online_security": "No",
     "online_backup": "Yes", "device_protection": "No", "tech_support": "No",
     "streaming_tv": "No", "streaming_movies": "No", "contract": "Month-to-month",
     "paperless_billing": "Yes", "payment_method": "Electronic check"},
    {"senior_citizen": 1, "tenure": 60, "monthly_charges": 45.0, "total_charges": 2700.0,
     "gender": "Female", "partner": "No", "dependents": "No", "phone_service": "Yes",
     "multiple_lines": "Yes", "internet_service": "DSL", "online_security": "Yes",
     "online_backup": "No", "device_protection": "Yes", "tech_support": "No",
     "streaming_tv": "Yes", "streaming_movies": "No", "contract": "Two year",
     "paperless_billing": "No", "payment_method": "Bank transfer (automatic)"}
  ]'
```

```json
{
  "predictions": [
    {"churn_probability": 0.8131, "prediction": 1, "risk_level": "high"},
    {"churn_probability": 0.1243, "prediction": 0, "risk_level": "low"}
  ],
  "count": 2
}
```

---

## 🛡️ Rate Limiting

A API limita **100 requisições por 60 segundos** por IP.
Ao exceder, retorna `429 Too Many Requests` com `retry_after` em segundos.
Headers `X-RateLimit-Limit` e `X-RateLimit-Remaining` em todas as respostas.

---

## 🔎 Observabilidade

A API expõe métricas no formato Prometheus e rastreamento por requisição:

```bash
# Verifica métricas
curl http://localhost:8000/metrics | grep churn_
```


| Métrica                            | Tipo      | Descrição                                          |
| ---------------------------------- | --------- | -------------------------------------------------- |
| `churn_predictions_total`          | Counter   | Total de predições por auth_method e risk_level    |
| `churn_prediction_latency_seconds` | Histogram | Latência das predições                             |
| `churn_request_latency_seconds`    | Histogram | Latência total das requisições                     |
| `churn_requests_total`             | Counter   | Total de requisições por método, endpoint e status |
| `churn_model_loaded`               | Gauge     | Indica se o modelo está carregado (1=sim, 0=não)   |
| `churn_login_attempts_total`       | Counter   | Tentativas de login por status (success/failed)    |
| `churn_rate_limit_hits_total`      | Counter   | Requisições bloqueadas por rate limiting           |
| `churn_prediction_probability`     | Histogram | Distribuição das probabilidades de churn preditas  |


Todas as respostas incluem os headers:

- `X-Trace-ID` — identificador único por requisição para rastreamento end-to-end
- `X-Latency-Ms` — latência da requisição em milissegundos

---

## 🐳 Docker

### Build e execução simples

```bash
docker build -t churn-prediction:latest .
docker run -p 8000:8000 \
  -e APP_ENV=production \
  -e JWT_SECRET_KEY=$(openssl rand -hex 32) \
  -e API_KEY=$(openssl rand -hex 16) \
  churn-prediction:latest
```

Com `APP_ENV=production`, a aplicação recusa iniciar se algum segredo continuar
no valor de desenvolvimento — ver **Segredos e ambientes**.

### Stack completa com MLflow, Prometheus e Grafana

```bash
# exige .env na raiz com JWT_SECRET_KEY e API_KEY preenchidos
docker compose up -d
```


| Serviço    | URL                                                             | Descrição                          |
| ---------- | --------------------------------------------------------------- | ---------------------------------- |
| API        | [http://localhost:8000/docs](http://localhost:8000/docs)        | FastAPI + Swagger UI               |
| MLflow UI  | [http://localhost:5001](http://localhost:5001)                  | Experimentos, métricas e artefatos |
| Prometheus | [http://localhost:9090](http://localhost:9090)                  | Coleta de métricas                 |
| Grafana    | [http://localhost:3000](http://localhost:3000) (admin/admin123) | Dashboards                         |


> **Nota:** A porta 5001 é usada para o MLflow para evitar conflito com o AirPlay Receiver do macOS (porta 5000).
> Os arquivos de configuração (`monitoring/prometheus.yml`, `monitoring/grafana/`) são versionados.
> Os dados gerados pelo Prometheus e Grafana (`monitoring/prometheus/data/`, `monitoring/grafana/data/`) estão no `.gitignore`.

### Performance de build

O Dockerfile usa cache do uv (`--mount=type=cache`) para otimizar builds:


| Execução                      | Tempo       |
| ----------------------------- | ----------- |
| Primeira (cache vazio)        | ~6 minutos  |
| Subsequentes (cache populado) | ~2 segundos |


> O cache é mantido localmente pelo Docker. No CI/CD o cache é gerenciado via GitHub Actions cache.

---

## 🔄 CI/CD


| Workflow | Trigger           | O que faz                        |
| -------- | ----------------- | -------------------------------- |
| `ci.yml` | Todo push e PR    | Lint (ruff) + 75 testes (pytest) |
| `cd.yml` | Merge para `main` | Build Docker + push para GHCR    |


---

## ☁️ Deploy AWS com Terraform (sem hardcode de credenciais)

Foi adicionada uma base de IaC em `infra/terraform` para provisionar o stack minimo da API:

- VPC + subnets publicas + security groups
- ECR + ECS Fargate + ALB + API Gateway HTTP
- MLflow + Prometheus + Grafana em ECS (acesso via API Gateway)
- CloudWatch Logs

Para manter o `README` enxuto, o passo a passo completo (credenciais sem hardcode, `terraform init/plan/apply`, push para ECR e validacao) ficou em:

- `docs/aws_terraform_deploy.md`
- `infra/terraform/README.md`

---

## 📊 Métricas Principais


| Modelo                        | AUC-ROC    | F1         | Precision  | Recall     |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- |
| DummyClassifier               | 0.5163     | 0.2903     | 0.2891     | 0.2914     |
| RandomForest                  | 0.8337     | 0.5514     | 0.6229     | 0.4947     |
| GradientBoosting              | 0.8555     | 0.5944     | 0.6689     | 0.5348     |
| **LogisticRegression** 🏆     | 0.8533     | **0.6205** | 0.5103     | 0.7914     |
| **MLP (PyTorch)**             | 0.8539     | **0.6299** | 0.5040     | 0.8396     |


🏆 = promovido no Model Registry como `churn-classifier@champion`.

*Valores do conjunto de teste, extraídos de `models/results.json` (ver também
`docs/model_card.md`). Execute `uv run dvc repro` para reproduzir — os seeds fixos
garantem os mesmos números.*

**Por que a LogisticRegression e não o GradientBoosting?** O critério de seleção é
F1, e por F1 a LogReg vence entre os baselines. O GBT tem accuracy e precision
superiores, mas recall bem menor (0.5348 contra 0.7914). Para retenção de clientes,
deixar de identificar quem vai cancelar custa mais do que oferecer desconto a quem
ficaria de qualquer forma. Vale notar que LogReg e RandomForest usam
`class_weight="balanced"` e o GBT não — a comparação não é perfeitamente isonômica.

A MLP supera todos os baselines em F1 e recall, mas o Registry recebe o modelo
sklearn conforme o escopo da Fase 2; a MLP permanece rastreada no MLflow Tracking.

---

## 🧪 Testes

75 testes passando, cobrindo: smoke, schema (pandera), API (JWT + API Key + batch),
model, preprocessing, fairness, Model Registry, configuração/segredos e aderência
aos padrões documentados.

```bash
make test
# ou
uv run pytest tests/ -v
```

Os testes do Registry (`tests/test_registry.py`) usam um backend SQLite temporário
por teste — exercitam a API real do MLflow sem exigir um servidor de tracking ativo,
o que os mantém executáveis em CI.

`tests/test_config.py` isola cada caso com `_env_file=None`, garantindo que a
recusa de segredos em produção seja verificada independentemente do `.env` da
máquina que roda a suíte.

> Sem o dataset em `data/raw/`, seis testes de schema são **pulados** e a suíte
> ainda reporta sucesso. Use `uv run pytest -rs` para ver os skips.

### Warnings conhecidos


| Warning                                     | Origem                       | Status                                                   |
| ------------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| `DeprecationWarning: 'crypt' is deprecated` | `passlib` (lib de terceiros) | Aguardando correção upstream                             |
| `DeprecationWarning: datetime.utcnow()`     | `src/api/security.py`        | Corrigido — substituído por `datetime.now(UTC)`          |
| `FutureWarning: import pandera as pa`       | `src/data/schema.py`         | Pendente — o caminho novo é `pandera.pandas`             |


---

## 👥 Equipe


| Nome                                | RM       | E-mail                                                                  | Papel                                |
| ----------------------------------- | -------- | ----------------------------------------------------------------------- | ------------------------------------ |
| Anna Luiza de Angelis Souza Freitas | RM375350 | [annaluizafreitas17@hotmail.com](mailto:annaluizafreitas17@hotmail.com) | Dados / Machine Learning Engineering |
| Bruno Brito de Souza                | RM374808 | [brunobrito.learning@gmail.com](mailto:brunobrito.learning@gmail.com)   | Dados / Machine Learning Engineering |
| Fellipe Resende Bastos              | RM373040 | [fbastos95@gmail.com](mailto:fbastos95@gmail.com)                       | Dados / Machine Learning Engineering |
| German Eduardo Brunner              | RM375046 | [brunner.brunner@gmail.com](mailto:brunner.brunner@gmail.com)           | Dados / Machine Learning Engineering |
| Marcelo da Cruz Salvador            | RM375166 | [macrusal@gmail.com](mailto:macrusal@gmail.com)                         | Software Engineering                 |


---

## 📐 Etapas do Projeto


| Etapa | Foco                                                                                                      | Status      |
| ----- | --------------------------------------------------------------------------------------------------------- | ----------- |
| 1     | EDA + Baselines + MLflow                                                                                  | ✅ Concluída |
| 2     | MLP PyTorch + comparação de modelos                                                                       | ✅ Concluída |
| 3     | Refatoração + FastAPI + testes + Makefile                                                                 | ✅ Concluída |
| +     | Segurança API: JWT + API Key + Rate Limiting + CORS                                                       | ✅ Concluída |
| +     | 43 testes: smoke, schema, API (JWT), model, preprocessing                                                 | ✅ Concluída |
| +     | Logging estruturado (loguru) + config centralizado                                                        | ✅ Concluída |
| +     | Validação de dados com Pandera                                                                            | ✅ Concluída |
| 4     | Model Card + README + Docker multi-stage + CI/CD GitHub Actions                                           | ✅ Concluída |
| +     | Docker: multi-stage build com uv + usuário não-root + healthcheck                                         | ✅ Concluída |
| +     | Docker: cache de build (6min → 2s na segunda execução)                                                    | ✅ Concluída |
| +     | CI: lint + testes automáticos em todo PR (GitHub Actions)                                                 | ✅ Concluída |
| +     | CD: build e push automático para GHCR no merge para main                                                  | ✅ Concluída |
| +     | Observabilidade: Prometheus /metrics + trace_id + X-Trace-ID                                              | ✅ Concluída |
| +     | Fairness: MetricFrame + mf.difference() por gender, senior_citizen e contract                             | ✅ Concluída |
| +     | Docker Compose: API + Prometheus + Grafana                                                                | ✅ Concluída |
| +     | Docstrings completas: modelos, treino e utilitários (Aula 3 — Bibliotecas Internas)                       | ✅ Concluída |
| +     | Refatoração SOLID: SRP, OCP, DIP, ISP — 6 módulos extraídos, app.py 478→130 linhas                        | ✅ Concluída |
| +     | Design Patterns: Strategy (RiskClassifier), Repository (UserRepo + ModelRepo), Facade (PredictionService) | ✅ Concluída |
| +     | Cobertura de testes: 43/43 mantidos, 70% de cobertura medida                                              | ✅ Concluída |
| 5     | Deploy AWS ECS Fargate                                                                                    | ✅ Concluída |


### Fase 2 — MLOps


| Etapa | Foco                                                                                          | Status        |
| ----- | --------------------------------------------------------------------------------------------- | ------------- |
| 1     | DVC: versionamento do dataset + pipeline em estágios (`preprocess` → `train`)                 | ✅ Concluída   |
| +     | `params.yaml`: seed e hiperparâmetros centralizados e rastreados                              | ✅ Concluída   |
| +     | Trava de build/CD sem artefatos de modelo                                                     | ✅ Concluída   |
| 2     | MLflow Model Registry: promoção do melhor baseline com alias `@champion`                      | ✅ Concluída   |
| +     | Assinatura inferida + `input_example` (colunas inteiras como float, tolerante a nulos)        | ✅ Concluída   |
| +     | 6 testes do Registry com backend SQLite isolado                                               | ✅ Concluída   |
| 3     | Dependências / `.env` / Docker                                                                | ✅ Concluída   |
| +     | `APP_ENV`: segredo separado de configuração, com recusa de placeholder em produção            | ✅ Concluída   |
| +     | Factory do repositório de modelo (`build_model_repository`) — ponto único de troca            | ✅ Concluída   |
| +     | Estatísticas de referência para detecção de drift (`models/reference_stats.npz`)              | ✅ Concluída   |
| +     | 15 testes novos (configuração e aderência aos padrões documentados)                           | ✅ Concluída   |
| 4     | README + vídeo STAR                                                                           | 🔶 Parcial     |
| +     | Seção de padrões de projeto (GoF + ML), ancorada em arquivo e símbolo                         | ✅ Concluída   |
| 5     | Ajustes de `metrics.json` e `.gitignore`                                                      | 🔲 Pendente    |
| —     | Migrar remote do DVC para storage compartilhado (desbloqueia o CD)                            | 🔲 Pendente    |


---

## 📄 Licença

MIT License