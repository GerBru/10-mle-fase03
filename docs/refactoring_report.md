# Refactoring Report — SOLID Patterns Refactor

**Branch:** `feature/solid-patterns-refactor`  
**Data:** 2026-05-31  
**Escopo:** `src/` — 8 passos, todos os 43 testes mantidos passando  

---

## Métricas de Qualidade

| Métrica | Antes | Depois |
|---|---|---|
| Testes passando | 43 / 43 | 43 / 43 |
| Cobertura geral | ~68% (estimado) | **70%** (medido) |
| Linhas em `app.py` | 478 | 130 |
| Linhas em `train.py:main()` | 113 (função única) | 25 (orquestrador) |
| Arquivos em `src/api/` | 3 | 7 |
| Arquivos em `src/models/` | 2 | 3 |
| Arquivos em `src/data/` | 3 | 4 |

> **Nota de cobertura:** os 43 testes são idênticos antes e depois. O valor pós-refatoração de 70% foi medido com `pytest --cov=src`. O valor pré-refatoração é estimado com base nos tamanhos originais dos módulos e nas mesmas suítes de teste.

---

## 1. Princípios SOLID Aplicados

### 1.1 SRP — Single Responsibility Principle

#### Violação: `src/api/app.py` (478 linhas, 5 responsabilidades)

**Antes** — Um único arquivo concentrava segurança, observabilidade, carregamento de modelo, lógica de predição e roteamento:

```python
# app.py (original) — tudo junto
JWT_SECRET_KEY = "churn-secret-key-..."
USERS_DB = {
    "admin": {"password": bcrypt.hashpw(b"admin123", ...), "role": "admin"},
}
PREDICTIONS_TOTAL = Counter("churn_predictions_total", ...)
PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", ...)
# ... 5 outros objetos Prometheus

@asynccontextmanager
async def lifespan(app):
    ckpt = torch.load(pt_path, ...)       # carregamento de artefato
    model = ChurnMLP(...)
    model.load_state_dict(ckpt["state_dict"])  # deserialização

def _risk_level(prob):                    # regra de negócio
    if prob >= 0.7: return "high"

async def predict(cliente, bundle, ...):
    X = bundle["pipeline"].transform(...)  # preprocessing
    prob = predict_proba(bundle["model"], X)[0]  # inferência
    risk = _risk_level(prob)              # regra de negócio
    PREDICTIONS_TOTAL.labels(...).inc()   # métricas
```

**Depois** — Cada responsabilidade em seu próprio módulo:

```
src/api/
├── app.py               (~130 linhas) controller: roteamento + middlewares
├── security.py          JWT, API Key, InMemoryUserRepository, rate limiting
├── metrics.py           7 objetos Prometheus
├── model_loader.py      ModelRepository protocol + LocalModelRepository
└── prediction_service.py PredictionService + RiskClassifier
```

```python
# app.py (depois) — apenas orquestra
async def predict(cliente, service: ModelState, current_user=Depends(verify_token)):
    prob, prediction, risk = service.predict(cliente)   # delega
    PREDICTIONS_TOTAL.labels(...).inc()                  # regista métrica
    return PredictionOutput(...)
```

---

#### Violação: `train.py:main()` (God Function de 113 linhas)

**Antes** — Uma função executava 7 etapas encadeadas sem separação de concerns:

```python
def main():
    # 1. carrega dados
    df_raw = load_data(DATA_PATH)
    validate_raw(df_raw)
    # 2. limpa e divide
    df = clean_data(df_raw)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(df)
    # 3. pipeline
    pipeline = build_full_pipeline()
    X_train_arr = pipeline.fit_transform(X_train_df)
    joblib.dump(pipeline, ...)
    # 4. baselines (com loop completo inline)
    for name, bl_pipeline, params in build_baselines():
        res = train_baseline(...)
        if res["metrics"]["f1"] > best_baseline_f1: ...
    # 5. MLP
    # ... 30+ linhas de MLflow + training + salvamento
    # 6. salvar artefatos
    torch.save(...)
    json.dump(...)
    # 7. gerar relatório
    for name, metrics in results.items(): ...
```

**Depois** — Cada etapa é uma função privada com assinatura e responsabilidade claras:

```python
def _load_and_validate(data_path): ...          # IO + Pandera
def _build_preprocessed_splits(df): ...         # split + fit_transform + dump
def _run_baselines(X_train_df, ...): ...        # loop + MLflow nested runs
def _train_mlp_experiment(X_train, ...): ...    # MLP + MLflow + history log
def _save_artifacts(X_train, mlp_result): ...   # torch.save + json.dump

def main():                                      # 25 linhas — só orquestra
    df_raw = _load_and_validate(DATA_PATH)
    pipeline, *splits = _build_preprocessed_splits(df_raw)
    with mlflow.start_run(run_name="churn_experiment"):
        results, _, _ = _run_baselines(...)
        mlp_result    = _train_mlp_experiment(...)
        _save_artifacts(X_train, mlp_result)
```

---

#### Violação: `src/data/preprocessing.py` (288 linhas, 3 responsabilidades)

**Antes** — Transformers sklearn conviviam com funções de I/O e constantes de formato CSV:

```python
# preprocessing.py (original)
NUMERICAL_FEATURES = ["Tenure Months", ...]   # constantes de formato
BINARY_FEATURES    = ["Senior Citizen", ...]

class OutlierClipper(...):   ...   # transformer
class TotalChargesImputer(...): ... # transformer
class BinaryEncoder(...): ...      # transformer

def load_data(path): ...           # I/O
def clean_data(df): ...            # limpeza
def build_preprocessing_pipeline(): ...  # factory
```

**Depois** — Transformers extraídos para `src/data/transformers.py`:

```python
# transformers.py — apenas transformers + constantes de formato CSV
NUMERICAL_FEATURES = ["Tenure Months", ...]
class OutlierClipper(BaseEstimator, TransformerMixin): ...
class TotalChargesImputer(BaseEstimator, TransformerMixin): ...
class BinaryEncoder(BaseEstimator, TransformerMixin): ...

# preprocessing.py — apenas I/O, limpeza e factories de pipeline
from src.data.transformers import OutlierClipper, TotalChargesImputer, BinaryEncoder
def load_data(path): ...
def clean_data(df): ...
def build_preprocessing_pipeline(): ...
def build_full_pipeline(): ...
```

---

#### Violação: `src/models/baseline.py` — métricas de avaliação acopladas a modelos

**Antes** — `evaluate_model` e `compute_metrics` viviam no mesmo arquivo das factories de modelos, com lógica quase duplicada:

```python
# baseline.py (original)
def evaluate_model(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"accuracy": ..., "roc_auc": ..., "f1": ..., "tp": tp, ...}

def compute_metrics(y_true, y_pred, y_prob=None):   # quase duplicata
    metrics = {"accuracy": accuracy_score(...), "f1": f1_score(...), ...}
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(...)     # chave diferente: auc_roc ≠ roc_auc
    return metrics
```

**Depois** — Extraídas para `src/models/evaluation.py` com `compute_metrics` delegando para `evaluate_model`:

```python
# evaluation.py
def evaluate_model(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"accuracy": ..., "roc_auc": ..., "pr_auc": ..., "f1": ...,
            "precision": ..., "recall": ..., "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    if y_prob is not None:
        base = evaluate_model(y_true, y_pred, y_prob)   # delega — sem duplicação
        return {"accuracy": base["accuracy"], "f1": base["f1"],
                "precision": base["precision"], "recall": base["recall"],
                "auc_roc": base["roc_auc"], "pr_auc": base["pr_auc"]}
    return {"accuracy": ..., "f1": ..., "precision": ..., "recall": ...}
```

---

### 1.2 OCP — Open/Closed Principle

#### Violação: `_risk_level()` com thresholds hard-coded

**Antes** — Alterar a classificação de risco exigia modificar `app.py`:

```python
def _risk_level(prob: float) -> str:   # app.py
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"
```

**Depois** — `RiskClassifier` aceita thresholds como parâmetros; `app.py` não precisa ser alterado para ajustar a lógica:

```python
@dataclass
class RiskClassifier:                  # prediction_service.py
    low_threshold: float = 0.4
    high_threshold: float = 0.7

    def classify(self, prob: float) -> str:
        if prob >= self.high_threshold: return "high"
        if prob >= self.low_threshold:  return "medium"
        return "low"

# Para mudar thresholds por ambiente/experimento:
service = PredictionService(pipeline, model, RiskClassifier(low_threshold=0.3, high_threshold=0.6))
```

---

### 1.3 DIP — Dependency Inversion Principle

#### Violação: lifespan dependia diretamente de `joblib`, `torch` e `json`

**Antes** — O controller (`app.py`) conhecia detalhes de serialização de artefatos:

```python
@asynccontextmanager
async def lifespan(app):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)  # concreto
    input_dim = ckpt.get("input_dim", input_dim)
    model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(ckpt["state_dict"])                           # concreto
    _state["pipeline"] = joblib.load(pipeline_path)                     # concreto
```

**Depois** — O lifespan depende da abstração `ModelRepository`; a implementação concreta é injetada:

```python
# model_loader.py
class ModelRepository(Protocol):       # abstração
    def load(self) -> dict: ...

class LocalModelRepository:           # implementação concreta: filesystem
    def load(self) -> dict:
        pipeline = self._load_pipeline()
        model, input_dim = self._load_model()
        return {"pipeline": pipeline, "model": model, "input_dim": input_dim}

# app.py — depende da abstração
@asynccontextmanager
async def lifespan(app):
    repo: ModelRepository = LocalModelRepository(Path(settings.models_dir))
    loaded = repo.load()               # troca para S3Repository, MLflowRepository etc.
    _state.update(loaded)
```

---

### 1.4 ISP — Interface Segregation Principle (consolidação)

**Antes** — `evaluate_model` e `compute_metrics` tinham interfaces quase idênticas mas inconsistentes, forçando o chamador a escolher entre elas:

```python
# Qual usar? Nomes de chave diferentes:
evaluate_model(y_true, y_pred, y_proba)  → "roc_auc"
compute_metrics(y_true, y_pred, y_prob)  → "auc_roc"
```

**Depois** — Interface unificada em `evaluation.py`: `compute_metrics` é o contrato público do training pipeline; `evaluate_model` é a implementação completa para análise detalhada.

---

## 2. Design Patterns Implementados

### 2.1 Strategy — Classificação de Risco

**Problema:** `_risk_level()` com thresholds embutidos tornava impossível ajustar a sensibilidade do modelo sem editar código de produção.

**Pattern:** Strategy — comportamento (classificação de risco) parametrizado e substituível sem alterar o contexto de uso.

**Solução:**

```python
# prediction_service.py
@dataclass
class RiskClassifier:
    low_threshold: float = 0.4
    high_threshold: float = 0.7

    def classify(self, prob: float) -> str: ...

# Contexto (PredictionService) recebe a estratégia via injeção
class PredictionService:
    def __init__(self, pipeline, model, risk_classifier: RiskClassifier | None = None):
        self.risk_classifier = risk_classifier or RiskClassifier()

# Chamador pode substituir a estratégia sem alterar PredictionService:
conservative = PredictionService(p, m, RiskClassifier(low_threshold=0.2, high_threshold=0.5))
aggressive   = PredictionService(p, m, RiskClassifier(low_threshold=0.5, high_threshold=0.8))
```

---

### 2.2 Repository — Acesso a Dados (usuários e artefatos de modelo)

**Problema (usuários):** `USERS_DB` era um dict literal em `app.py`. Trocar por banco de dados ou LDAP exigia reescrita do controller.

**Problema (modelo):** A lógica de `torch.load` e `joblib.load` estava misturada ao lifespan da API.

**Pattern:** Repository — encapsula o acesso à fonte de dados atrás de uma interface estável.

**Solução — usuários:**

```python
# security.py
class InMemoryUserRepository:
    def __init__(self):
        self._db = {
            "admin": {"password": bcrypt.hashpw(b"admin123", ...), "role": "admin"},
        }

    def get(self, username: str) -> dict | None:
        return self._db.get(username)

    def authenticate(self, username: str, password: str) -> dict | None:
        user = self.get(username)
        if user and bcrypt.checkpw(password.encode(), user["password"]):
            return user
        return None

# app.py — usa o repositório, não o dict diretamente
user_repo = InMemoryUserRepository()
user = user_repo.authenticate(username, password)
```

**Solução — artefatos de modelo:**

```python
# model_loader.py
class ModelRepository(Protocol):
    def load(self) -> dict: ...     # contrato independente de storage

class LocalModelRepository:        # hoje: filesystem
    def load(self) -> dict: ...     # amanhã: S3Repository, MLflowRepository

# Para migrar para S3 basta criar S3ModelRepository implementando o Protocol
# sem tocar em app.py
```

---

### 2.3 Facade (Service Layer) — Predição

**Problema:** A lógica completa de predição (conversão de schema, preprocessing, inferência, classificação de risco) estava duplicada em três endpoints (`/predict`, `/predict-apikey`, `/predict-batch`), totalizando ~80 linhas repetidas.

**Pattern:** Facade / Service Layer — expõe uma interface simplificada sobre um subsistema complexo.

**Solução:**

```python
# prediction_service.py
class PredictionService:
    def predict(self, cliente: ClienteInput) -> tuple[float, int, str]:
        X = self.pipeline.transform(_to_dataframe(cliente)).astype(np.float32)
        prob = float(predict_proba(self.model, X)[0])
        return prob, int(prob >= 0.5), self.risk_classifier.classify(prob)

    def predict_batch(self, clientes: list[ClienteInput]) -> list[PredictionOutput]:
        df = pd.concat([_to_dataframe(c) for c in clientes], ignore_index=True)
        probs = predict_proba(self.model, self.pipeline.transform(df).astype(np.float32))
        return [PredictionOutput(...) for p in probs]

# app.py — cada endpoint reduzido a 4 linhas:
async def predict(cliente, service: ModelState, current_user=Depends(verify_token)):
    prob, prediction, risk = service.predict(cliente)
    PREDICTIONS_TOTAL.labels(auth_method="jwt", risk_level=risk).inc()
    return PredictionOutput(churn_probability=round(prob, 4), ...)
```

---

### 2.4 Template Method — Pipeline de Treinamento

**Problema:** `main()` era um script procedural de 113 linhas. Qualquer variação (pular validação em dev, substituir o modelo, mudar o storage) exigia modificar o orquestrador.

**Pattern:** Template Method — define o esqueleto do algoritmo (sequência invariante) e permite substituir etapas individuais.

**Solução:** As 5 funções privadas definem o contrato; `main()` define a sequência. Cada etapa pode ser sobrescrita ou mockada independentemente:

```
main()
 ├── _load_and_validate()       ← substituível por versão sem validação
 ├── _build_preprocessed_splits() ← substituível por splits de dados alternativos
 ├── _run_baselines()           ← substituível por subset de modelos
 ├── _train_mlp_experiment()    ← substituível por outro arquitetura
 └── _save_artifacts()          ← substituível por S3/MLflow registry
```

---

### 2.5 Protocol (DIP como padrão estrutural)

**Problema:** `app.py` dependia de `torch`, `joblib`, `json` — detalhes de implementação que impediam trocar o mecanismo de storage sem alterar o controller.

**Pattern:** Protocol (Python typing) — define contratos por duck typing, sem herança obrigatória.

**Solução:**

```python
# model_loader.py
from typing import Protocol

class ModelRepository(Protocol):
    def load(self) -> dict: ...   # qualquer classe com load() satisfaz o contrato

# app.py usa a anotação de tipo, não a implementação concreta:
repo: ModelRepository = LocalModelRepository(Path(settings.models_dir))
```

---

## 3. Estrutura de Microsserviços Proposta

O monolito já possui separação lógica suficiente para extração em 3 serviços independentes. A refatoração completou os pré-requisitos estruturais de cada boundary.

```
┌─────────────────────────────────────────────────────────┐
│                    MONOLITO ATUAL                        │
│                                                          │
│   src/api/          src/training/      src/monitoring/   │
│   (serving)         (batch job)        (analytics)       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼  extração
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│  Inference   │  │  Training        │  │  Monitoring     │
│  Service     │  │  Service         │  │  Service        │
│              │  │                  │  │                  │
│ FastAPI      │  │ Batch job        │  │ REST API         │
│ /predict     │  │ mlflow           │  │ /drift           │
│ /predict-*   │  │ sklearn + torch  │  │ /stats           │
│ /health      │  │ → artefatos      │  │                  │
│ /metrics     │  │   para S3/       │  │ KS test + PSI   │
│              │  │   MLflow Registry│  │ sem deps src/   │
└──────┬───────┘  └────────┬─────────┘  └────────┬────────┘
       │                   │                      │
       │  lê artefatos     │  grava artefatos      │  lê stats
       └───────────────────┴──────────────────────┘
                  Model Registry / Object Storage
```

### MS-1: Inference Service (`src/api/`)

| Item | Detalhe |
|---|---|
| **Entrypoint** | `src.api.app:app` |
| **Pré-requisito cumprido** | `ModelRepository` Protocol — basta implementar `S3ModelRepository` ou `MLflowModelRepository` |
| **Escala** | horizontal independente de training |
| **Bloqueio restante** | substituir `LocalModelRepository` por implementação de object storage |

### MS-2: Training Service (`src/training/` + `src/models/` + `src/data/`)

| Item | Detalhe |
|---|---|
| **Entrypoint** | `python -m src.training.train` |
| **Pré-requisito cumprido** | `_save_artifacts()` isolada — só ela precisa ser alterada para gravar em S3/MLflow |
| **Padrão natural** | batch job acionado via API REST ou fila (Celery, Temporal, AWS Step Functions) |
| **Bloqueio restante** | interface de saída de artefatos (hoje `joblib.dump`/`torch.save` local) |

### MS-3: Monitoring Service (`src/monitoring/`)

| Item | Detalhe |
|---|---|
| **Acoplamento externo** | **zero** — nenhum import de outros módulos `src/` |
| **Pré-requisito cumprido** | funções puras: `ks_test()`, `psi()`, `analyze_drift()` já são stateless |
| **Bloqueio restante** | expor via HTTP (hoje são funções chamadas diretamente); adicionar `POST /drift/analyze` |
| **Implantação** | serviço mais simples de extrair — basta envolver as funções existentes num router FastAPI |

---

## 4. Cobertura por Módulo (pós-refatoração)

```
Name                                Stmts   Miss  Cover
-------------------------------------------------------
src/api/app.py                        130     18    86%
src/api/metrics.py                      9      0   100%
src/api/model_loader.py                50      9    82%
src/api/prediction_service.py          33      2    94%
src/api/schemas.py                     34      0   100%
src/api/security.py                    57      5    91%
src/data/preprocessing.py              73      3    96%
src/data/transformers.py               43      2    95%
src/features/engineering.py            28      0   100%
src/models/baseline.py                 40     24    40%  *
src/models/evaluation.py               10      4    60%
src/models/mlp.py                     104      2    98%
src/monitoring/drift_detection.py      53     53     0%  †
src/training/train.py                 107    107     0%  †
src/utils/config.py                    33      6    82%
src/utils/logger.py                    11      0   100%
-------------------------------------------------------
TOTAL                                 832    249    70%
```

> `*` `baseline.py` — 40% porque `train_baseline()` requer MLflow ativo; não é coberta por testes unitários.  
> `†` `train.py` e `drift_detection.py` — 0% por design: ambos são executados como scripts/jobs, não como módulos importados nos testes.

---

## 5. Arquivos Criados / Modificados

### Novos (6 arquivos)

| Arquivo | Conteúdo | SOLID / Pattern |
|---|---|---|
| `src/api/security.py` | JWT, API Key, `InMemoryUserRepository`, rate limit state | SRP + Repository |
| `src/api/metrics.py` | 7 objetos Prometheus | SRP |
| `src/api/model_loader.py` | `ModelRepository` Protocol + `LocalModelRepository` | SRP + DIP + Repository |
| `src/api/prediction_service.py` | `PredictionService` + `RiskClassifier` | SRP + Strategy + Facade |
| `src/data/transformers.py` | `OutlierClipper`, `TotalChargesImputer`, `BinaryEncoder` | SRP |
| `src/models/evaluation.py` | `evaluate_model`, `compute_metrics` (sem duplicação) | SRP + ISP |

### Modificados (7 arquivos)

| Arquivo | O que mudou |
|---|---|
| `src/api/app.py` | 478 → 130 linhas; delegação para 4 módulos especializados |
| `src/data/preprocessing.py` | removidos transformers e constantes CSV; `build_preprocessor` → `_build_preprocessor` (privada) |
| `src/models/baseline.py` | removidas `evaluate_model` e `compute_metrics`; importa de `evaluation.py` |
| `src/training/train.py` | `main()` decomposta em 5 funções privadas; `train_mlp` renomeada para `_train_mlp_experiment` |
| `tests/test_api.py` | `create_access_token` importada de `src.api.security` |
| `tests/test_smoke.py` | `evaluate_model` importada de `src.models.evaluation` |

### Inalterados (por decisão)

| Arquivo | Motivo |
|---|---|
| `src/models/mlp.py` | ISP (`MLPTrainer` une treino+inferência) — adiado: todos os testes de modelo dependem dessa interface |
| `src/data/schema.py` | sem violações identificadas |
| `src/api/schemas.py` | sem violações identificadas |
| `src/monitoring/drift_detection.py` | sem dependências internas — candidato a MS-3 sem mudanças |
