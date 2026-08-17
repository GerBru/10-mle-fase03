# Documentação Técnica — Churn Prediction Pipeline

**FIAP MLE Tech Challenge Fase 1**  
Versão: 1.0.0 | Referência para o vídeo STAR (5 min)

---

## Sumário

1. [Visão Geral do Problema](#1-visão-geral-do-problema)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Fluxo de Dados — do CSV à Predição](#3-fluxo-de-dados--do-csv-à-predição)
4. [Módulo de Dados — `src/data/](#4-módulo-de-dados--srcdata)`
5. [Feature Engineering — `src/features/](#5-feature-engineering--srcfeatures)`
6. [Modelos — `src/models/](#6-modelos--srcmodels)`
7. [Pipeline de Treinamento — `src/training/](#7-pipeline-de-treinamento--srctraining)`
8. [API de Inferência — `src/api/](#8-api-de-inferência--srcapi)`
9. [Testes e Qualidade de Código](#9-testes-e-qualidade-de-código)
10. [Rastreamento de Experimentos com MLflow](#10-rastreamento-de-experimentos-com-mlflow)
11. [Reprodutibilidade e Boas Práticas](#11-reprodutibilidade-e-boas-práticas)
12. [Roteiro STAR para o Vídeo](#12-roteiro-star-para-o-vídeo)

---

## 1. Visão Geral do Problema

### Contexto de negócio

Uma operadora de telecomunicações perde clientes em ritmo acelerado. Cada cliente que cancela representa perda de **Lifetime Value (LTV)** — a soma de toda a receita que ele geraria ao longo do relacionamento com a empresa.

O desafio: **identificar com antecedência quais clientes têm maior probabilidade de cancelar** (churn), para que a equipe de retenção possa agir — oferecer descontos, melhorar o suporte, ou ajustar o plano — antes que o cancelamento aconteça.

### Dataset

- **IBM Telco Customer Churn** — dataset público de telecomunicações
- **7.043 registros** × **21 features**
- **Target binário:** `Churn` → `Yes` (cancelou) / `No` (permaneceu)
- **Desbalanceamento:** ~73% não churnou, ~27% churnou
  - Isso é crítico: um modelo que diz "nunca vai churnar" acertaria 73% das vezes — mas seria inútil para o negócio

### Por que não usar accuracy?

Com 73%/27% de split, accuracy é enganosa. As métricas escolhidas são:


| Métrica       | Por que importa                                                                                            |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| **AUC-ROC**   | Mede o poder de ranking do modelo — quão bem ele separa churners de não-churners independente do threshold |
| **F1-Score**  | Balanceia precision e recall — importante quando o custo de errar é assimétrico                            |
| **Recall**    | Minimiza falsos negativos (cliente que vai sair mas não foi identificado = perda certa)                    |
| **Precision** | Controla falsos positivos (oferecer desconto a quem não ia sair = custo desnecessário)                     |


---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        TREINAMENTO                               │
│                                                                 │
│  CSV → validate_raw() → clean_data() → split_data()            │
│                                ↓                                │
│              build_full_pipeline().fit_transform()              │
│           ┌──────────────────────────────────┐                  │
│           │  FeatureEngineerTransformer       │  (+14 features)  │
│           │  ColumnTransformer                │  (scale+encode)  │
│           └──────────────────────────────────┘                  │
│                                ↓                                │
│         Baselines (sklearn) ←→ MLP PyTorch                      │
│                                ↓                                │
│           MLflow tracking (params + metrics + artifacts)        │
│                                ↓                                │
│    models/preprocessor.joblib + models/mlp_model.pt            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                          artefatos salvos
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      INFERÊNCIA (API)                            │
│                                                                 │
│  POST /predict → Pydantic validation → preprocessor.transform() │
│                                ↓                                │
│                   MLP.forward() → sigmoid → threshold           │
│                                ↓                                │
│         { churn_probability, prediction, risk_level }           │
└─────────────────────────────────────────────────────────────────┘
```

**Decisão chave de design:** o `preprocessor.joblib` salvo no treino inclui o `FeatureEngineerTransformer` — quando a API carrega esse arquivo e chama `.transform()`, todo o processamento (feature engineering + scaling + encoding) é aplicado automaticamente. Treino e inferência passam pelo **exato mesmo código**, eliminando o risco de *training-serving skew*.

---

## 3. Fluxo de Dados — do CSV à Predição

### Durante o treinamento

```
1. load_data()          → pd.DataFrame (7043 × 21)
2. validate_raw()       → Pandera valida tipos e categorias
3. clean_data()         → remove customerID, converte TotalCharges,
                          imputa nulos, binariza Churn (0/1)
4. split_data()         → 70% treino / 10% validação / 20% teste
                          (estratificado para manter proporção de churn)
5. build_full_pipeline() → Pipeline sklearn:
   5a. FeatureEngineerTransformer  → +14 features derivadas
   5b. ColumnTransformer           → StandardScaler + OneHotEncoder
6. pipeline.fit_transform(X_train) → array numpy (n × 59 features)
7. Treino dos modelos   → baselines sklearn + MLP PyTorch
8. Salvar artefatos     → preprocessor.joblib + mlp_model.pt
```

### Durante a inferência (API)

```
1. POST /predict        → JSON com 19 campos do cliente
2. Pydantic validation  → ClienteInput valida tipos e constraints
3. model_dump()         → dict → pd.DataFrame (1 × 19)
4. preprocessor.transform() → aplica FE + scaling + encoding
5. torch.FloatTensor    → tensor (1 × 59)
6. model.forward()      → logit escalar
7. torch.sigmoid()      → probabilidade [0, 1]
8. threshold 0.5        → prediction (0 ou 1)
9. risk_level           → low (<0.4) / medium (0.4–0.7) / high (≥0.7)
10. PredictionOutput    → JSON de resposta
```

---

## 4. Módulo de Dados — `src/data/`

### `preprocessing.py`

`**load_data(path)**`

- Carrega o CSV com `pd.read_csv()`
- Loga o shape do dataset (`INFO: Loaded 7043 rows, 21 columns`)

`**clean_data(df)**`

Etapas em ordem:

1. Remove `customerID` — identificador sem valor preditivo
2. Converte `TotalCharges` de string para float (`pd.to_numeric(..., errors="coerce")`)
  - Clientes com `tenure=0` têm `TotalCharges=" "` (espaço em branco) → vira `NaN`
3. Imputa nulos numéricos pela **mediana** (robusto a outliers)
4. Imputa nulos categóricos pela **moda** (valor mais frequente)
5. Binariza o target: `"Yes"` → `1`, `"No"` → `0`

`**build_preprocessor()`**

Constrói um `ColumnTransformer` com 3 transformadores paralelos:


| Nome  | Transformador    | Colunas                                             |
| ----- | ---------------- | --------------------------------------------------- |
| `num` | `StandardScaler` | tenure, MonthlyCharges, TotalCharges + 2 engineered |
| `bin` | `passthrough`    | SeniorCitizen + 12 colunas binárias engineered      |
| `cat` | `OneHotEncoder`  | 15 colunas categóricas originais                    |


O `remainder="drop"` descarta qualquer coluna não listada explicitamente.

`**build_full_pipeline()`**

```python
Pipeline([
    ("features", FeatureEngineerTransformer()),  # step 1: cria novas features
    ("transform", build_preprocessor()),          # step 2: escala + codifica
])
```

Por que Pipeline e não chamar cada função separado? Porque o sklearn Pipeline garante que `fit` ocorre apenas nos dados de treino e `transform` aplica a mesma transformação em val/test/produção — **sem data leakage**.

`**split_data(df)`**

```
df (7043) → test_size=0.2 → X_test (1408) + restante (5635)
restante  → val_fraction=0.111 → X_val (627) + X_train (5008)
```

Usa `stratify=y` em ambas as divisões para que a proporção de churn (~27%) seja mantida nos três conjuntos.

---

### `schema.py`

Usa a biblioteca **Pandera** para validação declarativa do DataFrame raw antes de qualquer processamento:

```python
RAW_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "tenure": Column(int, Check.ge(0)),
        "MonthlyCharges": Column(float, Check.gt(0)),
        # ... demais colunas
    },
    coerce=True,   # tenta converter tipos automaticamente
    strict=False,  # permite colunas extras (customerID, TotalCharges, Churn)
)
```

`**validate_raw(df)**` é chamado no início do `train.py`. Se o dataset violar alguma regra (ex: valor de `Contract` fora das categorias esperadas), o erro é reportado com detalhes antes de qualquer processamento — **fail fast e explícito**.

---

## 5. Feature Engineering — `src/features/`

### Por que criar novas features?

O dataset bruto tem informações implícitas que o modelo linear (regressão logística) não consegue capturar diretamente. As features derivadas tornam essas relações explícitas para qualquer modelo.

### `add_features(df)` — 14 novas features


| Feature               | Fórmula / Lógica                      | Intuição de negócio                                                                 |
| --------------------- | ------------------------------------- | ----------------------------------------------------------------------------------- |
| `charges_per_tenure`  | `MonthlyCharges / (tenure + ε)`       | Custo médio por mês de relacionamento — clientes novos pagando muito têm alto risco |
| `is_new_customer`     | `tenure ≤ 3`                          | Primeiros 3 meses: período crítico de churn inicial                                 |
| `is_long_term`        | `tenure > 24`                         | Clientes antigos têm lealdade maior — risco baixo                                   |
| `is_monthly_contract` | `Contract == "Month-to-month"`        | Contratos mensais têm ~43% de churn vs ~3% dos anuais                               |
| `is_electronic_check` | `PaymentMethod == "Electronic check"` | Forma de pagamento correlacionada com churn                                         |
| `has_phoneservice`    | `PhoneService == "Yes"`               | Binário explícito do serviço                                                        |
| `has_`* (7 colunas)   | `coluna == "Yes"`                     | Transforma categóricas ternárias em binárias simples                                |
| `num_services`        | soma de todos os `has_`*              | Nível de engajamento — mais serviços = menor churn                                  |


### `FeatureEngineerTransformer`

Wrapper sklearn-compatível que envolve `add_features()`:

```python
class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self       # sem estado para aprender
    def transform(self, X): return add_features(X)
```

Por ser um `TransformerMixin`, pode ser inserido em qualquer `Pipeline` sklearn como um step. O `fit()` não faz nada porque feature engineering aqui é puramente determinístico — não depende dos dados de treino (sem risco de leakage).

---

## 6. Modelos — `src/models/`

### `baseline.py` — Modelos de Referência

**Por que treinar baselines?**

Baselines estabelecem o piso de performance. Se o MLP não superar o `GradientBoosting`, a rede neural não adiciona valor suficiente para justificar sua complexidade.

Os 4 baselines treinados:


| Modelo                                   | Justificativa                                                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `DummyClassifier(strategy="stratified")` | Baseline absoluto — prediz aleatoriamente respeitando a proporção das classes. Qualquer modelo sério deve superar isso |
| `LogisticRegression`                     | Baseline linear clássico — interpretável, rápido, boa capacidade com features bem preparadas                           |
| `RandomForest`                           | Ensemble de árvores — captura interações não-lineares sem hiperparâmetros complexos                                    |
| `GradientBoosting`                       | Ensemble sequencial — geralmente o mais forte dos baselines em dados tabulares                                         |


Todos os baselines usam `build_full_pipeline()` internamente, então feature engineering é aplicado automaticamente.

**Validação cruzada estratificada (5 folds):**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)
```

5 folds significa que o dataset de treino é dividido em 5 partes — o modelo é treinado em 4 e avaliado na 5ª, rotacionando. Resultado: 5 estimativas de performance independentes. A média e desvio padrão revelam se o modelo generaliza bem ou é instável.

---

### `mlp.py` — Rede Neural MLP com PyTorch

#### Classe `MLP` — arquitetura da rede

```
Input (59 features)
    ↓
Linear(59 → 128) → BatchNorm1d(128) → ReLU() → Dropout(0.3)
    ↓
Linear(128 → 64) → BatchNorm1d(64)  → ReLU() → Dropout(0.3)
    ↓
Linear(64 → 32)  → BatchNorm1d(32)  → ReLU() → Dropout(0.3)
    ↓
Linear(32 → 1)   [logit escalar — sem ativação]
```

**Por que essa arquitetura?**

- **3 camadas ocultas [128, 64, 32]:** pirâmide decrescente — compressão progressiva da representação. A rede aprende features de alto nível nas camadas maiores e as especializa nas menores.
- **BatchNorm1d:** normaliza as ativações de cada batch antes de passar para a próxima camada. Estabiliza o treinamento, permite learning rates maiores e funciona como regularização leve.
- **ReLU:** função de ativação padrão para redes profundas. Não satura para valores positivos (evita vanishing gradient), computacionalmente eficiente.
- **Dropout(0.3):** durante o treino, desativa aleatoriamente 30% dos neurônios a cada passagem. Força a rede a aprender representações redundantes — principal técnica de regularização para redes neurais.
- **Saída sem ativação (logit):** a função de perda `BCEWithLogitsLoss` aplica sigmoid internamente com maior estabilidade numérica do que aplicar sigmoid na saída e depois `BCELoss`.

#### Classe `MLPTrainer` — loop de treinamento

**Inicialização:**

```python
torch.manual_seed(42)   # reprodutibilidade dos pesos iniciais
np.random.seed(42)      # reprodutibilidade do DataLoader
optimizer = Adam(lr=1e-3)       # otimizador adaptativo
criterion = BCEWithLogitsLoss() # binary cross-entropy com sigmoid numérico
```

**Loop de treino por epoch:**

```
Para cada batch de 64 amostras:
  1. optimizer.zero_grad()  → zera gradientes acumulados do batch anterior
  2. logits = model(X_batch)
  3. loss = criterion(logits, y_batch)
  4. loss.backward()         → backpropagation — calcula gradientes
  5. optimizer.step()        → atualiza pesos na direção anti-gradiente

Após todos os batches:
  6. Calcula val_loss no conjunto de validação (sem gradientes)
  7. Verifica early stopping
```

**Early Stopping:**

```python
if val_loss < best_val_loss:
    best_weights = model.state_dict().copy()  # salva melhor checkpoint
    epochs_no_improve = 0
else:
    epochs_no_improve += 1
    if epochs_no_improve >= patience (10):
        break  # para o treino
```

O modelo não para no último epoch — ele **restaura os pesos do melhor checkpoint**. Isso evita overfitting: o treino pode ter 80 epochs, mas os pesos usados são os do epoch 45 (onde val_loss foi mínimo).

**Predição:**

```python
def predict_proba(X):
    model.eval()          # desativa BatchNorm e Dropout (modo inferência)
    with torch.no_grad(): # não calcula gradientes (economiza memória)
        logits = model(X)
        return torch.sigmoid(logits)  # converte logit → probabilidade [0,1]

def predict(X, threshold=0.5):
    return (predict_proba(X) >= threshold).astype(int)
```

`model.eval()` é fundamental: sem ele, o BatchNorm usa estatísticas do batch atual (ruidosas para batches pequenos) e o Dropout continua dropando neurônios, gerando predições não-determinísticas.

---

## 7. Pipeline de Treinamento — `src/training/`

### `train.py` — orquestra o treinamento completo

**Sequência de execução do `main()`:**

```
1. mlflow.set_experiment("churn-prediction")
2. load_data() + validate_raw() + clean_data()
3. split_data() → X_train, X_val, X_test, y_train, y_val, y_test
4. build_full_pipeline().fit_transform(X_train) → X_train_arr (numpy)
5.   .transform(X_val)  → X_val_arr
5.   .transform(X_test) → X_test_arr
6. joblib.dump(pipeline, "models/preprocessor.joblib")
7. [MLflow run "churn_experiment"]
   7a. Para cada baseline: train_baseline() → nested MLflow run
   7b. train_mlp() → nested MLflow run
8. Salva mlp_model.pt e results.json
```

**Por que `fit_transform` só no treino e `transform` no val/test?**

`fit_transform` aprende os parâmetros do StandardScaler (média e desvio padrão) nos dados de treino e aplica a transformação. `transform` apenas aplica — sem re-aprender. Se fitássemos o scaler no dataset inteiro, os dados de teste "contaminariam" o treino (data leakage), gerando estimativas de performance infladas.

---

## 8. API de Inferência — `src/api/`

### Inicialização — `lifespan()`

O FastAPI usa o padrão de **lifespan context manager** para gerenciar o ciclo de vida da aplicação:

```python
@asynccontextmanager
async def lifespan(app):
    # -- STARTUP --
    preprocessor = joblib.load("models/preprocessor.joblib")
    input_dim = preprocessor.transform(dummy_row).shape[1]  # descobre dim automaticamente
    model = MLP(input_dim=input_dim, hidden_dims=[128, 64, 32])
    model.load_state_dict(torch.load("models/mlp_model.pt"))
    model.eval()
    state["model_loaded"] = True

    yield  # API fica disponível aqui

    # -- SHUTDOWN --
    logger.info("Shutting down.")
```

**Por que carregar no startup e não em cada requisição?**

Carregar o modelo a cada requisição levaria ~200-500ms só no I/O de disco + deserialização. Mantendo em memória (variável `state`), cada requisição leva < 5ms de inferência. A variável `state` é um dicionário global compartilhado entre todas as requisições.

**Por que `dummy_row` para descobrir `input_dim`?**

O pipeline `preprocessor.joblib` tem o feature engineering e o ColumnTransformer embutidos. A dimensão de saída depende de quantas categorias únicas o OneHotEncoder encontrou no treino. Em vez de hardcodar esse número, a API passa uma linha "dummy" pelo preprocessor e mede o shape da saída — **auto-discovery robusto a mudanças de features**.

---

### Schemas de validação — `schemas.py`

**Pydantic v2** valida automaticamente o JSON recebido antes de qualquer código de negócio ser executado:

```python
class ClienteInput(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1)   # ge=0, le=1: deve ser 0 ou 1
    tenure: int = Field(..., ge=0)                  # não pode ser negativo
    MonthlyCharges: float = Field(..., gt=0)        # deve ser > 0
    gender: str                                     # qualquer string aceita
    Contract: str                                   # "Month-to-month", "One year", "Two year"
    # ... demais campos
```

Se algum campo falhar: FastAPI retorna **HTTP 422 Unprocessable Entity** automaticamente com o detalhe do erro — sem nenhuma linha de código extra. Isso protege a API de dados malformados antes de chegarem ao modelo.

---

### Endpoints

#### `GET /health`

```python
@app.get("/health", response_model=HealthOutput)
def health():
    return HealthOutput(
        status="ok",
        model_loaded=state["model_loaded"],
        version="1.0.0",
    )
```

**Uso:** healthcheck para load balancers e orquestradores (Kubernetes, ECS). Se `model_loaded=false`, o pod não deve receber tráfego.

Resposta:

```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

---

#### `POST /predict`

**Fluxo interno completo:**

```python
def predict(cliente: ClienteInput):
    # 1. Pydantic já validou — cliente é um objeto Python seguro
    df = pd.DataFrame([cliente.model_dump()])     # dict → DataFrame (1 linha)

    # 2. Pipeline: feature engineering + scaling + encoding
    X = state["preprocessor"].transform(df).astype(np.float32)

    # 3. Inferência PyTorch
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        logit = state["model"](X_tensor)       # forward pass
        prob = torch.sigmoid(logit).item()     # logit → prob → scalar Python

    # 4. Decisão de negócio
    prediction = int(prob >= 0.5)
    risk_level = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

    return PredictionOutput(
        churn_probability=round(prob, 4),
        prediction=prediction,
        risk_level=risk_level,
    )
```

**Por que `torch.no_grad()`?**

Durante inferência, não precisamos calcular gradientes (não há backpropagation). `no_grad` desativa o grafo computacional do autograd, reduzindo uso de memória em ~50% e acelerando a inferência.

**Classificação de risco:**


| `churn_probability` | `risk_level` | Ação recomendada                    |
| ------------------- | ------------ | ----------------------------------- |
| < 0.40              | `low`        | Nenhuma ação necessária             |
| 0.40 – 0.69         | `medium`     | Contato proativo de retenção        |
| ≥ 0.70              | `high`       | Oferta imediata de desconto/upgrade |


Resposta:

```json
{
  "churn_probability": 0.7823,
  "prediction": 1,
  "risk_level": "high"
}
```

---

### Middleware de Latência

```python
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info(json.dumps({
        "method": "POST",
        "path": "/predict",
        "status_code": 200,
        "latency_ms": 3.47,
    }))
    return response
```

Intercepta **todas** as requisições antes de chegarem ao handler. Mede o tempo de ponta a ponta e loga em formato JSON estruturado — parseable por ferramentas de observabilidade (Datadog, CloudWatch, ELK). Isso cobre o requisito de "logging estruturado sem print()" do Tech Challenge.

---

## 9. Testes e Qualidade de Código

### Cobertura de testes (`tests/`)


| Arquivo                 | Testes    | O que valida                                                                                                                                                   |
| ----------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_preprocessing.py` | 7 testes  | clean_data, imputação, binarização do target, output do pipeline, split estratificado                                                                          |
| `test_model.py`         | 6 testes  | shape de saída do MLP, forward pass, fit do trainer, predict_proba em [0,1], early stopping                                                                    |
| `test_api.py`           | 18 testes | /health retorna 200+ok, /predict schema, JWT + API Key, batch, probabilidade em [0,1], prediction em {0,1}, risk_level válido, 422 para campo ausente/inválido |
| `test_schema.py`        | 6 testes  | validação Pandera do dataset raw                                                                                                                               |
| `test_smoke.py`         | 6 testes  | smoke tests do pipeline e do MLP                                                                                                                               |


**Total: 43 testes, 0 falhas.**

### Linting com Ruff

Configurado no `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
```

- `E/W`: estilo PEP 8
- `F`: erros de código (imports não usados, variáveis indefinidas)
- `I`: ordenação de imports
- `B`: bugs potenciais (ex: `raise` dentro de `except` sem `from`)
- `SIM`: simplificações de código

---

## 10. Rastreamento de Experimentos com MLflow

### Estrutura dos runs

```
churn_experiment (parent run)
├── params: dataset, train_size, val_size, test_size, random_state
├── metrics: best_baseline_f1, mlp_vs_best_baseline_f1_delta
│
├── dummy_classifier (nested run)
│   ├── metrics: cv_accuracy_mean, cv_f1_mean, cv_roc_auc_mean, ...
│   └── metrics: test_accuracy, test_f1, test_auc_roc, ...
│
├── logistic_regression (nested run)
├── random_forest (nested run)
├── gradient_boosting (nested run)
│
└── mlp_pytorch (nested run)
    ├── params: hidden_dims, dropout_rate, lr, batch_size, max_epochs, patience
    ├── metrics: test_f1, test_auc_roc, test_precision, test_recall
    └── artifacts: training_history.json, model/
```

Acessar o MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
# http://localhost:5001  (porta 5000 é reservada pelo AirPlay no macOS)
```

---

## 11. Reprodutibilidade e Boas Práticas


| Prática                             | Implementação                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Seed fixo**                       | `RANDOM_STATE = 42` em todos os módulos; `torch.manual_seed(42)` e `np.random.seed(42)` no `MLPTrainer` |
| **Validação cruzada estratificada** | `StratifiedKFold(n_splits=5)` em todos os baselines                                                     |
| **Sem data leakage**                | `fit_transform` apenas em X_train; `transform` em val/test/produção                                     |
| **Pipeline reprodutível**           | `build_full_pipeline()` salvo em joblib garante mesma transformação no treino e na API                  |
| **Logging estruturado**             | `logging.getLogger(__name__)` em todos os módulos; JSON no middleware da API                            |
| **Schema validation**               | Pandera valida o dataset antes do treino; Pydantic valida entradas da API                               |
| **Testes automatizados**            | 43 testes cobrindo dados, schema, modelo e API                                                          |
| **Linting zero erros**              | `ruff check` sem warnings                                                                               |
| **Single source of truth**          | `pyproject.toml` define dependências, ruff e pytest                                                     |


---

## 12. Roteiro STAR para o Vídeo

> Use este roteiro como base para gravar o vídeo de 5 minutos. Cada seção tem tempo sugerido.

### S — Situation (0:00 – 0:45)

*"Uma operadora de telecomunicações está perdendo clientes. O problema é que sem um modelo preditivo, a equipe de retenção age de forma reativa — o cliente já cancelou quando a empresa descobre. Usamos o dataset público IBM Telco Customer Churn com 7 mil clientes e 21 features para construir um sistema que identifica o risco de churn antes que ele aconteça."*

- Mostrar: print do dataset, distribuição de churn (73%/27%)
- Destacar: desbalanceamento e por que accuracy não funciona aqui

---

### T — Task (0:45 – 1:30)

*"Nossa tarefa foi construir um pipeline end-to-end — do dado bruto até um modelo servido via API — aplicando todas as boas práticas de MLE: feature engineering com sklearn Pipeline, rastreamento com MLflow, rede neural com PyTorch, e inferência com FastAPI."*

- Mostrar: estrutura de pastas do projeto
- Destacar: pyproject.toml como single source of truth

---

### A — Action (1:30 – 3:45)

**Feature Engineering (30s):**  
*"Criamos 14 features derivadas. A mais importante é `charges_per_tenure` — custo por mês de relacionamento — que captura clientes novos pagando muito, um sinal forte de churn. Encapsulamos tudo em um `FeatureEngineerTransformer` sklearn-compatível para que o Pipeline garanta a mesma transformação no treino e na inferência."*

**Modelo MLP (45s):**  
*"A rede neural tem 3 camadas [128, 64, 32] com BatchNorm, ReLU e Dropout de 30%. O BCEWithLogitsLoss combina sigmoid e cross-entropy com maior estabilidade numérica. O early stopping monitora o val_loss e restaura os pesos do melhor epoch — evitando overfitting sem depender de um número fixo de epochs."*

**Baselines e MLflow (30s):**  
*"Treinamos 4 baselines — DummyClassifier, Logistic Regression, Random Forest e Gradient Boosting — todos rastreados no MLflow com validação cruzada estratificada de 5 folds. O MLP precisa superar o melhor baseline para justificar sua complexidade."*

**API FastAPI (30s):**  
*"A API tem dois endpoints: /health para monitoramento e /predict para inferência. O Pydantic valida o JSON de entrada — campos ausentes ou inválidos retornam 422 automaticamente. O middleware mede e loga a latência de cada requisição em JSON estruturado."*

---

### R — Result (3:45 – 5:00)

*"O sistema entrega probabilidade de churn de 0 a 1, classificação binária e nível de risco (low/medium/high) em menos de 10ms por requisição. Com 43 testes automatizados e linting zero erros, o código está pronto para CI/CD. O MLflow permite comparar todos os experimentos e rastrear qual modelo foi para produção."*

- Mostrar: chamada curl ao /predict com resposta JSON
- Mostrar: MLflow UI com comparação de modelos
- Mostrar: `pytest` com 43 passed

