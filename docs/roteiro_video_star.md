# Roteiro do vídeo STAR — Tech Challenge Fase 2

Duração alvo: 5 minutos. O roteiro prioriza evidência executável: DVC, MLflow
Registry, API e resultado. Não acelere a fala; grave previamente os comandos mais
demorados e deixe o Docker Compose em execução antes de iniciar.

## Preparação antes da gravação

1. Preencha `.env` com segredos locais e execute `docker compose up --build -d`.
2. Confirme `docker compose ps`: MLflow saudável, pipeline concluído e API saudável.
3. Abra quatro abas: terminal na raiz, MLflow em `localhost:5001`, Swagger em
   `localhost:8000/docs` e `models/results.json` no editor.
4. Gere um token pela rota `/token` ou deixe preparada uma chamada com API key.
5. Aumente a fonte do terminal e o zoom do navegador; não mostre `.env`,
   `.dvc/config.local`, tokens ou credenciais.

## S — Situação (0:00–0:40)

### Fala

“Uma operadora de telecomunicações precisa antecipar quais clientes têm maior
risco de cancelamento. Na Fase 1 construímos o modelo MLP e uma API de inferência.
Nesta Fase 2, o desafio passou a ser operacional: reproduzir o treinamento,
versionar dados e artefatos, comparar modelos sem vazamento e promover de forma
rastreável o modelo que será consumido pela aplicação.”

### Mostrar

- Título do README e a visão geral do repositório.
- `data/raw/Telco_customer_churn.csv.dvc`, sem abrir dados sensíveis.

## T — Tarefa (0:40–1:10)

### Fala

“A tarefa foi transformar a solução em um pipeline completo de Machine Learning
Engineering. Precisávamos reaproveitar o dataset e a API da Fase 1, controlar o
fluxo com DVC, rastrear experimentos no MLflow, registrar um campeão e garantir
que a mesma aplicação servisse esse modelo, tudo executável localmente com Docker
e protegido por testes e CI.”

### Mostrar

- Diagrama de arquitetura no README.
- Os estágios de `dvc.yaml` e os parâmetros centralizados em `params.yaml`.

## A — Ação (1:10–3:45)

### 1:10–1:50 — Reprodutibilidade com DVC

### Fala

“O DVC controla o dataset, as dependências, os parâmetros e os artefatos. O estágio
de preprocessamento valida o schema e cria splits estratificados. O treino consome
esses splits, salva modelos e métricas e também produz uma evidência do Registry.
Uma alteração de parâmetro invalida somente os estágios necessários.”

### Mostrar e executar

```bash
uv run --extra train dvc dag
uv run --extra train dvc status
uv run --extra train dvc metrics show
```

### 1:50–2:35 — Seleção sem vazamento e rastreabilidade

### Fala

“Dummy, Logistic Regression, Random Forest e Gradient Boosting são comparados com
validação cruzada estratificada apenas no conjunto de desenvolvimento. A escolha
usa F1 médio de CV; o teste fica intocado. Depois da seleção, a Logistic Regression
é reajustada em todo o desenvolvimento e avaliada uma única vez no teste. O MLflow
registra parâmetros, métricas e artefatos.”

### Mostrar

- MLflow UI com as runs e métricas.
- `models/results.json`, destacando os blocos `cv` e `test`.

### 2:35–3:05 — Model Registry

### Fala

“O campeão é registrado como `churn-classifier`, recebe uma versão imutável e o
alias `champion`. A promoção é obrigatória no fluxo normal: se o Registry falhar,
o pipeline falha, em vez de aparentar uma entrega completa. O arquivo
`models/registry.json` registra nome, versão, alias, run e URI como evidência.”

### Mostrar

- MLflow: Models → `churn-classifier` → alias `champion`.
- `models/registry.json` no editor.

### 3:05–3:45 — API da Fase 1 integrada ao campeão

### Fala

“A API da Fase 1 foi preservada com JWT, API key, rate limiting, batch e métricas
Prometheus. O repositório de modelos agora carrega por padrão o campeão sklearn;
a MLP continua disponível por configuração. O Docker Compose espera o MLflow,
executa o pipeline DVC e só libera a API depois que os artefatos existem.”

### Mostrar e executar

```bash
curl -s http://localhost:8000/health
```

Destaque na resposta: `model_source`, `model_name`, `model_version` e
`model_alias`. Em seguida, faça uma predição pelo Swagger ou `/predict-apikey` e
mostre probabilidade, classe e nível de risco.

## R — Resultado (3:45–4:40)

### Fala

“A Logistic Regression venceu os baselines com F1 médio de validação cruzada de
0,6424 e AUC de 0,8586. No teste intocado, obteve F1 de 0,6176, recall de 0,7861 e
AUC de 0,8531. A MLP preservada obteve F1 de 0,6308. A diferença está documentada:
o campeão sklearn atende ao escopo de Registry da Fase 2, enquanto a MLP continua
disponível como alternativa. A suíte final possui 82 testes, 81,86% de cobertura
e lint sem erros. Também validamos a stack completa: pipeline concluído, modelo
registrado e API respondendo com o alias champion.”

### Mostrar

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run ruff check .
docker compose ps
```

Use uma captura prévia do resultado para não gastar o vídeo esperando os testes.

## Encerramento (4:40–5:00)

### Fala

“Assim, a entrega não é apenas um notebook ou um arquivo de modelo. É um fluxo
reprodutível do dado à inferência, com separação correta do teste, rastreabilidade,
versionamento, promoção explícita e operação em containers. Como evolução,
podemos automatizar retreinamento por drift e publicar a mesma imagem em cloud.”

## Checklist final da evidência

- DVC DAG e status em dia.
- MLflow com runs e `churn-classifier@champion`.
- `registry.json` coerente com a UI.
- `/health` identifica o campeão e uma predição retorna HTTP 200.
- Terminal mostra 82 testes, cobertura acima de 80% e Ruff sem erros.
- Nenhuma credencial aparece na gravação.
- Link do vídeo adicionado ao README antes da entrega.
