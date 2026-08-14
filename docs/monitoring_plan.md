# Plano de Monitoramento — Telco Churn Prediction API

## Objetivos

Detectar degradação de performance, data drift e problemas operacionais antes que impactem decisões de negócio.

---

## 1. Métricas Operacionais (infraestrutura)

| Métrica | Threshold de Alerta | Ação |
|---|---|---|
| Latência p95 (`/predict`) | > 200ms | Investigar gargalo; escalar instância se necessário |
| Taxa de erro HTTP 5xx | > 1% em 5 min | Verificar logs da API; rollback se > 5% |
| Taxa de erro HTTP 4xx | > 10% em 5 min | Revisar schema da request; possível mudança upstream |
| Throughput (req/min) | < 10% do baseline | Verificar se client está enviando dados |

**Coleta:** Middleware de latência integrado na API — cada request gera `X-Latency-Ms` header + log estruturado com method, path, status e latency_ms.

---

## 2. Métricas de Modelo (performance)

Calculadas quando o ground-truth (churn real) estiver disponível (30–60 dias após predição).

| Métrica | Baseline (teste) | Alerta | Ação |
|---|---|---|---|
| AUC-ROC | ~0.857 | < 0.80 | Re-treinar com dados recentes |
| F1-Score | ~0.625 | < 0.55 | Revisar threshold ou re-treinar |
| PR-AUC | ~0.72 | < 0.62 | Verificar distribuição do target |
| Taxa de churn real vs. previsto | ~26% | Desvio > 5pp | Investigar concept drift |

---

## 3. Data Drift (features de entrada)

Monitorar distribuição das features em produção vs. distribuição de treino.

| Feature | Método | Threshold |
|---|---|---|
| `tenure` / `Tenure Months` | KS test | p-value < 0.05 |
| `monthly_charges` / `Monthly Charges` | KS test | p-value < 0.05 |
| `total_charges` / `Total Charges` | KS test | p-value < 0.05 |
| `contract` / `Contract` | Chi-quadrado | p-value < 0.05 |
| `internet_service` / `Internet Service` | Chi-quadrado | p-value < 0.05 |
| `churn_probability` (output) | KS test vs. baseline | p-value < 0.05 |

**Ferramenta recomendada:** [Evidently AI](https://www.evidentlyai.com/) para dashboards de drift.

**Frequência:** Semanal, ou ao atingir 500+ predições acumuladas desde a última verificação.

O módulo `src/monitoring/drift_detection.py` implementa KS test e Chi-quadrado para uso em batch offline.

---

## 4. Monitoramento de Fairness

Executar mensalmente ou após cada re-treino:

- Calcular **demographic parity** e **equalized odds** por:
  - `gender` (Male vs. Female)
  - `senior_citizen` (Yes vs. No)
  - `contract` (Month-to-month vs. contratos longos)

**Threshold:** Diferença entre grupos > 10pp requer revisão (parâmetro `DIFFERENCE_THRESHOLD` em `src/monitoring/fairness.py`).

**Implementação:** `src/monitoring/fairness.py` — usa [Fairlearn](https://fairlearn.org/) `MetricFrame` com `mf.difference()` para as métricas:
- `false_negative_rate` — churners não detectados por grupo (impacto direto de negócio)
- `false_positive_rate` — falsos alarmes por grupo
- `selection_rate` — taxa de predições positivas por grupo (demographic parity)

```bash
make fairness   # requer make train executado previamente
```

---

## 5. Logs Estruturados

Cada requisição ao `/predict` é registrada com (via loguru + middleware):

```json
{
  "timestamp": "2026-05-27T14:32:00Z",
  "method": "POST",
  "path": "/predict",
  "status": 200,
  "latency_ms": 12.4,
  "churn_prob": 0.73,
  "prediction": 1,
  "risk_level": "high"
}
```

Armazenar em sistema centralizado (ex.: CloudWatch, Datadog, ELK Stack).

---

## 6. Playbook de Resposta a Incidentes

### AUC-ROC abaixo do threshold

1. Verificar distribuição das features (data drift nos últimos 30 dias).
2. Coletar novos dados de treino dos últimos 60–90 dias.
3. Re-treinar via `make train` (ou `python -m src.training.train`); comparar métricas no holdout.
4. Registrar novo experimento no MLflow antes de promover.
5. Deploy gradual (canary): 10% → 50% → 100% do tráfego.

### Alta taxa de erros 5xx

1. Verificar logs da API por stack traces.
2. Confirmar se artefatos estão acessíveis (`models/mlp_model.pt`, `models/preprocessor.joblib`).
3. Rollback para versão anterior via MLflow Model Registry se necessário.

### Data drift detectado

1. Identificar qual feature driftou (gráfico de distribuição vs. referência).
2. Consultar time de dados sobre mudança upstream (novo produto, campanha, mudança de precificação).
3. Decidir entre: (a) re-treinar com dados recentes, (b) ajustar threshold, (c) aguardar normalização.

---

## 7. Frequência de Re-treino

| Gatilho | Ação |
|---|---|
| AUC-ROC real < 0.80 | Re-treino imediato |
| Data drift em ≥ 3 features (KS p < 0.05) | Re-treino em 1 semana |
| Ciclo calendário | Re-treino mensal preventivo |
| Mudança significativa no produto/preços | Re-treino imediato |

---

## 8. Artefatos Monitorados

| Artefato | Caminho | Verificação |
|---|---|---|
| MLP checkpoint | `models/mlp_model.pt` | Tamanho > 0, carrega sem erros |
| Pipeline sklearn | `models/preprocessor.joblib` | Transforma sample sem NaN |
| Configuração | `models/model_config.json` | `input_dim` presente e > 0 |
| Logs | `logs/churn_prediction.log` | Rotação automática a cada 10MB |
