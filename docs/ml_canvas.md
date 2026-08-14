# 🧩 ML Canvas — Churn Prediction

> Artefato da **Etapa 1**. Estrutura o problema de negócio antes da modelagem:
> stakeholders, decisões, dados, métricas de negócio e SLOs.

---

## 1. Proposta de Valor (Value Proposition)

**Problema de negócio:** uma operadora de telecomunicações sofre perda acelerada de clientes (churn ~26% na base Telco). Cada cliente perdido representa receita recorrente (MRR) e LTV perdidos, além do alto custo de aquisição de um substituto (CAC).

**Solução de ML:** prever a **probabilidade de churn** por cliente para que a equipe de Customer Success atue **preventivamente** (ofertas de retenção, contato proativo) nos clientes de maior risco — otimizando o orçamento de retenção.

**Por que ML (e não regra fixa):** o churn depende de interações não lineares entre contrato, tempo de casa (tenure), tipo de serviço e cobrança. Um modelo aprende esses padrões melhor que regras estáticas e fornece um *score* contínuo priorizável.

---

## 2. Stakeholders


| Stakeholder                | Interesse                             | Como usa o modelo                            |
| -------------------------- | ------------------------------------- | -------------------------------------------- |
| **Customer Success (CS)**  | Reduzir churn, bater meta de retenção | Recebe lista priorizada de clientes em risco |
| **Marketing / Retenção**   | Alocar orçamento de campanhas         | Define ofertas por faixa de risco/LTV        |
| **Diretoria / Financeiro** | Proteger receita recorrente (MRR/LTV) | Acompanha churn evitado em R$                |
| **Equipe de ML / Eng.**    | Manter o modelo saudável em produção  | Monitora drift, retreina, versiona           |
| **Cliente final**          | Não receber abordagem indevida        | Beneficiado por ofertas relevantes           |


---

## 3. Decisões (Decisions)

- **Decisão operacional:** abordar ou não um cliente com ação de retenção nesta semana.
- **Como a predição vira decisão:** clientes acima do *threshold* operacional entram na
fila de retenção, ordenados por `churn_probability × LTV` (risco ponderado por valor).
- **Threshold:** ajustável por custo (ver Seção 6 e Model Card). Default técnico = 0.5;
threshold de negócio escolhido pela curva custo-benefício.

---

## 4. Fonte de Dados (Data Sources)

- **Telco Customer Churn (IBM)** — 7.043 registros, 21 features, target `Churn` (Yes/No).
- Categorias: demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`),
contratuais (`Contract`, `PaymentMethod`, `PaperlessBilling`), serviço
(`InternetService`, `OnlineSecurity`, ...), uso/cobrança (`tenure`, `MonthlyCharges`,
`TotalCharges`).
- **Desbalanceamento:** ~26% churn → tratado com SMOTE (somente no conjunto de treino).

---

## 5. Coleta & Preparação (Data Collection / Features)

- **Split:** 70% treino / 10% validação / 20% teste — estratificado por `Churn`.
- **Pré-processamento:** imputação de `TotalCharges`, encoding categórico, normalização
numérica, pipeline `scikit-learn` versionado junto ao modelo.
- **Janela de retreino:** mensal ou sob *trigger* de drift (PSI > 0.2).

---

## 6. Métricas de Negócio & SLOs (Metrics)

### 6.1 Métrica técnica (modelo)

- **Primária:** F1-Score e PR-AUC (apropriadas para classe minoritária).
- **Secundárias:** AUC-ROC, Precision, Recall.

### 6.2 Métrica de negócio — Churn evitado (R$)

A decisão é guiada pelo **valor financeiro**, não só por F1. Ver detalhamento e cálculo
no [Model Card](model_card.md#-métrica-de-negócio--custo--churn-evitado-r). Resumo:

- **Custo de FN** (não detectar quem vai sair) ≈ **LTV perdido** do cliente.
- **Custo de FP** (oferecer retenção a quem ficaria) ≈ **custo da oferta/contato**.
- Como `LTV >> custo da oferta`, o threshold ótimo de negócio costuma ser **< 0.5**
(privilegia Recall), capturando mais churners reais.

### 6.3 SLOs (produção)


| SLO                       | Alvo                                      |
| ------------------------- | ----------------------------------------- |
| Disponibilidade da API    | ≥ 99,5%                                   |
| Latência `/predict` (p95) | < 300 ms                                  |
| Frescor do modelo         | retreino ≤ 30 dias                        |
| Qualidade (F1)            | não cair > 5 p.p. vs. baseline de release |
| Drift de features         | PSI < 0.2 (alerta), retreino se ≥ 0.2     |


---

## 7. Avaliação & Live (Evaluation / Monitoring)

- **Offline:** comparação contra baselines (Dummy, Logistic Regression, RF, GB) — ver
Model Card. MLP só é promovido se superar baselines em F1/PR-AUC.
- **Online:** monitorar distribuição de `churn_probability` (semanal), taxa de churn
real vs. prevista, e PSI por feature. Alertas conforme SLOs.
- **Fairness:** avaliar `false_negative_rate` por `gender` e `SeniorCitizen` antes de
produção (Fairlearn).

---

## 8. Riscos & Limitações

- Dados de uma única operadora → risco de baixa generalização.
- Ausência de variáveis de satisfação (NPS, reclamações) → teto de performance.
- Data drift no perfil de clientes degrada o modelo → mitigado por monitoramento + retreino.
- Decisões automáticas na zona limiar (0.4–0.6) exigem revisão humana.

