# Model Card — Telco Customer Churn Prediction

## Model Details


| Campo                | Valor                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nome**             | `churn-classifier@champion` (serving padrão) + ChurnMLP (comparativo da Fase 1)                                                                |
| **Versão**           | Controlada pelo MLflow Registry e registrada em `models/registry.json`                                                                          |
| **Tipo**             | Pipeline sklearn (Logistic Regression campeã) e Multi-Layer Perceptron (PyTorch)                                                               |
| **Arquitetura**      | Linear(59→128) → BN → ReLU → Dropout(0.3) → Linear(128→64) → BN → ReLU → Dropout(0.3) → Linear(64→32) → BN → ReLU → Dropout(0.3) → Linear(32→1) |
| **Loss**             | BCEWithLogitsLoss com pos_weight (balanceamento de classe)                                                                                      |
| **Otimizador**       | Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau                                                                                           |
| **Treinamento**      | Early stopping (patience=10, baseado em val_loss)                                                                                               |
| **Seed**             | 42 (reprodutibilidade total)                                                                                                                    |
| **Desenvolvido por** | FIAP 10MLET — Tech Challenge Fase 1                                                                                                             |


## Uso Pretendido

Sistema para prever a probabilidade de churn de clientes de uma operadora de telecomunicacoes. A API serve por padrão o pipeline sklearn promovido como `churn-classifier@champion`; a MLP da Fase 1 foi preservada e pode ser selecionada com `MODEL_SOURCE=mlp`.

**Caso de uso primario:** ranquear clientes por risco de cancelamento para priorizacao de acoes preventivas de retencao pela equipe de Customer Success.

---

**Usuários pretendidos:** Equipe de CRM/retenção, cientistas de dados da operadora.


| Atributo           | Valor                                                                |
| ------------------ | -------------------------------------------------------------------- |
| Fonte              | IBM Telco Customer Churn Dataset                                     |
| Arquivo            | `data/raw/Telco_customer_churn.csv`                                  |
| Colunas no CSV     | 33 (demograficas, geograficas, contratuais, de uso, cobranca, churn) |
| Colunas usadas     | 21 (mapeadas via `COLUMN_MAP` em `src/data/preprocessing.py`)        |
| Registros totais   | 7.043 clientes                                                       |
| Split              | 70% treino / 10% validacao / 20% teste (estratificado por churn)     |
| Taxa de churn      | 26,5% (1.869 churners / 7.043 clientes)                              |
| Features raw       | 19 variaveis (3 numericas + 1 binaria + 15 categoricas)              |
| Features apos eng. | 33 variaveis (19 raw + 14 derivadas)                                 |
| Input dim (MLP)    | 59 (apos one-hot encoding das 15 categoricas)                        |
| Target             | `churn` (binario: 0 = nao cancela, 1 = cancela)                      |


> **Nota:** Colunas `Churn Reason`, `Churn Score`, `CLTV`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Country`, `State` e `Count` sao descartadas antes do treinamento para evitar data leakage e reducao de dimensionalidade geografica.

### Features Raw Utilizadas


| Grupo       | Colunas                                                                                                                                                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Numericas   | `tenure`, `monthly_charges`, `total_charges`                                                                                                                                                                                                             |
| Binaria     | `senior_citizen`                                                                                                                                                                                                                                         |
| Categoricas | `gender`, `partner`, `dependents`, `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `contract`, `paperless_billing`, `payment_method` |


### Features Derivadas (Engenharia de Features)


| Feature                 | Descricao                                             |
| ----------------------- | ----------------------------------------------------- |
| `charges_per_tenure`    | Custo medio por mes: `monthly_charges / (tenure + ε)` |
| `num_services`          | Contagem de servicos ativos (soma dos `has_`*)        |
| `is_new_customer`       | Flag: tenure <= 3 meses                               |
| `is_long_term`          | Flag: tenure > 24 meses                               |
| `is_monthly_contract`   | Flag: contrato "Month-to-month"                       |
| `is_electronic_check`   | Flag: pagamento via cheque eletronico                 |
| `has_phone_service`     | Flag: servico de telefonia ativo                      |
| `has_multiple_lines`    | Flag: multiplas linhas ativas                         |
| `has_online_security`   | Flag: seguranca online ativa                          |
| `has_online_backup`     | Flag: backup online ativo                             |
| `has_device_protection` | Flag: protecao de dispositivo ativa                   |
| `has_tech_support`      | Flag: suporte tecnico ativo                           |
| `has_streaming_tv`      | Flag: streaming de TV ativo                           |
| `has_streaming_movies`  | Flag: streaming de filmes ativo                       |


---

## Arquitetura do Modelo

```text
Input (59)
   │
   ├─► Linear(59→128) → BatchNorm(128) → ReLU → Dropout(0.3)
   │
   ├─► Linear(128→64) → BatchNorm(64)  → ReLU → Dropout(0.3)
   │
   ├─► Linear(64→32)  → BatchNorm(32)  → ReLU → Dropout(0.3)
   │
   └─► Linear(32→1)   [logit]
           │
        Sigmoid → churn_probability ∈ [0, 1]
```


| Componente      | Escolha            | Justificativa                                          |
| --------------- | ------------------ | ------------------------------------------------------ |
| Camadas ocultas | [128, 64, 32]      | Compressao gradual; evita bottleneck abrupto           |
| Ativacao        | ReLU               | Evita vanishing gradient em redes moderadas            |
| Regularizacao   | Dropout(0.3)       | Reduz overfitting em dataset pequeno (~5k treino)      |
| Normalizacao    | BatchNorm          | Estabiliza gradientes, acelera convergencia            |
| Loss            | BCEWithLogitsLoss  | Numericamente estavel para classificacao binaria       |
| Otimizador      | Adam (lr=1e-3)     | Adaptativo; padrao para MLPs em dados tabulares        |
| Early stopping  | patience=10 epochs | Interrompe ao detectar overfitting; restaura best ckpt |


---

## Metricas (conjunto de teste — 1.409 registros, 374 churners)

### Modelo servido: Logistic Regression campeã (threshold = 0.50)


| Metrica  | Valor      |
| -------- | ---------- |
| F1-Score | **0.6176** |
| AUC-ROC  | **0.8531** |
| Precisao | 0.5087     |
| Recall   | 0.7861     |
| Acuracia | 0.7417     |


## Selecao e comparacao

Os baselines são selecionados por validação cruzada estratificada apenas no conjunto
de desenvolvimento. O teste permanece intocado até o reajuste final do campeão.


| Baseline              | F1 médio CV | AUC-ROC médio CV |
| --------------------- | ----------- | ---------------- |
| DummyClassifier       | 0.2715      | 0.5068           |
| Random Forest         | 0.5300      | 0.8322           |
| Gradient Boosting     | 0.6023      | 0.8578           |
| Logistic Regression 🏆 | **0.6424**  | **0.8586**       |

| Modelo final          | F1 teste | AUC-ROC | Precisao | Recall | Acuracia |
| --------------------- | -------- | ------- | -------- | ------ | -------- |
| Logistic Regression 🏆 | 0.6176   | 0.8531  | 0.5087   | 0.7861 | 0.7417   |
| **MLP PyTorch**       | **0.6308** | 0.8521 | 0.5082   | **0.8316** | 0.7417 |


**Conclusao:** A Logistic Regression foi a melhor baseline por F1 médio de CV e foi
promovida como campeã. A MLP obteve F1 e recall de teste ligeiramente maiores, mas
o Registry mantém o pipeline sklearn como entrega da Fase 2 e a API torna essa
origem verificável no endpoint `/health`.

---

## Analise de Threshold e Custo de Negocio

### Premissas de custo


| Erro                | Custo estimado | Descricao                                        |
| ------------------- | -------------- | ------------------------------------------------ |
| Falso Positivo (FP) | $10            | Acionar retencao para cliente que nao cancelaria |
| Falso Negativo (FN) | $500           | Perder cliente que cancela sem intervencao       |


### Impacto (threshold = 0.50)


| Threshold | TP  | FP  | FN  | TN  | Custo Total |
| --------- | --- | --- | --- | --- | ----------- |
| 0.50      | 222 | 115 | 152 | 920 | **$77.150** |


> **Nota:** Analise completa de multiplos thresholds (0.10, 0.30, 0.40) disponivel no notebook `notebooks/modeling.ipynb`. O threshold otimo varia conforme a capacidade operacional de atendimento da equipe de retencao.

**Recomendacao operacional:** Reduzir o threshold para 0.30–0.40 quando o objetivo e maximizar o Recall (capturar mais churners), aceitando maior volume de contatos desnecessarios. Para capacidade operacional limitada, manter 0.50.

---

**Trade-off de custo:**

- Falso Negativo (churn não detectado): alto custo — cliente perdido sem intervenção.
- Falso Positivo (churn previsto erroneamente): baixo custo — ação de retenção desnecessária.
- Recomenda-se threshold abaixo de 0.5 para maximizar recall em contextos onde o custo de FN é dominante.

## Limitações

- **Aplicacao:** Ranquear clientes por risco de cancelamento para acoes proativas de retencao.
- **Usuarios:** Equipe de Customer Success; analistas de CRM.
- **Integracao:** API REST (`POST /predict`) que retorna `churn_probability` e `predicted_churn`.
- **Threshold padrao na API:** 0.5 (ajustavel via parametro `threshold` na requisicao).
- **Saida:** Probabilidade continua em [0,1] + classificacao binaria com threshold configuravel.

---

## Considerações de Viés e Fairness

- Treinado com dados de uma unica operadora norte-americana — pode nao generalizar para outros contextos, regioes ou perfis de clientes.
- Nao inclui variaveis de satisfacao do cliente (NPS, reclamacoes, interacoes com suporte).
- `Churn Reason` e `Churn Score` foram excluidos do pipeline para evitar leakage — sao validos apenas para diagnostico no EDA.
- Assume estacionariedade dos padroes de churn; performance degrada com mudancas no perfil de clientes (concept drift ou data drift).
- Dataset com ~7k registros e de tamanho moderado; modelos de maior complexidade podem sofrer overfitting em producao com distribuicao diferente.
- Features de cobranca (`monthly_charges`, `total_charges`) sao sensiveis a mudancas de precificacao ou politicas comerciais.
- Nao recomendado para decisoes automaticas sem revisao humana em casos com probabilidade entre 0.35 e 0.65 (zona de incerteza).

---

## Cenarios de Falha


| Cenario                                   | Impacto Esperado                           | Mitigacao                                      |
| ----------------------------------------- | ------------------------------------------ | ---------------------------------------------- |
| Novos tipos de contrato                   | Features de contrato invalidas ou nulas    | Validar schema de entrada na API               |
| Mudanca no plano de cobranca              | Distribuicao de `monthly_charges` desviada | Monitorar PSI mensal; alertar se > 0.2         |
| Segmento de clientes novo (ex.: empresas) | Features demograficas fora do range        | Rejeitar ou sinalizar como out-of-distribution |
| Dados faltantes em features criticas      | Imputacao incorreta pelo pipeline          | Validar missingness < 5% por coluna            |
| API fora do ar durante pico               | Perda de scoring em batch                  | Implementar retry + fallback rule-based        |


---

## Implantação

- **Atributos sensiveis presentes nos dados:** `gender` (Male/Female), `senior_citizen` (Yes/No).
- O modelo foi treinado sem exclusao desses atributos; e necessario avaliar disparidade de performance por grupo antes de uso em producao.
- **Avaliacao recomendada:** Medir `false_negative_rate` e `false_positive_rate` separadamente para cada grupo usando Fairlearn (`MetricFrame`).
- Particular atencao ao `senior_citizen`: este grupo tem padroes de uso e churn distintos e pode ser desproporcionalmente afetado por limiares subotimos.
- Monitorar se a taxa de intervencao (acoes de retencao disparadas) e homogenea entre grupos demograficos.

---

## Monitoramento em Producao


| Metrica                             | Frequencia | Limiar de Alerta                            | Acao                           |
| ----------------------------------- | ---------- | ------------------------------------------- | ------------------------------ |
| Distribuicao de `churn_probability` | Semanal    | PSI > 0.2 vs. baseline                      | Investigar data drift          |
| F1-Score (ground truth delay)       | Mensal     | Queda > 5 p.p. vs. campeão (0.6176)         | Acionar retreinamento          |
| Taxa de missings por feature        | Diaria     | > 5% em qualquer feature critica            | Alertar engenharia de dados    |
| Volume de predicoes (OOD)           | Diaria     | > 10% requests com prob. extrema inesperada | Revisar distribuicao           |
| Latencia da API (p95)               | Continuo   | > 500ms                                     | Escalar ou otimizar inferencia |


**Politica de retreinamento:** Retreinar mensalmente com dados dos ultimos 6 meses, ou imediatamente quando PSI > 0.2 ou queda de F1 for detectada.

---

## Artefatos do Modelo


| Artefato        | Caminho                       | Descricao                                         |
| --------------- | ----------------------------- | ------------------------------------------------- |
| Pesos do MLP    | `models/mlp_model.pt`         | Checkpoint PyTorch com best val loss              |
| Preprocessador  | `models/preprocessor.joblib`  | Pipeline sklearn (FeatureEngineer + Scaler + OHE) |
| Melhor baseline | `models/best_baseline.joblib` | Logistic Regression (F1=0.6176 no teste intocado) |
| Evidência Registry | `models/registry.json`     | nome, versão, alias, run e URI do campeão          |
| Configuracao    | `models/model_config.json`    | `{"input_dim": 59}`                               |
| Metricas        | `models/results.json`         | CV dos baselines e teste apenas dos modelos finais |


---

## Manutenção


| Atributo            | Valor                          |
| ------------------- | ------------------------------ |
| Equipe              | FIAP Tech Challenge — Fase 1   |
| Autor               | Bruno Brito                    |
| Data de treinamento | 2026-05                        |
| Versao              | 1.1.0                          |
| Framework           | PyTorch 2.x / scikit-learn 1.x |
| Experimento MLflow  | `churn-prediction`             |
