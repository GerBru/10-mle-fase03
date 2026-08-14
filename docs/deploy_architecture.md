# Arquitetura de Deploy — Churn Prediction

> Justifica a escolha entre **batch** e **real-time**, descreve a infraestrutura
> (AWS ECS Fargate) e a estratégia de release (canary). Complementa o  `[model_card.md](model_card.md)` e o `[ml_canvas.md](ml_canvas.md)`.   

---

## 1. Batch vs. Real-time — qual usar?

### Análise do caso de uso

   A decisão de negócio (ver ML Canvas) é **acionar a equipe de Customer Success para reter clientes em risco**. Essa ação é **periódica e não interativa**: a fila de retenção é trabalhada ao longo da semana, não no milissegundo em que o cliente aparece.


| Critério                 | Batch                  | Real-time                 | Vencedor p/ churn                    |
| ------------------------ | ---------------------- | ------------------------- | ------------------------------------ |
| Latência exigida         | minutos/horas          | < 300 ms                  | **Batch** (não há usuário esperando) |
| Frequência de decisão    | diária/semanal         | por evento                | **Batch**                            |
| Volume                   | toda a base de uma vez | 1 cliente por chamada     | **Batch** (eficiência)               |
| Custo de infra           | baixo (job agendado)   | maior (serviço sempre on) | **Batch**                            |
| Complexidade operacional | menor                  | maior                     | **Batch**                            |


### ✅ Decisão: **Batch como modo primário**, com **API real-time disponível**

- **Primário — Batch scoring (recomendado):** job agendado (ex.:
 diário/semanal) que pontua **toda a base** de clientes e grava o `churn_probability` em  
 uma tabela,consumida pelo CRM/CS. É o que melhor atende ao caso de uso e minimiza custo.
- **Secundário — Real-time (API FastAPI):** mantido para **consultas**  
 **pontuais e integração** (ex.: agente de CS abre a ficha de um cliente e quer o  
 score na hora, simulações "what-if", testes). Já implementado em `src/api/`.

> Resumindo: **batch para operar a campanha de retenção em escala**;  
>  **real-time para > consulta individual e integrações sob demanda**.

---

## 2. Arquitetura de Referência (AWS)

### 2.1 Modo Batch (primário)

- **EventBridge** dispara a task no cron definido (ex.: `0 6 * * `*).    
- **ECS Fargate Task** (efêmera) roda o container, pontua e encerra — paga-se só pela execução.
- Artefato do modelo versionado via **MLflow Model Registry** (ou S3).

### 2.2 Modo Real-time (secundário)

- **ALB** (Application Load Balancer) → **ECS Fargate Service** com a API.
- Imagem buildada pelo `Dockerfile` e publicada no **ECR**.
- Escala horizontal por *target tracking* (CPU/requisições).

### 2.3 CI/CD

---

## 3. Estratégia de Release — Canary

 Para o **serviço real-time**, novos modelos/versões são promovidos com **canary deployment** (alinhado ao `monitoring_plan`):

1. Nova versão recebe **uma fração do tráfego** (ex.: 10%).
2. Compara métricas online (latência, erro, distribuição de score) vs. versão atual.
3. Se dentro dos SLOs por janela de observação → **rollout gradual** até 100%.
4. Qualquer violação de SLO → **rollback automático**.

Para o **batch**, a promoção usa **shadow scoring**: a nova versão pontua em paralelo sem efeito operacional; só assume após validação offline contra a versão vigente.

---

## 4. SLOs de Produção


| SLO                       | Alvo                              | Aplica-se a |
| ------------------------- | --------------------------------- | ----------- |
| Disponibilidade da API    | ≥ 99,5%                           | Real-time   |
| Latência `/predict` (p95) | < 300 ms                          | Real-time   |
| Janela do job de scoring  | concluir em < 30 min              | Batch       |
| Frescor do modelo         | retreino ≤ 30 dias                | Ambos       |
| Qualidade (F1)            | não cair > 5 p.p. vs. baseline    | Ambos       |
| Drift (PSI)               | < 0.2 (alerta), retreino se ≥ 0.2 | Ambos       |


---

## 5. Resumo da Decisão

> **Batch é o modo primário** porque a decisão de retenção é periódica, de alto volume e sem necessidade de latência interativa — o que reduz custo e complexidade. A **API real-time** é mantida para consultas individuais e integrações.  
> Releases usam **canary** (real-time) e **shadow scoring** (batch), com **rollback** ligado aos SLOs.

