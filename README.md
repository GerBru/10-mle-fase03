# Telco Customer Churn Prediction - Fase 2
## Machine Learning Engineering com MLOps, DVC e Scikit-Learn

**Status:** Sprint 1 ✅ COMPLETO  
**Branch:** `feature/sprint-1-setup` → Pronto para merge em `develop`  
**Repositório:** [10-machine-learning-engineering-02](https://github.com/bbryttos/10-machine-learning-engineering-02)

---

## 🎯 Sprint 1 Summary: O que Mudou da Fase 1 para Fase 2

### Fase 1 vs Fase 2: Decisões Estratégicas

| Aspecto | **Fase 1** | **Fase 2** | **Por quê?** |
|---------|----------|-----------|-------------|
| **Modelo Principal** | Rede Neural (PyTorch MLP) | Scikit-Learn Clássico | Foco em MLOps, não complexidade de modelo |
| **API REST** | ✅ FastAPI com endpoints | ❌ Removida | Fase 2 é pipeline reprodutível, não serve |
| **Observabilidade** | ✅ Prometheus + Grafana | ❌ Removida | Será adicionada em Deploy (opcional) |
| **Versionamento de Dados** | ❌ Manual/Git | ✅ **DVC** | Essencial para reprodutibilidade |
| **Tracking de Experimentos** | ✅ MLflow Tracking | ✅ MLflow Tracking + **Registry** | Gerenciamento de modelos em produção |
| **Foco** | Modelos avançados + Deploy | **Engenharia de ML + Reprodutibilidade** | Professional MLOps practices |

---

## 📊 Sprint 1: O que foi Entregue

### ✅ Estrutura Base

```
src/
├── data/ # Carregamento e preprocessamento
├── features/ # Feature engineering
├── models/ # Scikit-Learn clássico + MLP (legacy)
├── training/ # Loop de treinamento com MLflow
└── utils/ # Logging e helpers

tests/
├── test_model.py # 11 testes: MLP, baselines, evaluation
├── test_preprocessing.py # 7 testes: limpeza e transformação
├── test_schema.py # 6 testes: validação de dados (Pandera)
└── test_smoke.py # 6 testes: integração end-to-end
```

data/raw/ # Dataset Telco (será versionado com DVC em Sprint 2)
notebooks/ # EDA e exploração


---

## 🧪 Testes: 30/30 Passando ✅

### Cobertura por Módulo

- src/models/baseline.py 100% ✅ (tudo testado)
- src/models/evaluation.py 100% ✅ (tudo testado)
- src/models/mlp.py 99% ✅ (quase perfeito)
- src/data/preprocessing.py 96% ✅
- src/features/engineering.py 100% ✅ (tudo testado)
- src/utils/logger.py 100% ✅ (tudo testado)

TOTAL: 71% Coverage


### Breakdown dos 30 Testes

**Test Model (11):** MLP behavior, baselines, evaluation metrics  
**Test Preprocessing (7):** Data cleaning, transformation, imputation  
**Test Schema (6):** Data validation with Pandera  
**Test Smoke (6):** End-to-end integration tests  

---

## 🚀 Como Rodar Testes

```bash
# Instalar (primeira vez)
uv sync

# Todos os testes (30 passed)
uv run pytest -v

# Com cobertura
uv run pytest --cov=src --cov-report=term-missing

# Linting
uv run ruff check src/ tests/

# Ou via Makefile
make test
make lint
```

---

## 📋 O Que NÃO Faz Parte de Sprint 1

❌ **DVC** — Sprint 2  
❌ **config.yaml** — Sprint 2  
❌ **dvc.yaml** — Sprint 2  
❌ **MLflow experiments** — Sprint 3  
❌ **Notebook execution** — Sprint 2 (corrigir imports)  

---

## ✅ Checklist Sprint 1

- [x] Estrutura da Fase 1 integrada
- [x] Módulos desnecessários removidos
- [x] pyproject.toml atualizado
- [x] uv.lock gerado
- [x] 30/30 testes passando
- [x] 71% coverage
- [x] Linting clean
- [x] README atualizado com realidade Fase 2

---

## 🎯 Próximos Passos (Sprint 2)

**Sprint 2: DVC + Reprodutibilidade**

[ ] Inicializar DVC
[ ] Versionar dataset Telco
[ ] Criar config.yaml
[ ] Criar dvc.yaml pipeline
[ ] Testar dvc repro


---

## 📞 FAQ

**P: Por que removeram FastAPI?**  
R: Fase 2 foca em pipeline reprodutível e MLOps. API é adicionada em deploy opcional.

**P: E PyTorch?**  
R: Fase 2 usa Scikit-Learn. PyTorch fica como legacy.

**P: Como rodar testes?**  
R: `uv run pytest -v` (30 passed em ~12s)

**P: Qual cobertura?**  
R: 71% total. Baseline 100%, Evaluation 100%, MLP 99%. Bom para Fase 2.

---

**Última atualização:** 08/08/2026  
**Status:** Sprint 1 ✅ COMPLETO  
**Próximo:** Sprint 2 — DVC Setup
