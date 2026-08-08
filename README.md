# Telco Customer Churn Prediction - Fase 2
## Machine Learning Engineering com MLOps, DVC e Scikit-Learn

**Repositório:** [10-machine-learning-engineering-02](https://github.com/bbryttos/10-machine-learning-engineering-02)

---

## 📋 Contexto do Desafio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. O objetivo da **Fase 2** é construir um **pipeline reprodutível de Machine Learning** que:

- ✅ Identifique clientes com risco de churn
- ✅ Seja totalmente reprodutível (dados + código + modelos versionados)
- ✅ Siga boas práticas de engenharia de ML (Clean Code, type hints, docstrings)
- ✅ Rastreie experimentos com MLflow
- ✅ Versione dados e modelos com DVC
- ✅ Use **Scikit-Learn clássico** (não redes neurais como Fase 1)

---

## 🎯 Diferenças: Fase 1 vs Fase 2

### Fase 1 — Foco em Modelos Complexos
| Aspecto | Fase 1 |
|---------|--------|
| **Modelo** | Rede Neural (MLP) em PyTorch |
| **API** | FastAPI com endpoints REST |
| **Observabilidade** | Prometheus + Grafana |
| **Foco** | Modelos avançados + deploy em produção |

### Fase 2 — Foco em Engenharia de ML
| Aspecto | Fase 2 |
|---------|--------|
| **Modelo** | Scikit-Learn clássico (Random Forest, Regressão Logística) |
| **API** | ❌ Removida |
| **Versionamento de dados** | ✅ **DVC (obrigatório)** |
| **Reprodutibilidade** | ✅ Pipeline completo com DVC |
| **Clean Code** | ✅ Etapa 1 do desafio |
| **Foco** | Engenharia de ML + MLOps + Reprodutibilidade |

---

## 📊 Status Atual: Sprint 1 ✅ COMPLETO

### O que foi feito

#### ✅ Estrutura Copiada da Fase 1
- `src/` — Código-fonte organizado
- `tests/` — Suite de testes
- `notebooks/` — EDA e exploração
- `data/raw/` — Dataset Telco
- `Dockerfile`, `Makefile` — Infraestrutura

#### ✅ Refatoração para Fase 2
- ❌ **Removido:** `src/api/` (FastAPI não é necessário)
- ❌ **Removido:** `src/monitoring/` (Prometheus/Grafana não é necessário)
- ❌ **Removido:** Dependências de PyTorch
- ✅ **Adicionado:** DVC
- ✅ **Atualizado:** `pyproject.toml` com dependências Fase 2

#### ✅ Gerenciamento de Dependências
- `pyproject.toml` — Single source of truth
- `uv.lock` — Reprodutibilidade garantida
- Python 3.12.2 — Fixado com `pyenv`
- Stack: pandas, scikit-learn, mlflow, dvc, pytest

#### ✅ Estrutura de Diretórios Final

```
10-machine-learning-engineering-02/
├── src/
│ ├── init.py
│ ├── data/ # Carregamento e preprocessamento
│ ├── features/ # Feature engineering
│ ├── models/ # Baselines e modelos Scikit-Learn
│ ├── training/ # Loop de treinamento com MLflow
│ └── utils/ # Logging e helpers
├── tests/ # Unit, integration, e smoke tests
├── notebooks/ # EDA e exploração
├── data/
│ └── raw/ # Dataset (versionado com DVC a partir de Sprint 2)
├── models/ # Artefatos treinados
├── pyproject.toml # Dependências (prod + dev)
├── uv.lock # Lock file (reprodutibilidade)
├── Dockerfile # Containerização
├── Makefile # Automação
└── README.md # Este arquivo
```

---

## 🚀 Roadmap: 4 Sprints (8 semanas)

### **Sprint 1: Setup Limpo ✅ COMPLETO**
**Semana 1 | Status: FINALIZADO**

Objetivos:
- [x] Copiar estrutura da Fase 1
- [x] Remover FastAPI, PyTorch, Prometheus
- [x] Atualizar `pyproject.toml` para Fase 2
- [x] Gerar `uv.lock` com novas dependências
- [x] Validar instalação: `uv sync`

---

### **Sprint 2: DVC + Reprodutibilidade ⏳ PRÓXIMO**
**Semana 2 | Status: A INICIAR**

Objetivos:
- [ ] Inicializar DVC no repositório
- [ ] Versionar dataset Telco com DVC
- [ ] Criar `config.yaml` com parâmetros
- [ ] Criar `dvc.yaml` com pipeline (prepare → train)
- [ ] Validar reproducibilidade: `dvc repro`

---

### **Sprint 3: MLflow Registry + Modelo Scikit-Learn ⏳ FUTURO**
**Semana 2-3 | Status: A INICIAR**

Objetivos:
- [ ] Refatorar `src/training/train.py` para Scikit-Learn puro
- [ ] Implementar MLflow Tracking
- [ ] Adicionar MLflow Model Registry
- [ ] Treinar Random Forest e Regressão Logística
- [ ] Comparar modelos com ≥4 métricas
- [ ] Testes automatizados

---

### **Sprint 4: Model Card + Documentação Final ⏳ FUTURO**
**Semana 3-4 | Status: A INICIAR**

Objetivos:
- [ ] Expandir Model Card (`docs/model_card.md`)
- [ ] Atualizar README com instruções executáveis
- [ ] Criar plano de monitoramento
- [ ] Gravar vídeo STAR (5 minutos)
- [ ] (Opcional) Deploy em nuvem

---

## 📦 Stack Tecnológico

| Categoria | Ferramentas | Versão |
|-----------|-------------|--------|
| **Linguagem** | Python | 3.12.2 |
| **Gerenciador de deps** | uv | 0.11.14 |
| **Dados** | pandas, numpy | ^2.1.0, ^1.24.0 |
| **ML Clássico** | scikit-learn | ^1.3.0 |
| **MLOps** | MLflow | ^2.10.0 |
| **Versionamento de dados** | DVC | ^3.40.0 |
| **Validação** | Pydantic, Pandera | ^2.5.0, ^0.18.0 |
| **Logging** | loguru | ^0.7.2 |
| **Testes** | pytest, pytest-cov | ^7.4.0, ^4.1.0 |
| **Linting** | ruff | ^0.1.0 |

---

## 🚀 Quick Start

### Pré-requisitos
```bash
# Python 3.12.2
pyenv install 3.12.2
pyenv local 3.12.2

# uv
brew install uv
```

### Setup
```bash
# 1. Clonar repositório
git clone https://github.com/bbryttos/10-machine-learning-engineering-02.git
cd 10-machine-learning-engineering-02

# 2. Instalar dependências
uv sync

# 3. Validar
uv run python -c "import sklearn, mlflow, dvc, pandas; print('✅ OK')"
```

---

## 📋 Comandos Úteis

```bash
# Instalar
make install

# Testar
make test

# Lint
make lint

# DVC pipeline (Sprint 2)
make dvc-repro

# Limpar
make clean
```

---

## 🎥 Vídeo STAR (5 minutos - Sprint 4)

- **S**ituation: Telecom churn problem (contexto)
- **T**ask: Pipeline reprodutível com DVC
- **A**ction: Decisões técnicas (Scikit-Learn, MLflow, DVC)
- **R**esult: Pipeline funcional, reprodutibilidade 100%

---

## 🤝 Equipe

| Nome | RM | Papel |
|------|-----|-------|
| Anna Luiza de Angelis Souza Freitas | RM375350 | Dados / ML Engineering |
| Bruno Brito de Souza | RM374808 | Dados / ML Engineering |
| Fellipe Resende Bastos | RM373040 | Dados / ML Engineering |
| German Eduardo Brunner | RM375046 | Dados / ML Engineering |
| Marcelo da Cruz Salvador | RM375166 | Software Engineering / MLOps |

---

## ❓ FAQ

**P: Posso usar PyTorch?**  
R: Não. Fase 2 é Scikit-Learn puro.

**P: Preciso deployar em nuvem?**  
R: Não. É opcional (bonus 5%).

**P: Como reproduzir?**  
R: `uv sync` + `dvc repro`

---

**Última atualização:** 08/08/2026 (Sprint 1 Completo)  
**Próxima etapa:** Sprint 2 — DVC Setup