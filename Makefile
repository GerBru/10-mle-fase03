.PHONY: install lint test train run fairness clean dvc-setup repro dvc-push dvc-pull dvc-use-local dvc-use-gdrive

# Força UTF-8 no Python: o MLflow imprime URLs de run com emoji (🏃) que quebram em terminais Windows com codificação legada(cp1252).

export PYTHONUTF8 := 1

install:
	uv sync --extra dev

lint:
	uv run ruff check  .

lint-fix:
	uv run ruff check --fix .

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

train:
	uv run --extra train python -m src.training.train

# Grava as credenciais OAuth do Google Drive em .dvc/config.local (fora do git).
# Defina GDRIVE_CLIENT_ID e GDRIVE_CLIENT_SECRET (ex.: exportados do seu .env)
# do seu OAuth Client "Desktop" criado no Google Cloud Console.
dvc-setup:
	uv run --extra train dvc remote modify --local gdrive gdrive_client_id $(GDRIVE_CLIENT_ID)
	uv run --extra train dvc remote modify --local gdrive gdrive_client_secret $(GDRIVE_CLIENT_SECRET)

# ── Escolha do remote (por máquina, grava em .dvc/config.local, fora do git) ──
# Depois de escolher, `dvc push`/`dvc pull`/`make dvc-push` usam esse remote.
dvc-use-local:
	uv run --extra train dvc remote default --local local
	@echo "Remote ativo: local (../dvc-storage-fase2)"

dvc-use-gdrive:
	uv run --extra train dvc remote default --local gdrive
	@echo "Remote ativo: gdrive (Google Drive)"

# Reproduz o pipeline completo (preprocess -> train), pulando estágios sem mudança.
repro:
	uv run --extra train dvc repro

# Envia/baixa os artefatos versionados no remote ativo (local por padrão).
dvc-push:
	uv run --extra train dvc push

dvc-pull:
	uv run --extra train dvc pull

run:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	uv run --extra train mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001

fairness:
	uv run python -m src.monitoring.fairness

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
