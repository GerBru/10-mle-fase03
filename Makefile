.PHONY: install lint test train run fairness clean dvc-setup repro dvc-push

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
	uv run python -m src.training.train

# Grava as credenciais OAuth do Google Drive em .dvc/config.local (fora do git).
# Defina GDRIVE_CLIENT_ID e GDRIVE_CLIENT_SECRET (ex.: exportados do seu .env)
# do seu OAuth Client "Desktop" criado no Google Cloud Console.
dvc-setup:
	uv run dvc remote modify --local storage gdrive_client_id $(GDRIVE_CLIENT_ID)
	uv run dvc remote modify --local storage gdrive_client_secret $(GDRIVE_CLIENT_SECRET)

# Reproduz o pipeline completo (preprocess -> train), pulando estágios sem mudança.
repro:
	uv run dvc repro

# Envia os artefatos versionados para o remote (Google Drive).
dvc-push:
	uv run dvc push

run:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001

fairness:
	uv run python -m src.monitoring.fairness

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
