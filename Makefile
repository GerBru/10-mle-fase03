.PHONY: install lint test train run fairness clean

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
