.PHONY: install lint test train run clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

train:
	uv run python -m src.training.train

run:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
