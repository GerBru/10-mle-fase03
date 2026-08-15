# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Instala dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala uv para gerenciamento de dependências
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copia arquivos de dependências
COPY pyproject.toml uv.lock ./

# Instala apenas dependências de produção (sem [dev]) com cache do Docker
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY pyproject.toml ./
COPY models/ ./models/

# Os artefatos de modelo sao gerenciados pelo DVC (nao ficam no git). Falha o
# build cedo se estiverem ausentes do contexto -- evita publicar uma imagem
# que sobe saudavel mas nunca carrega o modelo (ver `dvc pull` no workflow de CD).
RUN test -f models/preprocessor.joblib && \
    test -f models/mlp_model.pt && \
    test -f models/model_config.json

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 1 worker: cada worker carrega PyTorch + modelo na memoria; multiplos workers
# duplicam o uso de RAM e causam OOM em tasks Fargate pequenas (256/512).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]