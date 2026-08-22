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

# ── Stage 2: DVC training pipeline ───────────────────────────────────────────
FROM builder AS training-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra train --no-dev --frozen

FROM python:3.12-slim AS trainer

WORKDIR /app
RUN useradd -m -u 1000 appuser \
    && chown appuser:appuser /app
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --chown=appuser:appuser --from=training-builder /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml uv.lock params.yaml dvc.yaml dvc.lock .dvcignore ./
COPY --chown=appuser:appuser .dvc/ ./.dvc/
RUN /app/.venv/bin/dvc config core.no_scm true \
    && mkdir -p data models .dvc/cache \
    && chown -R appuser:appuser .dvc
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["dvc", "repro", "--force"]

# ── Stage 3: API runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN useradd -m -u 1000 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser models/ ./models/

# Os artefatos de modelo sao gerenciados pelo DVC (nao ficam no git). Falha o
# build cedo se estiverem ausentes do contexto -- evita publicar uma imagem
# que sobe saudavel mas nunca carrega o modelo (ver `dvc pull` no workflow de CD).
RUN test -f models/preprocessor.joblib && \
    test -f models/mlp_model.pt && \
    test -f models/model_config.json && \
    test -f models/best_baseline.joblib && \
    test -f models/registry.json

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
