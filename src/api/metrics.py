"""Objetos Prometheus para observabilidade da API de churn."""

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total de predições realizadas",
    ["auth_method", "risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latência das predições em segundos",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REQUEST_LATENCY = Histogram(
    "churn_request_latency_seconds",
    "Latência total das requisições em segundos",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
REQUESTS_TOTAL = Counter(
    "churn_requests_total",
    "Total de requisições recebidas",
    ["method", "endpoint", "status"],
)
MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "Indica se o modelo está carregado (1=sim, 0=não)",
)
LOGIN_ATTEMPTS = Counter(
    "churn_login_attempts_total",
    "Total de tentativas de login",
    ["status"],
)
RATE_LIMIT_HITS = Counter(
    "churn_rate_limit_hits_total",
    "Total de requisições bloqueadas por rate limiting",
)
CHURN_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribuição das probabilidades de churn preditas",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
