"""
Churn Prediction API — FIAP Tech Challenge Fase 1

Segurança implementada:
- API Key via header X-API-Key (autenticação simples)
- JWT (JSON Web Token) via header Authorization: Bearer <token>
- Rate Limiting: máximo de requisições por IP por janela de tempo
- CORS: controle de origens permitidas
- Logging estruturado (loguru) sem print()
- Middleware de latência (X-Latency-Ms)

Observabilidade:
- trace_id por requisição (X-Trace-ID)
- Métricas Prometheus (/metrics)
- Logs JSON estruturados com trace_id
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.metrics import (
    CHURN_PROBABILITY,
    LOGIN_ATTEMPTS,
    MODEL_LOADED,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
    RATE_LIMIT_HITS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
)
from src.api.model_loader import LocalModelRepository, ModelRepository
from src.api.prediction_service import PredictionService
from src.api.schemas import (
    BatchPredictionOutput,
    ClienteInput,
    HealthOutput,
    PredictionOutput,
)
from src.api.security import (
    JWT_EXPIRE_MINUTES,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    InMemoryUserRepository,
    check_rate_limit,
    create_access_token,
    remaining_requests,
    verify_api_key,
    verify_token,
)
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

user_repo = InMemoryUserRepository()

# ── Estado da aplicação ───────────────────────────────────────────────────────
_state: dict = {"pipeline": None, "model": None, "input_dim": None}

BATCH_INPUT_EXAMPLE = [
    {
        "senior_citizen": 0,
        "tenure": 12,
        "monthly_charges": 65.5,
        "total_charges": 786.0,
        "gender": "Male",
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    },
    {
        "senior_citizen": 1,
        "tenure": 60,
        "monthly_charges": 45.0,
        "total_charges": 2700.0,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
    },
]


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    repo: ModelRepository = LocalModelRepository(Path(settings.models_dir))
    logger.info("Loading model artifacts from {}", str(settings.models_dir))
    try:
        loaded = repo.load()
        _state.update(loaded)
        MODEL_LOADED.set(1)
    except Exception as exc:
        MODEL_LOADED.set(0)
        logger.error("Failed to load model artifacts: {}", exc)
    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="""
API de predição de churn para operadora de telecomunicações.

**FIAP Tech Challenge Fase 1**

## Autenticação
- **JWT**: `POST /auth/login` → use o token em `Authorization: Bearer <token>`
- **API Key**: header `X-API-Key: <chave>`

## Observabilidade
- Métricas Prometheus em `/metrics`
- `X-Trace-ID` em todas as respostas
- `X-Latency-Ms` em todas as respostas
""",
    version=settings.api_version,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Monitoring", "description": "Health checks e métricas"},
        {"name": "Auth", "description": "Autenticação e tokens JWT"},
        {"name": "Inference", "description": "Predições de churn"},
    ],
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middlewares ───────────────────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    blocked, retry_after = check_rate_limit(client_ip)
    if blocked:
        RATE_LIMIT_HITS.inc()
        logger.warning("Rate limit exceeded for IP={}", client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Limite excedido: {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW}s",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining_requests(client_ip))
    return response


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware de observabilidade — trace_id, latência e métricas Prometheus."""
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.perf_counter()

    response = await call_next(request)

    latency = time.perf_counter() - start
    latency_ms = latency * 1000
    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)

    # Métricas Prometheus
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status=status_code).observe(latency)
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status_code).inc()

    # Logging estruturado com trace_id
    logger.info(
        "trace_id={} method={} path={} status={} latency_ms={:.2f}",
        trace_id, method, endpoint, status_code, latency_ms,
    )

    # Headers de observabilidade
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Dependency: modelo disponível ─────────────────────────────────────────────
def _require_model(request: Request) -> PredictionService:
    if _state["model"] is None or _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return PredictionService(pipeline=_state["pipeline"], model=_state["model"])


ModelState = Annotated[PredictionService, Depends(_require_model)]


# ── Endpoints públicos ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
async def health():
    return HealthOutput(
        status="ok",
        model_loaded=_state["model"] is not None,
        version=settings.api_version,
    )


@app.get("/ready", tags=["Monitoring"])
async def ready():
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics", tags=["Monitoring"], summary="Métricas Prometheus")
async def metrics():
    """Expõe métricas no formato Prometheus para scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"])
async def login(username: str, password: str):
    """Login e geração de token JWT. Usuários: admin/admin123, user/user123"""
    user = user_repo.authenticate(username, password)
    if not user:
        LOGIN_ATTEMPTS.labels(status="failed").inc()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuário ou senha incorretos.")
    LOGIN_ATTEMPTS.labels(status="success").inc()
    token = create_access_token(username, user["role"])
    logger.info("Login successful for user={}", username)
    return {"access_token": token, "token_type": "bearer", "expires_in": JWT_EXPIRE_MINUTES * 60}


@app.get("/auth/me", tags=["Auth"])
async def me(current_user: dict = Depends(verify_token)):
    return {"username": current_user["username"], "role": current_user["role"], "message": "Autenticado!"}


# ── Inferência ────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict(
    cliente: ClienteInput,
    service: ModelState,
    current_user: dict = Depends(verify_token),
):
    """Predição de churn para um cliente. Requer JWT."""
    try:
        start = time.perf_counter()
        prob, prediction, risk = service.predict(cliente)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
    except Exception as exc:
        logger.error("Prediction failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    PREDICTIONS_TOTAL.labels(auth_method="jwt", risk_level=risk).inc()
    CHURN_PROBABILITY.observe(prob)
    logger.info("user={} churn_prob={:.4f} pred={} risk={}", current_user["username"], prob, prediction, risk)
    return PredictionOutput(churn_probability=round(prob, 4), prediction=prediction, risk_level=risk)


@app.post("/predict-apikey", response_model=PredictionOutput, tags=["Inference"])
async def predict_apikey(
    cliente: ClienteInput,
    service: ModelState,
    _: str = Depends(verify_api_key),
):
    """Predição de churn para um cliente. Requer API Key."""
    try:
        start = time.perf_counter()
        prob, prediction, risk = service.predict(cliente)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    PREDICTIONS_TOTAL.labels(auth_method="api_key", risk_level=risk).inc()
    CHURN_PROBABILITY.observe(prob)
    return PredictionOutput(churn_probability=round(prob, 4), prediction=prediction, risk_level=risk)


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(
    clientes: Annotated[
        list[ClienteInput],
        Body(
            ...,
            openapi_examples={
                "default": {
                    "summary": "Exemplo de batch com 2 clientes",
                    "value": BATCH_INPUT_EXAMPLE,
                }
            },
        ),
    ],
    service: ModelState,
    current_user: dict = Depends(verify_token),
):
    """Predição em batch para múltiplos clientes (1 a 1000). Requer JWT."""
    if len(clientes) == 0 or len(clientes) > 1000:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 1000")

    try:
        start = time.perf_counter()
        predictions = service.predict_batch(clientes)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    logger.info("user={} batch_size={}", current_user["username"], len(clientes))
    for pred in predictions:
        PREDICTIONS_TOTAL.labels(auth_method="jwt_batch", risk_level=pred.risk_level).inc()
        CHURN_PROBABILITY.observe(pred.churn_probability)

    return BatchPredictionOutput(predictions=predictions, count=len(predictions))
