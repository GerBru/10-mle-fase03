"""HTTP integration tests for the Phase 1 API preserved in Phase 2."""

from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import pytest_asyncio

import src.api.app as app_module
from src.api.app import app
from src.api.security import API_KEY, create_access_token
from src.models.mlp import ChurnMLP

VALID_PAYLOAD = {
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
}

TEST_TOKEN = create_access_token(username="admin", role="admin")
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client():
    mock_pipeline = MagicMock()
    mock_pipeline.transform.side_effect = lambda frame: np.zeros(
        (len(frame), 30), dtype=np.float32
    )
    mock_model = ChurnMLP(input_dim=30, hidden_dims=[32, 16])
    mock_model.eval()
    app_module._state.update(
        {
            "pipeline": mock_pipeline,
            "model": mock_model,
            "model_source": "mlp",
            "model_metadata": {"framework": "pytorch"},
        }
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
        yield api
    app_module._state.update(
        {"pipeline": None, "model": None, "model_source": None, "model_metadata": {}}
    )


@pytest_asyncio.fixture
async def client_no_model():
    app_module._state.update({"pipeline": None, "model": None})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
        yield api


async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_source"] == "mlp"


async def test_predict_valid_payload_returns_200(client):
    response = await client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 200


async def test_predict_response_schema(client):
    response = await client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert set(response.json()) == {"churn_probability", "prediction", "risk_level"}


async def test_predict_probability_and_class_are_valid(client):
    response = await client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    result = response.json()
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["prediction"] in {0, 1}
    assert result["risk_level"] in {"low", "medium", "high"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("senior_citizen", 5), ("gender", "Other")],
)
async def test_predict_rejects_invalid_payload(client, field, value):
    response = await client.post(
        "/predict", json={**VALID_PAYLOAD, field: value}, headers=AUTH_HEADERS
    )
    assert response.status_code == 422


async def test_predict_rejects_missing_field(client):
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "tenure"}
    response = await client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


async def test_predict_model_not_loaded_returns_503(client_no_model):
    response = await client_no_model.post(
        "/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS
    )
    assert response.status_code == 503


async def test_predict_without_token_returns_401(client):
    assert (await client.post("/predict", json=VALID_PAYLOAD)).status_code == 401


async def test_auth_login_valid_credentials(client):
    response = await client.post("/auth/login?username=admin&password=admin123")
    assert response.status_code == 200
    assert "access_token" in response.json()


async def test_auth_login_invalid_credentials(client):
    response = await client.post("/auth/login?username=admin&password=wrong")
    assert response.status_code == 401


async def test_predict_apikey_valid(client):
    response = await client.post(
        "/predict-apikey", json=VALID_PAYLOAD, headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200


async def test_predict_apikey_invalid_key_returns_401(client):
    response = await client.post(
        "/predict-apikey", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


async def test_predict_batch_valid(client):
    response = await client.post(
        "/predict-batch",
        json=[VALID_PAYLOAD, VALID_PAYLOAD],
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2


async def test_predict_batch_empty_returns_422(client):
    response = await client.post("/predict-batch", json=[], headers=AUTH_HEADERS)
    assert response.status_code == 422


async def test_predict_batch_without_token_returns_401(client):
    response = await client.post("/predict-batch", json=[VALID_PAYLOAD])
    assert response.status_code == 401
