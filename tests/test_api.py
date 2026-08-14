import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import app
from src.api.security import create_access_token
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

# Token JWT válido para os testes
TEST_TOKEN = create_access_token(username="admin", role="admin")
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def client():
    from unittest.mock import MagicMock

    mock_pipeline = MagicMock()
    mock_pipeline.transform.side_effect = lambda df: np.zeros((len(df), 30), dtype=np.float32)

    mock_model = ChurnMLP(input_dim=30, hidden_dims=[32, 16])
    mock_model.eval()

    with TestClient(app) as c:
        app_module._state["pipeline"] = mock_pipeline
        app_module._state["model"] = mock_model
        yield c

    app_module._state["pipeline"] = None
    app_module._state["model"] = None


@pytest.fixture
def client_no_model():
    with TestClient(app) as c:
        app_module._state["pipeline"] = None
        app_module._state["model"] = None
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "version" in data


def test_predict_valid_payload_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_level" in data


def test_predict_probability_in_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_binary_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    prediction = response.json()["prediction"]
    assert prediction in {0, 1}


def test_predict_risk_level_valid(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    risk = response.json()["risk_level"]
    assert risk in {"low", "medium", "high"}


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_predict_invalid_senior_citizen_returns_422(client):
    payload = {**VALID_PAYLOAD, "senior_citizen": 5}
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_predict_invalid_categorical_returns_422(client):
    payload = {**VALID_PAYLOAD, "gender": "Other"}
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_predict_model_not_loaded_returns_503(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"


def test_predict_without_token_returns_401(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401


def test_auth_login_valid_credentials(client):
    response = client.post("/auth/login?username=admin&password=admin123")
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_auth_login_invalid_credentials(client):
    response = client.post("/auth/login?username=admin&password=wrong")
    assert response.status_code == 401


def test_predict_apikey_valid(client):
    response = client.post(
        "/predict-apikey",
        json=VALID_PAYLOAD,
        headers={"X-API-Key": "churn-api-key-fiap-2026"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["prediction"] in {0, 1}


def test_predict_apikey_invalid_key_returns_401(client):
    response = client.post(
        "/predict-apikey",
        json=VALID_PAYLOAD,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_batch_valid(client):
    response = client.post(
        "/predict-batch",
        json=[VALID_PAYLOAD, VALID_PAYLOAD],
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty_returns_422(client):
    response = client.post("/predict-batch", json=[], headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_predict_batch_without_token_returns_401(client):
    response = client.post("/predict-batch", json=[VALID_PAYLOAD])
    assert response.status_code == 401
