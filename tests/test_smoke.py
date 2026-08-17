"""Smoke tests: pipeline, MLP e baselines — verificação rápida de integridade."""

import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessing_pipeline
from src.models.baseline import get_baselines
from src.models.evaluation import evaluate_model
from src.models.mlp import ChurnMLP, predict_proba, train_mlp


def _make_dummy_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Tenure Months": rng.integers(0, 72, n),
        "Monthly Charges": rng.uniform(20, 120, n),
        "Total Charges": rng.uniform(0, 8000, n).astype(str),
        "Senior Citizen": rng.choice(["Yes", "No"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "Phone Service": rng.choice(["Yes", "No"], n),
        "Paperless Billing": rng.choice(["Yes", "No"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Multiple Lines": rng.choice(["Yes", "No", "No phone service"], n),
        "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Online Security": rng.choice(["Yes", "No", "No internet service"], n),
        "Online Backup": rng.choice(["Yes", "No", "No internet service"], n),
        "Device Protection": rng.choice(["Yes", "No", "No internet service"], n),
        "Tech Support": rng.choice(["Yes", "No", "No internet service"], n),
        "Streaming TV": rng.choice(["Yes", "No", "No internet service"], n),
        "Streaming Movies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Payment Method": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n),
    })


def test_pipeline_fit_transform():
    X = _make_dummy_df()
    y = np.random.randint(0, 2, len(X))
    pipeline = build_preprocessing_pipeline()
    X_t = pipeline.fit_transform(X, y)
    assert X_t.shape[0] == len(X)
    assert X_t.shape[1] >= 10
    assert not np.isnan(X_t).any(), "NaNs found after preprocessing"


def test_pipeline_shape_stable():
    X_train = _make_dummy_df(n=100)
    X_val = _make_dummy_df(n=20)
    y = np.random.randint(0, 2, len(X_train))
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train, y)
    assert pipeline.transform(X_train).shape[1] == pipeline.transform(X_val).shape[1]


def test_mlp_smoke():
    rng = np.random.default_rng(1)
    n, d = 120, 20
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)

    model, history = train_mlp(X[:80], y[:80], X[80:100], y[80:100], max_epochs=5, patience=3)

    assert len(history["train_loss"]) > 0
    probs = predict_proba(model, X)
    assert probs.shape == (n,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_churn_mlp_forward_shape():
    model = ChurnMLP(input_dim=20, hidden_dims=[32, 16])
    import torch
    X = torch.randn(10, 20)
    out = model(X)
    assert out.shape == (10,)


def test_baselines_fit_predict():
    rng = np.random.default_rng(2)
    n, d = 100, 20
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, n)

    for name, model in get_baselines().items():
        model.fit(X, y)
        preds = model.predict(X)
        probas = model.predict_proba(X)[:, 1]
        assert preds.shape == (n,), f"{name}: wrong prediction shape"
        assert probas.shape == (n,), f"{name}: wrong proba shape"


def test_evaluate_model_keys():
    y = np.array([0, 1, 0, 1, 1])
    pred = np.array([0, 1, 0, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
    metrics = evaluate_model(y, pred, proba)
    for key in ("accuracy", "roc_auc", "pr_auc", "f1", "tp", "fp", "tn", "fn"):
        assert key in metrics, f"Missing key: {key}"
