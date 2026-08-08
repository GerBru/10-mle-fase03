import numpy as np
import pytest
import torch

from src.models.mlp import MLP, MLPTrainer


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 20)).astype(np.float32)
    y = (rng.random(200) > 0.5).astype(np.float32)
    return X, y


def test_mlp_forward_output_shape():
    model = MLP(input_dim=20, hidden_dims=[64, 32])
    X = torch.randn(10, 20)
    out = model(X)
    assert out.shape == (10,), f"Expected (10,), got {out.shape}"


def test_mlp_forward_single_sample():
    model = MLP(input_dim=20, hidden_dims=[64, 32])
    model.eval()
    X = torch.randn(1, 20)
    out = model(X)
    assert out.shape == (1,)


def test_mlp_trainer_fit_runs(dummy_data):
    X, y = dummy_data
    trainer = MLPTrainer(input_dim=20, hidden_dims=[32, 16], max_epochs=5, patience=3)
    trainer.fit(X[:160], y[:160], X[160:], y[160:])
    assert len(trainer.history["train_loss"]) > 0
    assert len(trainer.history["val_loss"]) > 0


def test_mlp_predict_proba_in_range(dummy_data):
    X, y = dummy_data
    trainer = MLPTrainer(input_dim=20, hidden_dims=[32, 16], max_epochs=5, patience=3)
    trainer.fit(X[:160], y[:160], X[160:], y[160:])
    probs = trainer.predict_proba(X[160:])
    assert probs.min() >= 0.0, "Probabilities must be >= 0"
    assert probs.max() <= 1.0, "Probabilities must be <= 1"


def test_mlp_predict_binary_output(dummy_data):
    X, y = dummy_data
    trainer = MLPTrainer(input_dim=20, hidden_dims=[32, 16], max_epochs=5, patience=3)
    trainer.fit(X[:160], y[:160], X[160:], y[160:])
    preds = trainer.predict(X[160:])
    assert set(preds).issubset({0, 1}), f"Unexpected values in predictions: {set(preds)}"


def test_mlp_early_stopping(dummy_data):
    X, y = dummy_data
    trainer = MLPTrainer(
        input_dim=20, hidden_dims=[32], max_epochs=1000, patience=3, random_state=42
    )
    trainer.fit(X[:160], y[:160], X[160:], y[160:])
    # Early stopping deve ter pausado antes de 1000 epochs
    assert len(trainer.history["train_loss"]) < 1000
