"""Testes para modelos e avaliação - Sprint 1."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baseline import build_baselines, get_baselines, train_baseline
from src.models.evaluation import compute_metrics, evaluate_model
from src.models.mlp import MLPTrainer


@pytest.fixture
def sample_data():
    """Dados de exemplo para testes."""
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, random_state=42
    )
    return X, y


@pytest.fixture
def split_data(sample_data):
    """Dados divididos em treino/teste."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ============================================================================
# MLP Tests - Comportamento Real
# ============================================================================

def test_mlp_predict_proba_in_range(split_data):
    """MLP predict_proba retorna valores entre 0 e 1."""
    X_train, X_test, y_train, y_test = split_data
    trainer = MLPTrainer(input_dim=20, hidden_dims=[32], epochs=2, random_seed=42)
    trainer.fit(X_train, y_train, X_test, y_test)
    proba = trainer.predict_proba(X_test)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_mlp_predict_binary_output(split_data):
    """MLP predict retorna 0 ou 1."""
    X_train, X_test, y_train, y_test = split_data
    trainer = MLPTrainer(input_dim=20, hidden_dims=[32], epochs=2, random_seed=42)
    trainer.fit(X_train, y_train, X_test, y_test)
    pred = trainer.predict(X_test)
    assert np.all(np.isin(pred, [0, 1]))


def test_mlp_trainer_fit_runs(split_data):
    """MLP trainer consegue treinar sem erro."""
    X_train, X_test, y_train, y_test = split_data
    trainer = MLPTrainer(
        input_dim=20, hidden_dims=[32], learning_rate=0.01,
        epochs=2, batch_size=32, random_seed=42
    )
    trainer.fit(X_train, y_train, X_test, y_test)


def test_mlp_early_stopping(split_data):
    """MLP com early stopping roda sem erro."""
    X_train, X_test, y_train, y_test = split_data
    trainer = MLPTrainer(
        input_dim=20, hidden_dims=[32], epochs=200,
        patience=3, random_seed=42
    )
    trainer.fit(X_train, y_train, X_test, y_test)


# ============================================================================
# Baseline Tests
# ============================================================================

def test_get_baselines_returns_dict():
    """get_baselines retorna dict."""
    baselines = get_baselines()
    assert isinstance(baselines, dict)
    assert len(baselines) > 0


def test_build_baselines_returns_list():
    """build_baselines retorna lista."""
    baselines = build_baselines()
    assert isinstance(baselines, list)
    assert len(baselines) > 0


def test_train_baseline_with_pipeline(split_data):
    """train_baseline com pipeline."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    result = train_baseline(
        pipeline=pipeline, X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test, model_name="test_lr", params={"C": 1.0}
    )
    assert isinstance(result, dict)


# ============================================================================
# Evaluation Tests
# ============================================================================

def test_compute_metrics_without_proba(split_data):
    """compute_metrics sem probabilidades."""
    X_train, X_test, y_train, y_test = split_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob=None)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics and "f1" in metrics


def test_compute_metrics_with_proba(split_data):
    """compute_metrics com probabilidades."""
    X_train, X_test, y_train, y_test = split_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob=y_proba)
    assert isinstance(metrics, dict) and "auc_roc" in metrics


def test_evaluate_model_structure(split_data):
    """evaluate_model retorna dict com métricas."""
    X_train, X_test, y_train, y_test = split_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    result = evaluate_model(y_test, y_pred, y_proba)
    assert isinstance(result, dict) and "roc_auc" in result


def test_metrics_values_valid(split_data):
    """Métricas estão em ranges válidos."""
    X_train, X_test, y_train, y_test = split_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob=y_proba)
    assert 0 <= metrics.get("auc_roc", 0) <= 1
