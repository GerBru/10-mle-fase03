"""Testes do módulo de promoção no MLflow Model Registry.

Usa um backend SQLite temporário por teste: o Model Registry exige store com
banco de dados (file store não suporta `register_model`), mas não exige um
servidor de tracking rodando. Isso mantém a suíte executável em CI.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.registry import (
    CHAMPION_ALIAS,
    STAGE_TAG_KEY,
    STAGE_TAG_VALUE,
    load_champion,
    log_and_register_champion,
)

MODEL_NAME = "churn-classifier-test"


@pytest.fixture
def raw_features() -> pd.DataFrame:
    """DataFrame cru com coluna numérica e categórica, espelhando o dataset real."""
    return pd.DataFrame(
        {
            "tenure": [1, 34, 2, 45, 8, 22, 60, 3],
            "contract": [
                "Month-to-month", "Two year", "Month-to-month", "Two year",
                "One year", "Month-to-month", "Two year", "One year",
            ],
        }
    )


@pytest.fixture
def labels() -> pd.Series:
    """Rótulos binários de churn."""
    return pd.Series([1, 0, 1, 0, 1, 1, 0, 0])


@pytest.fixture
def fitted_pipeline(raw_features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    """Pipeline autocontido (pré-processamento + classificador) já treinado."""
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", ["tenure"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract"]),
        ]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("clf", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(raw_features, labels)
    return pipeline


@pytest.fixture
def tracking_backend(tmp_path, monkeypatch):
    """Aponta o MLflow para um SQLite isolado e devolve um client conectado a ele."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    experiment_id = mlflow.create_experiment(
        "test-registry", artifact_location=str(tmp_path / "artifacts")
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    yield MlflowClient(tracking_uri=uri, registry_uri=uri)

    if mlflow.active_run():
        mlflow.end_run()


def test_registers_new_version_and_assigns_alias(
    tracking_backend, fitted_pipeline, raw_features
):
    """O fluxo feliz cria a versão 1 e atribui o alias de promoção."""
    with mlflow.start_run(run_name="parent"):
        version = log_and_register_champion(
            fitted_pipeline,
            raw_features,
            baseline_name="logistic_regression",
            metrics={"f1": 0.6205},
            model_name=MODEL_NAME,
        )

    assert version is not None
    assert int(version.version) == 1

    promoted = tracking_backend.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert promoted.version == version.version


def test_writes_stage_tag_and_description(
    tracking_backend, fitted_pipeline, raw_features
):
    """A versão registrada recebe a tag de estágio e a descrição com o critério."""
    with mlflow.start_run(run_name="parent"):
        version = log_and_register_champion(
            fitted_pipeline,
            raw_features,
            baseline_name="logistic_regression",
            metrics={"f1": 0.6205},
            model_name=MODEL_NAME,
        )

    stored = tracking_backend.get_model_version(MODEL_NAME, version.version)
    assert stored.tags[STAGE_TAG_KEY] == STAGE_TAG_VALUE
    assert "logistic_regression" in stored.description
    assert "0.6205" in stored.description


def test_alias_moves_to_latest_version(tracking_backend, fitted_pipeline, raw_features):
    """Promover duas vezes move o alias para a versão mais nova."""
    for _ in range(2):
        with mlflow.start_run():
            log_and_register_champion(
                fitted_pipeline,
                raw_features,
                baseline_name="logistic_regression",
                model_name=MODEL_NAME,
            )

    promoted = tracking_backend.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert int(promoted.version) == 2


def test_loads_champion_and_predicts_on_raw_data(
    tracking_backend, fitted_pipeline, raw_features
):
    """O modelo promovido é autocontido: prediz a partir do DataFrame cru."""
    with mlflow.start_run():
        log_and_register_champion(
            fitted_pipeline,
            raw_features,
            baseline_name="logistic_regression",
            model_name=MODEL_NAME,
        )

    loaded = load_champion(model_name=MODEL_NAME)
    predictions = loaded.predict(raw_features)

    assert len(predictions) == len(raw_features)
    assert set(predictions).issubset({0, 1})


def test_raises_when_model_is_missing(tracking_backend, raw_features):
    """Sem campeão, a etapa obrigatória interrompe o pipeline."""
    with mlflow.start_run(), pytest.raises(ValueError, match="No champion model"):
        log_and_register_champion(
            None, raw_features, baseline_name="", model_name=MODEL_NAME
        )


def test_raises_when_required_registry_is_unavailable(
    tracking_backend, fitted_pipeline, raw_features, monkeypatch
):
    """Falha do Registry obrigatório interrompe o pipeline."""

    def _raise(*args, **kwargs):
        raise MlflowException("registry unavailable")

    monkeypatch.setattr(mlflow, "register_model", _raise)

    with (
        mlflow.start_run(),
        pytest.raises(MlflowException, match="registry unavailable"),
    ):
        log_and_register_champion(
            fitted_pipeline,
            raw_features,
            baseline_name="logistic_regression",
            model_name=MODEL_NAME,
        )

    with pytest.raises(MlflowException):
        tracking_backend.get_registered_model(MODEL_NAME)


def test_optional_registry_failure_returns_none(
    tracking_backend, fitted_pipeline, raw_features, monkeypatch
):
    """Ambientes exploratórios podem optar explicitamente por promoção opcional."""

    def _raise(*args, **kwargs):
        raise MlflowException("registry unavailable")

    monkeypatch.setattr(mlflow, "register_model", _raise)
    with mlflow.start_run():
        version = log_and_register_champion(
            fitted_pipeline,
            raw_features,
            baseline_name="logistic_regression",
            model_name=MODEL_NAME,
            required=False,
        )
    assert version is None
