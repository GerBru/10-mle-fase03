"""Promoção de modelos no MLflow Model Registry — FIAP Tech Challenge Fase 2.

Este módulo isola toda a interação com o Model Registry, mantendo o
`src/training/train.py` responsável apenas pela orquestração do treino.

O modelo promovido é o melhor baseline clássico (scikit-learn), que carrega o
pré-processamento embutido no próprio ``Pipeline`` e, portanto, prediz a partir
dos dados crus.

Promoção usa **alias** (`@champion`), não *stages* — depreciados desde o
MLflow 2.9. Uma tag textual `stage=production` é gravada na versão para manter
a legibilidade do estágio na UI sem invocar a API depreciada.

Uso:
    >>> from src.models.registry import log_and_register_champion
    >>> log_and_register_champion(best_pipeline, X_test_df, baseline_name="random_forest")
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from src.utils.logger import get_logger

logger = get_logger(__name__)

REGISTERED_MODEL_NAME = "churn-classifier"
CHAMPION_ALIAS = "champion"
STAGE_TAG_KEY = "stage"
STAGE_TAG_VALUE = "production"
ARTIFACT_NAME = "model"
INPUT_EXAMPLE_ROWS = 5


def log_and_register_champion(
    model: Any,
    X_sample: pd.DataFrame,
    *,
    baseline_name: str,
    metrics: dict[str, float] | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
) -> ModelVersion | None:
    """Loga o melhor baseline sklearn e o promove no Model Registry.

    Abre um nested run dedicado, registra o modelo com assinatura inferida,
    cria uma nova versão no Registry e atribui o alias de promoção.

    Falhas de Registry são capturadas e logadas em vez de propagadas: em CI ou
    com tracking server indisponível o treino deve terminar normalmente.

    Args:
        model: Pipeline sklearn treinado (pré-processamento + classificador).
        X_sample: Amostra de features cruas, usada para inferir a assinatura
            e gravar o input example.
        baseline_name: Nome do baseline vencedor, gravado como tag.
        metrics: Métricas de teste do modelo, logadas no nested run.
        model_name: Nome do modelo registrado no Registry.
        alias: Alias de promoção atribuído à nova versão.

    Returns:
        A ``ModelVersion`` criada, ou ``None`` se o modelo for inválido ou o
        Registry estiver indisponível.

    Example:
        >>> mv = log_and_register_champion(pipe, X_test_df, baseline_name="random_forest")
        >>> mv.version
        '1'
    """
    if model is None:
        logger.warning("No champion model to register — skipping Registry step.")
        return None

    with mlflow.start_run(run_name="sklearn_champion", nested=True):
        mlflow.set_tag("model_type", "classical_ml")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("baseline_name", baseline_name)

        if metrics:
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)

        example = X_sample.head(INPUT_EXAMPLE_ROWS).astype(
            {c: "float64" for c in X_sample.select_dtypes("int").columns}
        )
        signature = infer_signature(example, model.predict(example))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=ARTIFACT_NAME,
            signature=signature,
            input_example=example,
        )
        logger.info("Champion model logged at {}", model_info.model_uri)

        return _promote(model_info.model_uri, model_name, alias, baseline_name, metrics)


def _promote(
    model_uri: str,
    model_name: str,
    alias: str,
    baseline_name: str,
    metrics: dict[str, float] | None,
) -> ModelVersion | None:
    """Cria a versão no Registry e atribui alias, tag de estágio e descrição.

    Args:
        model_uri: URI do modelo logado, obtido do ``ModelInfo``.
        model_name: Nome do modelo registrado.
        alias: Alias de promoção.
        baseline_name: Nome do baseline vencedor, usado na descrição.
        metrics: Métricas de teste, usadas na descrição.

    Returns:
        A ``ModelVersion`` criada, ou ``None`` em caso de falha do Registry.
    """
    try:
        version = mlflow.register_model(model_uri, model_name)
        client = MlflowClient()
        client.set_registered_model_alias(model_name, alias, version.version)
        client.set_model_version_tag(
            model_name, version.version, STAGE_TAG_KEY, STAGE_TAG_VALUE
        )
        client.update_model_version(
            model_name,
            version.version,
            description=_build_description(baseline_name, metrics),
        )
    except MlflowException:
        logger.exception(
            "Model Registry unavailable — model logged but not promoted. "
            "Is the tracking server running with a database backend?"
        )
        return None

    logger.info(
        "Registered {} v{} and promoted to @{}", model_name, version.version, alias
    )
    return version


def _build_description(baseline_name: str, metrics: dict[str, float] | None) -> str:
    """Monta a descrição textual da versão registrada.

    Args:
        baseline_name: Nome do baseline vencedor.
        metrics: Métricas de teste do modelo.

    Returns:
        Descrição legível, com F1 quando disponível.
    """
    base = f"Melhor baseline sklearn ({baseline_name}), pipeline autocontido."
    if metrics and "f1" in metrics:
        return f"{base} F1 de teste: {metrics['f1']:.4f}."
    return base


def load_champion(
    model_name: str = REGISTERED_MODEL_NAME, alias: str = CHAMPION_ALIAS
) -> Any:
    """Carrega a versão promovida do modelo a partir do Registry.

    Args:
        model_name: Nome do modelo registrado.
        alias: Alias da versão desejada.

    Returns:
        O pipeline sklearn pronto para ``predict`` sobre dados crus.

    Example:
        >>> model = load_champion()
        >>> model.predict(df_raw)
    """
    return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
