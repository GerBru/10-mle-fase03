"""Análise de fairness usando Fairlearn MetricFrame.

Padrão de implementação do curso (Etapa 1, Aula 07):
    mf = MetricFrame(metrics=<fn>, y_true=..., y_pred=..., sensitive_features=...)
    mf.difference()   # disparidade máxima entre grupos

Atributos sensíveis avaliados (monitoring_plan.md §4):
  - gender         (Male / Female)
  - senior_citizen (0 / 1)
  - contract       (Month-to-month / One year / Two year)

Uso direto: python -m src.monitoring.fairness
"""

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)

logger = logging.getLogger(__name__)

SENSITIVE_FEATURES = ["gender", "senior_citizen", "contract"]

# Disparidade máxima entre grupos > 10pp requer revisão (monitoring_plan.md §4)
DIFFERENCE_THRESHOLD = 0.10


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    feature_name: str,
) -> dict:
    """Calcula disparidade por grupo para um único atributo sensível.

    Usa o padrão MetricFrame + mf.difference() do curso para cada métrica:
      - false_negative_rate : taxa de churners não detectados por grupo
      - false_positive_rate : taxa de falsos alarmes por grupo
      - selection_rate      : taxa de predições positivas por grupo (demographic parity)

    Args:
        y_true:       Labels reais, shape (n,).
        y_pred:       Predições binárias, shape (n,).
        sensitive:    Série com os valores do atributo sensível.
        feature_name: Nome do atributo (para logging e relatório).

    Returns:
        dict com by_group (DataFrame), difference por métrica e flag de alerta.
    """
    metrics = {
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "selection_rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    # mf.difference() — diferença máxima entre grupos, por métrica
    differences = mf.difference()

    alert = any(abs(differences[m]) > DIFFERENCE_THRESHOLD for m in differences.index)

    for metric, diff in differences.items():
        logger.info(
            "Fairness [%s] %s difference=%.4f alert=%s",
            feature_name,
            metric,
            diff,
            "SIM" if abs(diff) > DIFFERENCE_THRESHOLD else "NAO",
        )
    if alert:
        logger.warning(
            "ALERTA DE FAIRNESS em '%s' — disparidade entre grupos supera %.0f pp.",
            feature_name,
            DIFFERENCE_THRESHOLD * 100,
        )

    return {
        "feature": feature_name,
        "by_group": mf.by_group,
        "overall": mf.overall,
        "difference": differences,
        "alert": alert,
    }


def compute_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, dict]:
    """Executa análise de fairness para todos os atributos sensíveis disponíveis.

    Args:
        y_true:        Labels reais, shape (n,).
        y_pred:        Predições binárias, shape (n,).
        sensitive_df:  DataFrame com colunas dos atributos sensíveis.
        features:      Colunas a avaliar. Padrão: SENSITIVE_FEATURES.

    Returns:
        dict {feature_name: resultado de compute_group_metrics()}.
    """
    if features is None:
        features = SENSITIVE_FEATURES

    available = [f for f in features if f in sensitive_df.columns]
    missing = set(features) - set(available)
    if missing:
        logger.warning("Atributos sensíveis ausentes e ignorados: %s", missing)

    report = {}
    for feature in available:
        report[feature] = compute_group_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive_df[feature],
            feature_name=feature,
        )

    alerts = [f for f, r in report.items() if r["alert"]]
    if alerts:
        logger.warning("Resumo: %d atributo(s) com alerta — %s", len(alerts), alerts)
    else:
        logger.info("Resumo: nenhum alerta de fairness detectado.")

    return report


def print_report(report: dict[str, dict]) -> None:
    """Exibe o relatório de fairness no terminal."""
    for feature, result in report.items():
        print(f"\n{'='*60}")
        print(f"Atributo sensível: {feature.upper()}")
        print(f"{'='*60}")
        print("\n  mf.difference() — disparidade máxima entre grupos:")
        for metric, diff in result["difference"].items():
            flag = "  ⚠" if abs(diff) > DIFFERENCE_THRESHOLD else ""
            print(f"    {metric:<22}: {diff:+.4f}{flag}")
        print(f"\n  Alerta geral: {'⚠ SIM' if result['alert'] else 'OK'}")
        print("\n  Métricas por grupo (mf.by_group):")
        print(result["by_group"].to_string())
        print("\n  Métricas gerais (mf.overall):")
        print(result["overall"].to_string())


# ─── Execução standalone com dados e modelo reais ─────────────────────────────

if __name__ == "__main__":
    import joblib
    import torch

    from src.data.preprocessing import clean_data, load_data, split_data
    from src.models.mlp import ChurnMLP, predict_proba
    from src.utils import settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    MODELS_DIR = "models"
    MODEL_PATH = f"{MODELS_DIR}/mlp_model.pt"
    PIPELINE_PATH = f"{MODELS_DIR}/preprocessor.joblib"

    # 1. Carrega e divide — mesmo fluxo do train.py
    logger.info("Carregando dados de %s", settings.data_path)
    df = load_data(settings.data_path)
    df = clean_data(df)
    _, _, X_test_df, _, _, y_test = split_data(df)

    # 2. Features sensíveis do DataFrame cru (valores legíveis, antes do preprocessing)
    sensitive_df = X_test_df[SENSITIVE_FEATURES].copy()

    # 3. Transforma X_test com o pipeline salvo
    pipeline = joblib.load(PIPELINE_PATH)
    X_test = pipeline.transform(X_test_df)

    # 4. Carrega MLP e gera predições
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = ChurnMLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
    model.load_state_dict(ckpt["state_dict"])

    y_proba = predict_proba(model, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    # 5. Relatório de fairness
    logger.info("Conjunto de teste: %d amostras", len(y_test))
    report = compute_fairness_report(y_test.values, y_pred, sensitive_df)
    print_report(report)
