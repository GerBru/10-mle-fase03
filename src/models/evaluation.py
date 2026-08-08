"""Métricas de avaliação de modelos de classificação."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas completas incluindo PR-AUC e componentes da matriz de confusão.

    Args:
        y_true: Labels reais com shape (n_samples,).
        y_pred: Predições binárias com shape (n_samples,).
        y_proba: Probabilidades da classe positiva com shape (n_samples,).

    Returns:
        Dicionário com accuracy, roc_auc, pr_auc, f1, precision, recall, tp, fp, tn, fn.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Subconjunto de métricas para uso no training pipeline.

    Delega para evaluate_model() e retorna apenas as chaves relevantes para
    comparação de modelos: accuracy, f1, precision, recall, e opcionalmente
    auc_roc e pr_auc quando y_prob é fornecido.
    """
    if y_prob is not None:
        base = evaluate_model(y_true, y_pred, y_prob)
        return {
            "accuracy": base["accuracy"],
            "f1": base["f1"],
            "precision": base["precision"],
            "recall": base["recall"],
            "auc_roc": base["roc_auc"],
            "pr_auc": base["pr_auc"],
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
