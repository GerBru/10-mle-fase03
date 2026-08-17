"""Smoke tests para src/monitoring/fairness.py.

Usa dados sintéticos — não depende de modelo treinado nem do dataset real.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.fairness import (
    DIFFERENCE_THRESHOLD,
    SENSITIVE_FEATURES,
    compute_fairness_report,
    compute_group_metrics,
    print_report,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    sensitive_df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "senior_citizen": rng.integers(0, 2, n),
            "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        }
    )
    return y_true, y_pred, sensitive_df


def test_compute_group_metrics_keys(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    result = compute_group_metrics(y_true, y_pred, sensitive_df["gender"], "gender")
    assert set(result.keys()) == {"feature", "by_group", "overall", "difference", "alert"}


def test_compute_group_metrics_metrics_present(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    result = compute_group_metrics(y_true, y_pred, sensitive_df["gender"], "gender")
    for metric in ("false_negative_rate", "false_positive_rate", "selection_rate"):
        assert metric in result["difference"].index


def test_compute_group_metrics_by_group_shape(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    result = compute_group_metrics(y_true, y_pred, sensitive_df["gender"], "gender")
    assert result["by_group"].shape[0] == 2  # Male e Female
    assert result["by_group"].shape[1] == 3  # três métricas


def test_compute_group_metrics_alert_is_bool(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    result = compute_group_metrics(y_true, y_pred, sensitive_df["gender"], "gender")
    assert isinstance(result["alert"], bool)


def test_compute_fairness_report_all_features(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    report = compute_fairness_report(y_true, y_pred, sensitive_df)
    assert set(report.keys()) == set(SENSITIVE_FEATURES)


def test_compute_fairness_report_missing_feature(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    partial_df = sensitive_df[["gender", "senior_citizen"]]
    report = compute_fairness_report(y_true, y_pred, partial_df)
    assert "gender" in report
    assert "senior_citizen" in report
    assert "contract" not in report


def test_compute_fairness_report_custom_features(synthetic_data):
    y_true, y_pred, sensitive_df = synthetic_data
    report = compute_fairness_report(
        y_true, y_pred, sensitive_df, features=["gender"]
    )
    assert list(report.keys()) == ["gender"]


def test_alert_triggers_when_disparity_exceeds_threshold():
    """Garante que alert=True quando a disparidade claramente supera o threshold."""
    n = 200
    y_true = np.array([1] * n)
    # Grupo A: o modelo acerta todos; Grupo B: o modelo erra todos
    y_pred = np.array([1] * (n // 2) + [0] * (n // 2))
    sensitive = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))
    result = compute_group_metrics(y_true, y_pred, sensitive, "test_feature")
    assert result["alert"] is True
    assert abs(result["difference"]["false_negative_rate"]) > DIFFERENCE_THRESHOLD


def test_print_report_runs_without_error(synthetic_data, capsys):
    y_true, y_pred, sensitive_df = synthetic_data
    report = compute_fairness_report(y_true, y_pred, sensitive_df)
    print_report(report)
    captured = capsys.readouterr()
    assert "GENDER" in captured.out
    assert "mf.difference()" in captured.out
