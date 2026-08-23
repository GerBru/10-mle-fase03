"""Testes do módulo de detecção de drift (KS test + PSI).

O módulo é citado no `docs/monitoring_plan.md` e sustenta o padrão *Continued
Model Evaluation* documentado no README, mas era a maior superfície sem cobertura
em `src/`. Os cenários abaixo fixam a semente para manter os testes determinísticos.
"""

import numpy as np

from src.monitoring.drift_detection import (
    PSI_CRITICAL,
    analyze_drift,
    ks_test,
    load_reference_stats,
    psi,
    save_reference_stats,
)

RNG = np.random.default_rng(42)


def _sample(loc: float, scale: float, size: int) -> np.ndarray:
    return np.random.default_rng(7).normal(loc=loc, scale=scale, size=size)


def test_ks_test_reports_no_drift_for_same_distribution():
    reference = _sample(50, 10, 500)

    result = ks_test(reference, reference.copy(), "tenure")

    assert result["feature"] == "tenure"
    assert not result["drift"]
    assert result["p_value"] > 0.05


def test_ks_test_detects_shifted_distribution():
    reference = np.random.default_rng(1).normal(50, 5, 500)
    production = np.random.default_rng(2).normal(90, 5, 500)

    result = ks_test(reference, production, "monthly_charges")

    assert result["drift"]
    assert result["statistic"] > 0.5


def test_psi_is_near_zero_for_identical_distributions():
    reference = _sample(30, 8, 1000)

    assert psi(reference, reference.copy()) < 0.01


def test_psi_grows_with_distribution_shift():
    reference = np.random.default_rng(3).normal(30, 8, 1000)
    shifted = np.random.default_rng(4).normal(70, 8, 1000)

    assert psi(reference, shifted) >= PSI_CRITICAL


def test_analyze_drift_flags_only_shifted_features():
    stable = np.random.default_rng(5).normal(20, 4, 400)
    reference = {"tenure": stable, "charges": np.random.default_rng(6).normal(60, 6, 400)}
    production = {
        "tenure": np.random.default_rng(5).normal(20, 4, 400),
        "charges": np.random.default_rng(8).normal(120, 6, 400),
    }

    results = analyze_drift(reference, production)

    assert not results["tenure"]["alert"]
    assert results["charges"]["alert"]
    assert set(results) == {"tenure", "charges"}


def test_analyze_drift_ignores_features_absent_from_production():
    reference = {"tenure": _sample(20, 4, 200), "charges": _sample(60, 6, 200)}

    results = analyze_drift(reference, {"tenure": _sample(20, 4, 200)})

    assert set(results) == {"tenure"}


def test_reference_stats_round_trip(tmp_path):
    features = ["tenure", "charges"]
    matrix = np.column_stack([_sample(20, 4, 50), _sample(60, 6, 50)])
    path = tmp_path / "reference_stats.npz"

    save_reference_stats(matrix, features, str(path))
    loaded = load_reference_stats(str(path))

    assert set(loaded) == set(features)
    np.testing.assert_allclose(loaded["tenure"], matrix[:, 0])
