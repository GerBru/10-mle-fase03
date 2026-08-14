"""Drift detection — detecta data drift entre distribuições de referência e produção.

Implementa KS test e PSI para monitoramento pós-deploy.
Uso direto: python -m src.monitoring.drift_detection
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA = 0.05
PSI_MODERATE = 0.1
PSI_CRITICAL = 0.2

NUMERICAL_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]


def ks_test(reference: np.ndarray, production: np.ndarray, feature_name: str) -> dict:
    """Teste Kolmogorov-Smirnov entre distribuição de referência e produção."""
    statistic, p_value = stats.ks_2samp(reference, production)
    has_drift = p_value < ALPHA
    logger.info(
        "KS [%s]: stat=%.4f p=%.4f drift=%s",
        feature_name,
        statistic,
        p_value,
        "SIM" if has_drift else "NAO",
    )
    return {"feature": feature_name, "statistic": float(statistic), "p_value": float(p_value), "drift": has_drift}


def psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index — mede estabilidade da distribuição.

    PSI < 0.1: estável | 0.1-0.2: monitorar | >0.2: crítico
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference)
    prod_pct = np.histogram(production, bins=bins)[0] / len(production)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

    level = "ESTAVEL" if psi_value < PSI_MODERATE else "MODERADO" if psi_value < PSI_CRITICAL else "CRITICO"
    logger.info("PSI: %.4f → %s", psi_value, level)
    return psi_value


def analyze_drift(
    reference: dict[str, np.ndarray],
    production: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Executa KS test + PSI para cada feature numérica.

    Args:
        reference: dict {feature_name: array} com dados de treino.
        production: dict {feature_name: array} com dados recentes de produção.

    Returns:
        dict com resultados por feature: {ks: ..., psi: ..., alert: bool}
    """
    results = {}
    for feature in reference:
        if feature not in production:
            continue
        ref_arr = np.asarray(reference[feature], dtype=float)
        prod_arr = np.asarray(production[feature], dtype=float)

        ks_result = ks_test(ref_arr, prod_arr, feature)
        psi_value = psi(ref_arr, prod_arr)

        alert = ks_result["drift"] or psi_value >= PSI_CRITICAL
        results[feature] = {
            "ks": ks_result,
            "psi": psi_value,
            "alert": alert,
        }
        if alert:
            logger.warning("ALERTA DE DRIFT em '%s' — PSI=%.4f drift_ks=%s", feature, psi_value, ks_result["drift"])

    return results


def save_reference_stats(X: np.ndarray, feature_names: list[str], path: str) -> None:
    """Salva estatísticas de referência (treino) para comparação futura."""
    stats_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
    np.savez(path, **stats_dict)
    logger.info("Estatísticas de referência salvas em %s", path)


def load_reference_stats(path: str) -> dict[str, np.ndarray]:
    """Carrega estatísticas de referência salvas."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(42)

    reference_data = {
        "Tenure Months": rng.normal(loc=32, scale=24, size=1000),
        "Monthly Charges": rng.normal(loc=65, scale=30, size=1000),
        "Total Charges": rng.normal(loc=2280, scale=2270, size=1000),
    }

    scenarios = {
        "sem drift": {k: rng.normal(loc=np.mean(v), scale=np.std(v), size=300) for k, v in reference_data.items()},
        "drift moderado": {
            "Tenure Months": rng.normal(loc=40, scale=24, size=300),
            "Monthly Charges": rng.normal(loc=75, scale=30, size=300),
            "Total Charges": rng.normal(loc=3000, scale=2270, size=300),
        },
        "drift critico": {
            "Tenure Months": rng.normal(loc=55, scale=30, size=300),
            "Monthly Charges": rng.normal(loc=95, scale=40, size=300),
            "Total Charges": rng.exponential(scale=2000, size=300),
        },
    }

    for scenario_name, prod_data in scenarios.items():
        print(f"\n{'='*50}\nCenário: {scenario_name.upper()}\n{'='*50}")
        analyze_drift(reference_data, prod_data)
