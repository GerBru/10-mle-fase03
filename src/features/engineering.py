import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

_SERVICE_COLS = [
    "phone_service", "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas relevantes para churn em telecom."""
    df = df.copy()
    eps = 1e-6

    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + eps)
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)

    for col in _SERVICE_COLS:
        if col in df.columns:
            df[f"has_{col}"] = (df[col].str.lower() == "yes").astype(int)

    has_cols = [c for c in df.columns if c.startswith("has_")]
    if has_cols:
        df["num_services"] = df[has_cols].sum(axis=1)

    if "contract" in df.columns:
        df["is_monthly_contract"] = (df["contract"] == "Month-to-month").astype(int)

    if "payment_method" in df.columns:
        df["is_electronic_check"] = (
            df["payment_method"] == "Electronic check"
        ).astype(int)

    logger.info("Feature engineering: %d columns total", len(df.columns))
    return df


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compatível para add_features() — plugável em Pipeline."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineerTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_features(X)
