"""Transformers sklearn customizados para o pipeline de preprocessing de churn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Nomes originais do CSV (IBM/Kaggle) — usados por build_preprocessing_pipeline
NUMERICAL_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
BINARY_FEATURES = ["Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]
CATEGORICAL_FEATURES = [
    "Gender", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Payment Method",
]

# Colunas a descartar (leakage pós-churn, geográficas, identificadores)
_COLS_TO_DROP = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Churn Label", "Churn Score",
    "CLTV", "Churn Reason",
]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Limita outliers numéricos por percentil (fit no treino, apply no restante)."""

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> None:
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X, y=None):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class TotalChargesImputer(BaseEstimator, TransformerMixin):
    """Converte Total Charges para numérico e imputa brancos com mediana de treino."""

    def fit(self, X, y=None):
        col = "Total Charges" if "Total Charges" in X.columns else "total_charges"
        self._col = col
        self.median_ = pd.to_numeric(X[col], errors="coerce").median()
        return self

    def transform(self, X):
        X = X.copy()
        X[self._col] = pd.to_numeric(X[self._col], errors="coerce").fillna(self.median_)
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Mapeia Yes/No (e Male/Female para Gender) para 1/0."""

    _MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = self.columns or BINARY_FEATURES
        for col in cols:
            if col not in X.columns:
                continue
            if X[col].dtype == object:
                X[col] = X[col].map(self._MAP).fillna(0).astype(int)
            else:
                X[col] = X[col].fillna(0).astype(int)
        return X
