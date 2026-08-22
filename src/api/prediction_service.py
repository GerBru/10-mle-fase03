"""
Serviço de predição de churn: encapsula preprocessing + inferência + classificação de risco.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.api.schemas import ClienteInput, PredictionOutput
from src.models.mlp import predict_proba


@dataclass
class RiskClassifier:
    """Classifica probabilidade de churn em categoria de risco com thresholds configuráveis."""

    low_threshold: float = 0.4
    high_threshold: float = 0.7

    def classify(self, prob: float) -> str:
        if prob >= self.high_threshold:
            return "high"
        if prob >= self.low_threshold:
            return "medium"
        return "low"


class PredictionService:
    """Encapsula o fluxo completo de predição: preprocessing → inferência → risco."""

    def __init__(self, pipeline, model, risk_classifier: RiskClassifier | None = None) -> None:
        self.pipeline = pipeline
        self.model = model
        self.risk_classifier = risk_classifier or RiskClassifier()

    def predict(self, cliente: ClienteInput) -> tuple[float, int, str]:
        """Retorna (churn_probability, prediction, risk_level) para um cliente."""
        probs = self._predict_probabilities(_to_dataframe(cliente))
        prob = float(probs[0])
        prediction = int(prob >= 0.5)
        risk = self.risk_classifier.classify(prob)
        return prob, prediction, risk

    def predict_batch(self, clientes: list[ClienteInput]) -> list[PredictionOutput]:
        """Retorna lista de PredictionOutput para múltiplos clientes."""
        df = pd.concat([_to_dataframe(c) for c in clientes], ignore_index=True)
        probs = self._predict_probabilities(df)
        return [
            PredictionOutput(
                churn_probability=round(float(p), 4),
                prediction=int(float(p) >= 0.5),
                risk_level=self.risk_classifier.classify(float(p)),
            )
            for p in probs
        ]

    def _predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Executa MLP legada ou pipeline sklearn champion pela mesma interface."""
        if self.pipeline is None:
            return self.model.predict_proba(frame)[:, 1]
        transformed = self.pipeline.transform(frame).astype(np.float32)
        return predict_proba(self.model, transformed)


def _to_dataframe(cliente: ClienteInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "tenure": cliente.tenure,
        "monthly_charges": cliente.monthly_charges,
        "total_charges": cliente.total_charges,
        "senior_citizen": cliente.senior_citizen,
        "gender": cliente.gender,
        "partner": cliente.partner,
        "dependents": cliente.dependents,
        "phone_service": cliente.phone_service,
        "multiple_lines": cliente.multiple_lines,
        "internet_service": cliente.internet_service,
        "online_security": cliente.online_security,
        "online_backup": cliente.online_backup,
        "device_protection": cliente.device_protection,
        "tech_support": cliente.tech_support,
        "streaming_tv": cliente.streaming_tv,
        "streaming_movies": cliente.streaming_movies,
        "contract": cliente.contract,
        "paperless_billing": cliente.paperless_billing,
        "payment_method": cliente.payment_method,
    }])
