from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClienteInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "senior_citizen": 0,
                "tenure": 12,
                "monthly_charges": 65.5,
                "total_charges": 786.0,
                "gender": "Male",
                "partner": "Yes",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
            }
        },
    )

    senior_citizen: int = Field(..., ge=0, le=1, description="0=Nao, 1=Sim")
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    monthly_charges: float = Field(..., gt=0, description="Cobranca mensal em USD")
    total_charges: float = Field(..., ge=0, description="Cobranca total acumulada")

    gender: Literal["Male", "Female"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade de churn (0 a 1)")
    prediction: int = Field(..., description="0=Nao churn, 1=Churn")
    risk_level: Literal["low", "medium", "high"]


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    count: int


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_source: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
