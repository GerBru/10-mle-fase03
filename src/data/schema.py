import logging

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

RAW_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "senior_citizen": Column(
            checks=Check(
                lambda s: s.astype(str).isin(["0", "1", "Yes", "No"]).all(),
                error="senior_citizen must be 0/1 (Kaggle) or Yes/No (IBM)",
            ),
            nullable=False,
        ),
        "partner": Column(str, Check.isin(["Yes", "No"])),
        "dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.ge(0)),
        "phone_service": Column(str, Check.isin(["Yes", "No"])),
        "multiple_lines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        "internet_service": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "online_security": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "online_backup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "device_protection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "tech_support": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "streaming_tv": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "streaming_movies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "paperless_billing": Column(str, Check.isin(["Yes", "No"])),
        "payment_method": Column(
            str,
            Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
        ),
        "monthly_charges": Column(float, Check.gt(0), coerce=True),
    },
    coerce=True,
    strict=False,  # permite colunas extras (customer_id, total_charges, churn)
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed (%d rows)", len(validated))
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error("Schema validation failed:\n%s", exc.failure_cases)
        raise
