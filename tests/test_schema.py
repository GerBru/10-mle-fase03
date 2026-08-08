"""Validação de schema do dataset com pandera."""

import os

import pandera.pandas as pa
import pytest

from src.data.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    clean_data,
    load_data,
)

DATA_PATH = os.getenv("DATA_PATH", "data/raw/Telco_customer_churn.csv")

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "tenure": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        "monthly_charges": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
    }
)


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente — coloque Telco_customer_churn.csv em data/raw/",
)
def test_data_loads():
    df = load_data(DATA_PATH)
    assert len(df) >= 5000
    assert "churn" in df.columns


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente",
)
def test_churn_binary_after_clean():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    assert set(df["churn"].unique()).issubset({0, 1}), f"Unexpected values: {df['churn'].unique()}"


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente",
)
def test_required_columns_present():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente",
)
def test_no_negative_tenure():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    assert (df["tenure"] >= 0).all()


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente",
)
def test_pandera_schema():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    RAW_SCHEMA.validate(df)


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset ausente",
)
def test_no_nulls_after_clean():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} nulls after clean_data()"
