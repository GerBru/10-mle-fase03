import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import build_full_pipeline, clean_data, split_data


@pytest.fixture
def sample_df():
    n = 20
    genders = (["Male", "Female"] * 10)[:n]
    senior = (["No", "Yes", "No", "No"] * 5)[:n]
    partners = (["Yes", "No"] * 10)[:n]
    dependents = (["No", "Yes"] * 10)[:n]
    tenures = list(range(1, n + 1))
    phone = (["Yes", "No"] * 10)[:n]
    multi = (["No", "Yes", "No phone service"] * 7)[:n]
    internet = (["DSL", "Fiber optic", "No"] * 7)[:n]
    security = (["No", "Yes", "No internet service"] * 7)[:n]
    backup = (["Yes", "No", "No internet service"] * 7)[:n]
    device = (["No", "Yes", "No internet service"] * 7)[:n]
    tech = (["No", "Yes", "No internet service"] * 7)[:n]
    tv = (["No", "Yes", "No internet service"] * 7)[:n]
    movies = (["No", "Yes", "No internet service"] * 7)[:n]
    contracts = (["Month-to-month", "One year", "Two year"] * 7)[:n]
    billing = (["Yes", "No"] * 10)[:n]
    payment = (["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 5)[:n]
    monthly = [round(20 + i * 2.5, 2) for i in range(n)]
    total = [str(round(m * t, 2)) for m, t in zip(monthly, tenures, strict=False)]
    # 12 No, 8 Yes — enough for stratified splits with test_size=0.2, val_size=0.1
    churn = (["No"] * 3 + ["Yes"] * 2) * 4
    return pd.DataFrame({
        "customer_id": [f"A{i:03d}" for i in range(1, n + 1)],
        "gender": genders,
        "senior_citizen": senior,
        "partner": partners,
        "dependents": dependents,
        "tenure": tenures,
        "phone_service": phone,
        "multiple_lines": multi,
        "internet_service": internet,
        "online_security": security,
        "online_backup": backup,
        "device_protection": device,
        "tech_support": tech,
        "streaming_tv": tv,
        "streaming_movies": movies,
        "contract": contracts,
        "paperless_billing": billing,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "churn": churn,
    })


def test_clean_removes_customer_id(sample_df):
    df = clean_data(sample_df)
    assert "customer_id" not in df.columns


def test_clean_total_charges_numeric(sample_df):
    df = clean_data(sample_df)
    assert df["total_charges"].dtype in [float, np.float64]


def test_clean_target_binary(sample_df):
    df = clean_data(sample_df)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_clean_senior_citizen_binary(sample_df):
    df = clean_data(sample_df)
    assert set(df["senior_citizen"].unique()).issubset({0, 1})


def test_clean_no_nulls_after_imputation(sample_df):
    sample_df.loc[0, "monthly_charges"] = np.nan
    sample_df.loc[1, "total_charges"] = " "
    df = clean_data(sample_df)
    assert df.isnull().sum().sum() == 0


def test_preprocessor_output_2d(sample_df):
    df = clean_data(sample_df)
    X = df.drop(columns=["churn"])
    pipeline = build_full_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.ndim == 2
    assert X_transformed.shape[0] == len(df)
    assert X_transformed.shape[1] > 0


def test_split_stratified_sizes(sample_df):
    df = clean_data(sample_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_size=0.2, val_size=0.1)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df)
