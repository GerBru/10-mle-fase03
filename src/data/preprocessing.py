import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.transformers import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    BinaryEncoder,
    OutlierClipper,
    TotalChargesImputer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
TARGET_COL = "churn"

# Mapeamento dos nomes originais do CSV para snake_case interno.
# Suporta dois formatos:
#   - Kaggle: WA_Fn-UseC_-Telco-Customer-Churn.csv (colunas camelCase/PascalCase)
#   - IBM extended: Telco_customer_churn.csv (colunas com espaços, título)
COLUMN_MAP = {
    # ── Kaggle ────────────────────────────────────────────────────────────────
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
    # ── IBM extended ──────────────────────────────────────────────────────────
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Tenure Months": "tenure",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Value": "churn",
}

# Nomes snake_case (pipeline interno)
NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges"]
BINARY_COLS = ["senior_citizen"]
CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]

# Colunas geradas por src/features/engineering.py
ENGINEERED_NUMERIC_COLS = ["charges_per_tenure", "num_services"]
ENGINEERED_BINARY_COLS = [
    "is_new_customer", "is_long_term", "is_monthly_contract", "is_electronic_check",
    "has_phone_service", "has_multiple_lines", "has_online_security", "has_online_backup",
    "has_device_protection", "has_tech_support", "has_streaming_tv", "has_streaming_movies",
]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> Pipeline:
    """Pipeline para dados no formato CSV original (IBM/Kaggle, colunas PascalCase).

    Use build_full_pipeline() para dados já normalizados por clean_data() (snake_case).
    """
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper()),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    column_transformer = ColumnTransformer([
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("total_charges", TotalChargesImputer()),
        ("binary", BinaryEncoder()),
        ("features", column_transformer),
    ])


def _build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer interno para features snake_case + engineered."""
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS + ENGINEERED_NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS + ENGINEERED_BINARY_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_full_pipeline() -> Pipeline:
    """Pipeline sklearn completo: FeatureEngineer → ColumnTransformer (snake_case).

    Para dados já normalizados por clean_data() (colunas snake_case).
    Use build_preprocessing_pipeline() para dados no formato CSV original (IBM/Kaggle).
    """
    from src.features.engineering import FeatureEngineerTransformer

    return Pipeline([
        ("features", FeatureEngineerTransformer()),
        ("transform", _build_preprocessor()),
    ])


# ---------------------------------------------------------------------------
# I/O e split
# ---------------------------------------------------------------------------

def load_data(path) -> pd.DataFrame:
    """Carrega CSV com COLUMN_MAP e remove colunas desnecessárias."""
    logger.info("Loading data from {}", str(path))
    df = pd.read_csv(str(path), encoding="utf-8")
    df = df.rename(columns=COLUMN_MAP)
    extra = [c for c in df.columns if c not in set(COLUMN_MAP.values())]
    if extra:
        df = df.drop(columns=extra)
        logger.debug("Dropped {} unmapped columns", len(extra))
    logger.info("Loaded {} rows, {} columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    if "senior_citizen" in df.columns and df["senior_citizen"].dtype == object:
        df["senior_citizen"] = (
            df["senior_citizen"].str.strip().str.lower() == "yes"
        ).astype(int)

    if "total_charges" in df.columns and df["total_charges"].dtype == object:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info("Imputed {} nulls in '{}' with median {:.2f}", n_missing, col, median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info("Imputed {} nulls in '{}' with mode '{}'", n_missing, col, mode_val)

    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (
            df[TARGET_COL].str.strip().str.lower() == "yes"
        ).astype(int)

    logger.info("Cleaned data: {} rows, {} columns", *df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    logger.info("Train: {} | Val: {} | Test: {}", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test
