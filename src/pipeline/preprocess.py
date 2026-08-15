"""
Estágio de pré-processamento do pipeline DVC.

Carrega o dataset bruto, valida o schema, limpa, divide em treino/val/teste
e ajusta o pipeline de features. Persiste os splits processados e o
preprocessor ajustado para o estágio de treino (src/training/train.py) consumir.

Uso:
    uv run python -m src.pipeline.preprocess
    # ou
    dvc repro preprocess
"""
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocessing import (
    build_full_pipeline,
    clean_data,
    load_data,
    split_data,
)
from src.data.schema import validate_raw
from src.utils import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_PATH = settings.data_path
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_and_validate(data_path: Path) -> pd.DataFrame:
    """Carrega CSV e valida schema com Pandera."""
    logger.info("Loading and validating data from {}", data_path)
    df_raw = load_data(data_path)
    validate_raw(df_raw)
    return df_raw


def build_splits(df: pd.DataFrame) -> dict:
    """Limpa, divide e ajusta o pipeline de features. Retorna todos os artefatos dos splits."""
    df = clean_data(df)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(df)

    pipeline = build_full_pipeline()
    X_train = pipeline.fit_transform(X_train_df)
    X_val = pipeline.transform(X_val_df)
    X_test = pipeline.transform(X_test_df)

    logger.info("Full pipeline fitted. Feature dim: {}", X_train.shape[1])
    return {
        "pipeline": pipeline,
        "X_train_df": X_train_df,
        "X_val_df": X_val_df,
        "X_test_df": X_test_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def save_artifacts(splits: dict) -> None:
    """Persiste o preprocessor ajustado e os splits processados em disco."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(splits["pipeline"], MODELS_DIR / "preprocessor.joblib")
    joblib.dump(
        {k: v for k, v in splits.items() if k != "pipeline"},
        PROCESSED_DIR / "splits.joblib",
    )
    logger.info("Saved preprocessor.joblib and data/processed/splits.joblib")


def main() -> None:
    """Orquestra o estágio de pré-processamento: carga → validação → limpeza → split → features."""
    df_raw = load_and_validate(DATA_PATH)
    splits = build_splits(df_raw)
    save_artifacts(splits)


if __name__ == "__main__":
    main()
