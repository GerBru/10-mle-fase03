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
from sklearn.exceptions import NotFittedError

from src.data.preprocessing import (
    build_full_pipeline,
    clean_data,
    load_data,
    split_data,
)
from src.data.schema import validate_raw
from src.monitoring.drift_detection import save_reference_stats
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


def _feature_names(pipeline, n_features: int) -> list[str]:
    """Nomes das colunas após a transformação, com degradação em três níveis.

    `Pipeline.get_feature_names_out()` exige que todos os passos implementem o
    método, e `FeatureEngineerTransformer` não implementa. O `ColumnTransformer`
    final implementa e é quem define o layout real da matriz de saída, então ele
    é a fonte correta. O fallback posicional mantém o artefato utilizável caso a
    topologia do pipeline mude.
    """
    for fonte in (pipeline, pipeline[-1] if hasattr(pipeline, "__getitem__") else None):
        if fonte is None:
            continue
        try:
            return [str(name) for name in fonte.get_feature_names_out()]
        except (AttributeError, ValueError, NotFittedError):
            continue

    logger.warning("Nomes de features indisponíveis — usando nomes posicionais.")
    return [f"feature_{i}" for i in range(n_features)]


def save_reference_stats_for_drift(splits: dict) -> None:
    """Persiste a distribuição do treino como referência para detecção de drift.

    Sem este artefato, `src/monitoring/drift_detection.py` não tem contra o quê
    comparar os dados de produção — as funções `ks_test` e `psi` exigem uma
    amostra de referência. É o que torna o plano descrito em
    `docs/monitoring_plan.md` executável.
    """
    X_train = splits["X_train"]
    feature_names = _feature_names(splits["pipeline"], X_train.shape[1])
    save_reference_stats(X_train, feature_names, str(MODELS_DIR / "reference_stats.npz"))


def save_artifacts(splits: dict) -> None:
    """Persiste o preprocessor ajustado e os splits processados em disco."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(splits["pipeline"], MODELS_DIR / "preprocessor.joblib")
    joblib.dump(
        {k: v for k, v in splits.items() if k != "pipeline"},
        PROCESSED_DIR / "splits.joblib",
    )
    save_reference_stats_for_drift(splits)
    logger.info("Saved preprocessor.joblib and data/processed/splits.joblib")


def main() -> None:
    """Orquestra o estágio de pré-processamento: carga → validação → limpeza → split → features."""
    df_raw = load_and_validate(DATA_PATH)
    splits = build_splits(df_raw)
    save_artifacts(splits)


if __name__ == "__main__":
    main()
