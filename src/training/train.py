"""
Pipeline de treinamento — FIAP Tech Challenge Fase 1.

Executa o ciclo completo de treino:
    1. Carrega e valida o dataset com Pandera
    2. Pré-processa e divide em treino/val/teste
    3. Treina baselines (DummyClassifier, LogReg, RF, GBT) com cross-validation
    4. Treina MLP PyTorch com early stopping
    5. Loga todos os experimentos no MLflow
    6. Salva artefatos em models/

Uso:
    uv run python -m src.training.train
    # ou
    make train
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch

from src.data.preprocessing import (
    build_full_pipeline,
    clean_data,
    load_data,
    split_data,
)
from src.data.schema import validate_raw
from src.models.baseline import build_baselines, train_baseline
from src.models.evaluation import compute_metrics
from src.models.mlp import MLPTrainer
from src.utils import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = settings.data_path
MLFLOW_EXPERIMENT = "churn-prediction"
MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri

MLP_PARAMS = {
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.3,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,
    "use_batch_norm": True,
}


# ── Etapas do pipeline ────────────────────────────────────────────────────────

def _load_and_validate(data_path):
    """Carrega CSV e valida schema com Pandera."""
    logger.info("Loading and validating data from {}", data_path)
    df_raw = load_data(data_path)
    validate_raw(df_raw)
    return df_raw


def _build_preprocessed_splits(df) -> tuple:
    """Limpa, divide e aplica o pipeline completo. Retorna (pipeline, arrays, labels)."""
    df = clean_data(df)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(df)

    pipeline = build_full_pipeline()
    X_train = pipeline.fit_transform(X_train_df)
    X_val = pipeline.transform(X_val_df)
    X_test = pipeline.transform(X_test_df)

    joblib.dump(pipeline, MODELS_DIR / "preprocessor.joblib")
    logger.info("Full pipeline fitted. Feature dim: {}", X_train.shape[1])
    return pipeline, X_train_df, X_val_df, X_test_df, X_train, X_val, X_test, y_train, y_val, y_test


def _run_baselines(X_train_df, y_train, X_test_df, y_test) -> tuple:
    """Treina baselines com CV e retorna (results_dict, best_pipeline, best_name)."""
    results: dict = {}
    best_f1 = 0.0
    best_pipeline = None
    best_name = ""

    for name, bl_pipeline, params in build_baselines():
        res = train_baseline(bl_pipeline, X_train_df, y_train, X_test_df, y_test, name, params)
        results[name] = res["metrics"]
        if res["metrics"]["f1"] > best_f1:
            best_f1 = res["metrics"]["f1"]
            best_pipeline = res["pipeline"]
            best_name = name

    if best_pipeline:
        joblib.dump(best_pipeline, MODELS_DIR / "best_baseline.joblib")
        logger.info("Best baseline: {} (F1={:.4f})", best_name, best_f1)
        mlflow.log_param("best_baseline", best_name)
        mlflow.log_metric("best_baseline_f1", best_f1)

    return results, best_pipeline, best_name


def _train_mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, input_dim: int) -> dict:
    """Treina o MLP PyTorch e loga métricas, histórico e artefatos no MLflow."""
    with mlflow.start_run(run_name="mlp_pytorch", nested=True):
        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("framework", "pytorch")
        mlflow.log_params(MLP_PARAMS)

        trainer = MLPTrainer(input_dim=input_dim, **MLP_PARAMS, random_state=RANDOM_STATE)
        trainer.fit(X_train, y_train, X_val, y_val)

        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        for name, val in metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        mlflow.log_dict(
            {"train_loss": trainer.history["train_loss"], "val_loss": trainer.history["val_loss"]},
            "training_history.json",
        )
        mlflow.pytorch.log_model(trainer.model, "model")
        logger.info("MLP — Test F1: {:.4f} | AUC: {:.4f}", metrics["f1"], metrics.get("auc_roc", 0))

    return {"trainer": trainer, "metrics": metrics}


def _save_artifacts(X_train, mlp_result) -> None:
    """Persiste mlp_model.pt, model_config.json e results.json em MODELS_DIR."""
    input_dim = X_train.shape[1]
    torch.save(
        {
            "input_dim": input_dim,
            "hidden_dims": MLP_PARAMS["hidden_dims"],
            "state_dict": mlp_result["trainer"].model.state_dict(),
        },
        MODELS_DIR / "mlp_model.pt",
    )
    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump({"input_dim": input_dim, "hidden_dims": MLP_PARAMS["hidden_dims"]}, f)
    logger.info("Saved mlp_model.pt and model_config.json (input_dim={})", input_dim)


# ── Orquestrador ──────────────────────────────────────────────────────────────

def main():
    """Orquestra o pipeline completo de treino."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df_raw = _load_and_validate(DATA_PATH)
    (
        pipeline, X_train_df, X_val_df, X_test_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    ) = _build_preprocessed_splits(df_raw)

    results: dict = {}

    with mlflow.start_run(run_name="churn_experiment"):
        mlflow.log_param("dataset", DATA_PATH)
        mlflow.log_param("train_size", len(X_train_df))
        mlflow.log_param("val_size", len(X_val_df))
        mlflow.log_param("test_size", len(X_test_df))
        mlflow.log_param("random_state", RANDOM_STATE)

        logger.info("Training baselines...")
        baseline_results, _, _ = _run_baselines(X_train_df, y_train, X_test_df, y_test)
        results.update(baseline_results)

        logger.info("Training MLP PyTorch...")
        mlp_result = _train_mlp_experiment(
            X_train, y_train.values,
            X_val, y_val.values,
            X_test, y_test.values,
            input_dim=X_train.shape[1],
        )
        results["mlp_pytorch"] = mlp_result["metrics"]

        _save_artifacts(X_train, mlp_result)

        mlp_f1 = mlp_result["metrics"]["f1"]
        best_baseline_f1 = max(
            (m["f1"] for m in baseline_results.values()), default=0.0
        )
        mlflow.log_metric("mlp_vs_best_baseline_f1_delta", mlp_f1 - best_baseline_f1)

    logger.info("\nResults summary:")
    for name, metrics in results.items():
        logger.info(
            "  {:<28} F1={:.4f}  AUC={:.4f}  Precision={:.4f}  Recall={:.4f}",
            name, metrics["f1"], metrics.get("auc_roc", 0),
            metrics["precision"], metrics["recall"],
        )

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(
            {k: {m: round(v, 4) for m, v in v.items()} for k, v in results.items()},
            f, indent=2,
        )
    logger.info("Results saved to models/results.json")


if __name__ == "__main__":
    main()
