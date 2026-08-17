"""
Pipeline de treinamento — FIAP Tech Challenge Fase 2.

Consome os splits já processados pelo estágio de preprocess (src/pipeline/preprocess.py)
e executa o ciclo de treino:
    1. Carrega os splits processados (data/processed/splits.joblib)
    2. Treina baselines (DummyClassifier, LogReg, RF, GBT) com cross-validation
    3. Promove o melhor baseline sklearn no MLflow Model Registry
    4. Treina MLP PyTorch com early stopping
    5. Loga todos os experimentos no MLflow
    6. Salva artefatos em models/

Uso:
    dvc repro              # roda preprocess → train
    # ou, isoladamente (requer data/processed/splits.joblib já gerado):
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
import yaml

from src.models.baseline import build_baselines, train_baseline
from src.models.evaluation import compute_metrics
from src.models.mlp import MLPTrainer
from src.models.registry import log_and_register_champion
from src.utils import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")

MLFLOW_EXPERIMENT = "churn-prediction"
MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri


def _load_train_params() -> dict:
    """Carrega seed e hiperparâmetros do MLP a partir de params.yaml (rastreado pelo DVC)."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


TRAIN_PARAMS = _load_train_params()
RANDOM_STATE = TRAIN_PARAMS["random_state"]
MLP_PARAMS = TRAIN_PARAMS["mlp"]


# ── Etapas do pipeline ────────────────────────────────────────────────────────

def _load_splits() -> dict:
    """Carrega os splits processados gerados pelo estágio de preprocess."""
    splits_path = PROCESSED_DIR / "splits.joblib"
    logger.info("Loading preprocessed splits from {}", splits_path)
    return joblib.load(splits_path)


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

    splits = _load_splits()
    X_train_df, X_val_df, X_test_df = splits["X_train_df"], splits["X_val_df"], splits["X_test_df"]
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    results: dict = {}

    with mlflow.start_run(run_name="churn_experiment"):
        mlflow.log_param("dataset", str(settings.data_path))
        mlflow.log_param("train_size", len(X_train_df))
        mlflow.log_param("val_size", len(X_val_df))
        mlflow.log_param("test_size", len(X_test_df))
        mlflow.log_param("random_state", RANDOM_STATE)

        logger.info("Training baselines...")
        baseline_results, best_pipeline, best_name = _run_baselines(
            X_train_df, y_train, X_test_df, y_test
        )
        results.update(baseline_results)

        logger.info("Promoting best baseline to Model Registry...")
        log_and_register_champion(
            best_pipeline,
            X_test_df,
            baseline_name=best_name,
            metrics=baseline_results.get(best_name),
        )

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
