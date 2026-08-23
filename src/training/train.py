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
import pandas as pd
import torch
import yaml

from src.models.baseline import build_baselines, refit_and_evaluate, train_baseline
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
DVC_LOCK_PATH = Path("dvc.lock")

MLFLOW_EXPERIMENT = "churn-prediction"
MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri


def _load_train_params() -> dict:
    """Carrega seed e hiperparâmetros do MLP a partir de params.yaml (rastreado pelo DVC)."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


def _dvc_dataset_hash() -> str | None:
    """Lê do dvc.lock o md5 do dataset consumido pelo estágio preprocess.

    Amarra o run do MLflow à versão exata do dado rastreada pelo DVC.
    """
    if not DVC_LOCK_PATH.exists():
        return None
    with open(DVC_LOCK_PATH) as f:
        lock = yaml.safe_load(f) or {}
    deps = lock.get("stages", {}).get("preprocess", {}).get("deps", [])
    return next((d["md5"] for d in deps if d["path"].endswith(".csv")), None)


TRAIN_PARAMS = _load_train_params()
RANDOM_STATE = TRAIN_PARAMS["random_state"]
MLP_PARAMS = TRAIN_PARAMS["mlp"]


# ── Etapas do pipeline ────────────────────────────────────────────────────────

def _load_splits() -> dict:
    """Carrega os splits processados gerados pelo estágio de preprocess."""
    splits_path = PROCESSED_DIR / "splits.joblib"
    logger.info("Loading preprocessed splits from {}", splits_path)
    return joblib.load(splits_path)


def _run_baselines(X_train_df, y_train) -> tuple:
    """Seleciona o melhor baseline usando apenas o F1 médio da CV."""
    results: dict = {}
    best_f1 = 0.0
    best_pipeline = None
    best_name = ""

    for name, bl_pipeline, params in build_baselines():
        res = train_baseline(bl_pipeline, X_train_df, y_train, name, params)
        results[name] = {"cv": res["cv_metrics"]}
        if res["cv_metrics"]["f1"] > best_f1:
            best_f1 = res["cv_metrics"]["f1"]
            best_pipeline = res["pipeline"]
            best_name = name

    if best_pipeline:
        logger.info("Best baseline: {} (CV F1={:.4f})", best_name, best_f1)
        mlflow.log_param("best_baseline", best_name)
        mlflow.log_metric("best_baseline_cv_f1", best_f1)

    return results, best_pipeline, best_name


def _evaluate_classical_champion(best_pipeline, best_name: str, splits: dict) -> dict:
    """Refita no desenvolvimento, avalia o teste e persiste o campeão clássico."""
    X_development = pd.concat(
        [splits["X_train_df"], splits["X_val_df"]], ignore_index=True
    )
    y_development = pd.concat(
        [splits["y_train"], splits["y_val"]], ignore_index=True
    )
    champion, metrics = refit_and_evaluate(
        best_pipeline,
        X_development,
        y_development,
        splits["X_test_df"],
        splits["y_test"],
    )
    joblib.dump(champion, MODELS_DIR / "best_baseline.joblib")
    logger.info("{} — final test F1={:.4f}", best_name, metrics["f1"])
    return {"pipeline": champion, "metrics": metrics}


def _write_registry_evidence(version, best_name: str) -> dict:
    """Persiste a versão promovida para auditoria e consumo pela API."""
    evidence = {
        "name": settings.registered_model_name,
        "version": str(version.version),
        "alias": settings.champion_alias,
        "run_id": version.run_id,
        "source": version.source,
        "baseline_name": best_name,
    }
    path = MODELS_DIR / "registry.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


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
        mlflow.pytorch.log_model(trainer.model, name="model")
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

def _log_run_params(splits: dict) -> None:
    """Loga metadados do run: dataset (path + hash DVC), tamanhos dos splits e seed."""
    mlflow.log_param("dataset", str(settings.data_path))
    mlflow.log_param("train_size", len(splits["X_train_df"]))
    mlflow.log_param("val_size", len(splits["X_val_df"]))
    mlflow.log_param("test_size", len(splits["X_test_df"]))
    mlflow.log_param("random_state", RANDOM_STATE)
    if dataset_hash := _dvc_dataset_hash():
        mlflow.set_tag("dvc_dataset_md5", dataset_hash)


def _promote_champion(champion: dict, best_name: str, X_test_df: pd.DataFrame) -> None:
    """Promove o campeão clássico no Model Registry e persiste a evidência."""
    logger.info("Promoting best baseline to Model Registry...")
    version = log_and_register_champion(
        champion["pipeline"],
        X_test_df,
        baseline_name=best_name,
        metrics=champion["metrics"],
        model_name=settings.registered_model_name,
        alias=settings.champion_alias,
        required=settings.mlflow_registry_required,
    )
    if version is None:
        raise RuntimeError("Champion was not promoted to the MLflow Registry")
    _write_registry_evidence(version, best_name)


def _run_experiment(splits: dict, results: dict) -> None:
    """Executa baselines, promove o campeão e treina o MLP dentro do run principal."""
    X_train_df, X_test_df = splits["X_train_df"], splits["X_test_df"]
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    _log_run_params(splits)

    logger.info("Training baselines...")
    baseline_results, best_pipeline, best_name = _run_baselines(X_train_df, y_train)
    results.update(baseline_results)

    champion = _evaluate_classical_champion(best_pipeline, best_name, splits)
    results[best_name]["test"] = champion["metrics"]
    _promote_champion(champion, best_name, X_test_df)

    logger.info("Training MLP PyTorch...")
    mlp_result = _train_mlp_experiment(
        X_train, y_train.values,
        X_val, y_val.values,
        X_test, y_test.values,
        input_dim=X_train.shape[1],
    )
    results["mlp_pytorch"] = {"test": mlp_result["metrics"]}
    _save_artifacts(X_train, mlp_result)

    mlflow.log_metric(
        "mlp_vs_best_baseline_f1_delta",
        mlp_result["metrics"]["f1"] - champion["metrics"]["f1"],
    )


def _log_results_summary(results: dict) -> None:
    """Imprime um resumo tabular das métricas de cada modelo treinado."""
    logger.info("\nResults summary:")
    for name, metric_groups in results.items():
        metrics = metric_groups.get("test", metric_groups.get("cv", {}))
        logger.info(
            "  {:<28} F1={:.4f}  AUC={:.4f}  Precision={:.4f}  Recall={:.4f}",
            name, metrics["f1"], metrics.get("auc_roc", 0),
            metrics["precision"], metrics["recall"],
        )


def _persist_results(results: dict) -> None:
    """Salva as métricas de todos os modelos em models/results.json."""
    rounded = {
        model: {
            group: {metric: round(value, 4) for metric, value in values.items()}
            for group, values in groups.items()
        }
        for model, groups in results.items()
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(rounded, f, indent=2)
    logger.info("Results saved to models/results.json")


def main():
    """Orquestra o pipeline completo de treino."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    splits = _load_splits()
    results: dict = {}

    with mlflow.start_run(run_name="churn_experiment"):
        _run_experiment(splits, results)

    _log_results_summary(results)
    _persist_results(results)


if __name__ == "__main__":
    main()
