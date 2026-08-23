"""Regression tests for the Phase 2 selection, evidence and serving flow."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from src.api.model_loader import ChampionModelRepository, LocalModelRepository
from src.api.prediction_service import PredictionService
from src.api.schemas import ClienteInput
from src.models.baseline import refit_and_evaluate
from src.models.mlp import ChurnMLP
from src.training import train


def _customer() -> ClienteInput:
    return ClienteInput(
        senior_citizen=0,
        tenure=12,
        monthly_charges=65.5,
        total_charges=786.0,
        gender="Male",
        partner="Yes",
        dependents="No",
        phone_service="Yes",
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="No",
        online_backup="Yes",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
    )


class _RawChampion:
    def predict_proba(self, frame):
        return np.tile([0.25, 0.75], (len(frame), 1))


def test_champion_repository_and_prediction_service(tmp_path):
    joblib.dump(_RawChampion(), tmp_path / "best_baseline.joblib")
    evidence = {
        "name": "churn-classifier",
        "version": "3",
        "alias": "champion",
    }
    (tmp_path / "registry.json").write_text(json.dumps(evidence), encoding="utf-8")

    loaded = ChampionModelRepository(tmp_path).load()
    result = PredictionService(loaded["pipeline"], loaded["model"]).predict(_customer())

    assert loaded["model_metadata"] == evidence
    assert result == (0.75, 1, "high")


def test_champion_repository_requires_both_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Champion artifacts missing"):
        ChampionModelRepository(tmp_path).load()


def test_local_mlp_repository_loads_checkpoint(tmp_path):
    model = ChurnMLP(input_dim=3, hidden_dims=[4])
    torch.save(
        {"input_dim": 3, "hidden_dims": [4], "state_dict": model.state_dict()},
        tmp_path / "mlp_model.pt",
    )
    (tmp_path / "model_config.json").write_text(
        json.dumps({"input_dim": 3, "hidden_dims": [4]}), encoding="utf-8"
    )
    joblib.dump("pipeline", tmp_path / "preprocessor.joblib")

    loaded = LocalModelRepository(tmp_path).load()

    assert loaded["input_dim"] == 3
    assert loaded["model_source"] == "mlp"


def test_refit_and_evaluate_uses_untouched_test_set():
    X_development = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_development = np.array([0, 0, 1, 1])
    X_test = np.array([[0.5], [2.5]])
    y_test = np.array([0, 1])

    fitted, metrics = refit_and_evaluate(
        LogisticRegression(), X_development, y_development, X_test, y_test
    )

    assert fitted.predict(X_test).tolist() == [0, 1]
    assert metrics["f1"] == 1.0


def test_run_baselines_selects_using_cv_f1(monkeypatch):
    candidates = [("a", object(), {}), ("b", object(), {})]
    monkeypatch.setattr(train, "build_baselines", lambda: candidates)
    scores = iter([0.8, 0.2])

    def fake_train(pipeline, features, target, name, params):
        return {"pipeline": pipeline, "cv_metrics": {"f1": next(scores)}}

    monkeypatch.setattr(train, "train_baseline", fake_train)
    results, champion, name = train._run_baselines(pd.DataFrame(), pd.Series(dtype=int))

    assert name == "a"
    assert champion is candidates[0][1]
    assert results["a"]["cv"]["f1"] == 0.8


def _write_lock(path, deps) -> None:
    """Grava um dvc.lock mínimo com as dependências informadas no estágio preprocess."""
    path.write_text(
        yaml.safe_dump({"stages": {"preprocess": {"deps": deps}}}), encoding="utf-8"
    )


def test_dvc_dataset_hash_reads_csv_dependency(tmp_path, monkeypatch):
    lock = tmp_path / "dvc.lock"
    _write_lock(
        lock,
        [
            {"path": "src/pipeline/preprocess.py", "md5": "aaa"},
            {"path": "data/raw/Telco_customer_churn.csv", "md5": "bbb"},
        ],
    )
    monkeypatch.setattr(train, "DVC_LOCK_PATH", lock)

    assert train._dvc_dataset_hash() == "bbb"


def test_dvc_dataset_hash_returns_none_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "DVC_LOCK_PATH", tmp_path / "missing.lock")

    assert train._dvc_dataset_hash() is None


def test_dvc_dataset_hash_returns_none_without_csv_dependency(tmp_path, monkeypatch):
    lock = tmp_path / "dvc.lock"
    _write_lock(lock, [{"path": "src/pipeline/preprocess.py", "md5": "aaa"}])
    monkeypatch.setattr(train, "DVC_LOCK_PATH", lock)

    assert train._dvc_dataset_hash() is None


def test_log_run_params_tags_dataset_version(monkeypatch):
    tags: dict[str, str] = {}
    params: dict[str, object] = {}
    monkeypatch.setattr(train, "_dvc_dataset_hash", lambda: "63936da3")
    monkeypatch.setattr(train.mlflow, "set_tag", lambda k, v: tags.update({k: v}))
    monkeypatch.setattr(train.mlflow, "log_param", lambda k, v: params.update({k: v}))

    train._log_run_params(
        {
            "X_train_df": pd.DataFrame({"x": [1, 2]}),
            "X_val_df": pd.DataFrame({"x": [3]}),
            "X_test_df": pd.DataFrame({"x": [4]}),
        }
    )

    assert tags["dvc_dataset_md5"] == "63936da3"
    assert params["train_size"] == 2


def test_log_run_params_skips_tag_when_hash_unavailable(monkeypatch):
    tags: dict[str, str] = {}
    monkeypatch.setattr(train, "_dvc_dataset_hash", lambda: None)
    monkeypatch.setattr(train.mlflow, "set_tag", lambda k, v: tags.update({k: v}))
    monkeypatch.setattr(train.mlflow, "log_param", lambda *a: None)

    train._log_run_params(
        {
            "X_train_df": pd.DataFrame({"x": [1]}),
            "X_val_df": pd.DataFrame({"x": [2]}),
            "X_test_df": pd.DataFrame({"x": [3]}),
        }
    )

    assert tags == {}


def test_registry_evidence_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    version = SimpleNamespace(version="7", run_id="run-1", source="models:/m-1")

    evidence = train._write_registry_evidence(version, "logistic_regression")

    assert evidence["version"] == "7"
    assert json.loads((tmp_path / "registry.json").read_text())["run_id"] == "run-1"


def test_training_orchestrator_requires_and_records_promotion(tmp_path, monkeypatch):
    splits = {
        "X_train_df": pd.DataFrame({"x": [1]}),
        "X_val_df": pd.DataFrame({"x": [2]}),
        "X_test_df": pd.DataFrame({"x": [3]}),
        "X_train": np.ones((1, 1)),
        "X_val": np.ones((1, 1)),
        "X_test": np.ones((1, 1)),
        "y_train": pd.Series([0]),
        "y_val": pd.Series([0]),
        "y_test": pd.Series([1]),
    }
    champion_metrics = {
        "accuracy": 0.7,
        "f1": 0.6,
        "precision": 0.5,
        "recall": 0.8,
        "auc_roc": 0.85,
        "pr_auc": 0.65,
    }
    mlp_result = {
        "trainer": MagicMock(),
        "metrics": {**champion_metrics, "f1": 0.63},
    }
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "_load_splits", lambda: splits)
    monkeypatch.setattr(
        train,
        "_run_baselines",
        lambda *_: ({"logistic_regression": {"cv": {"f1": 0.64}}}, object(), "logistic_regression"),
    )
    monkeypatch.setattr(
        train,
        "_evaluate_classical_champion",
        lambda *_: {"pipeline": object(), "metrics": champion_metrics},
    )
    version = SimpleNamespace(version="1", run_id="run", source="models:/m-1")
    monkeypatch.setattr(train, "log_and_register_champion", lambda *a, **k: version)
    monkeypatch.setattr(train, "_train_mlp_experiment", lambda *a, **k: mlp_result)
    monkeypatch.setattr(train, "_save_artifacts", lambda *a: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda **k: nullcontext())
    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *a: None)
    monkeypatch.setattr(train.mlflow, "log_param", lambda *a: None)
    monkeypatch.setattr(train.mlflow, "log_metric", lambda *a: None)
    # Sem este stub, set_tag abriria um run real do MLflow (a API inicia um run
    # implicitamente quando não há nenhum ativo), fazendo o teste tocar o
    # tracking store de verdade.
    monkeypatch.setattr(train.mlflow, "set_tag", lambda *a: None)

    train.main()

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["logistic_regression"]["test"]["f1"] == 0.6
    assert (tmp_path / "registry.json").exists()
