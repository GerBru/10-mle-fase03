"""
Repositório de artefatos de modelo: abstração (ModelRepository) e implementação local.
"""

import json
from pathlib import Path
from typing import Protocol

import joblib
import torch

from src.models.mlp import ChurnMLP
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRepository(Protocol):
    """Contrato para carregamento de artefatos de treino."""

    def load(self) -> dict:
        """Retorna {"pipeline": ..., "model": ..., "input_dim": ...}"""
        ...


class LocalModelRepository:
    """Carrega pipeline e modelo MLP do filesystem local."""

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    def load(self) -> dict:
        pipeline = self._load_pipeline()
        model, input_dim = self._load_model()
        return {
            "pipeline": pipeline,
            "model": model,
            "input_dim": input_dim,
            "model_source": "mlp",
            "model_metadata": {"framework": "pytorch"},
        }

    def _load_pipeline(self):
        pipeline_path = self._dir / "preprocessor.joblib"
        legacy_path = self._dir / "preprocessing_pipeline.joblib"
        if pipeline_path.exists():
            return joblib.load(pipeline_path)
        if legacy_path.exists():
            return joblib.load(legacy_path)
        raise FileNotFoundError(f"No pipeline found in {self._dir}")

    def _load_model(self) -> tuple:
        input_dim, hidden_dims = self._read_config()
        pt_path = self._dir / "mlp_model.pt"
        legacy_pt = self._dir / "mlp_weights.pt"

        if pt_path.exists():
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
            input_dim = ckpt.get("input_dim", input_dim)
            hidden_dims = ckpt.get("hidden_dims", hidden_dims)
            model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.load_state_dict(ckpt["state_dict"])
        elif legacy_pt.exists():
            state_dict = torch.load(legacy_pt, map_location="cpu", weights_only=True)
            model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model .pt found in {self._dir}")

        model.eval()
        logger.info("Model ready — input_dim={}", input_dim)
        return model, input_dim

    def _read_config(self) -> tuple[int | None, list[int]]:
        cfg_path = self._dir / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("input_dim"), cfg.get("hidden_dims", [64, 32, 16])
        return None, [64, 32, 16]


class ChampionModelRepository:
    """Carrega o pipeline sklearn que foi promovido como champion no treino."""

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    def load(self) -> dict:
        model_path = self._dir / "best_baseline.joblib"
        evidence_path = self._dir / "registry.json"
        if not model_path.exists() or not evidence_path.exists():
            raise FileNotFoundError(
                "Champion artifacts missing; run the DVC training pipeline first"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        return {
            "pipeline": None,
            "model": joblib.load(model_path),
            "input_dim": None,
            "model_source": "champion",
            "model_metadata": evidence,
        }


def build_model_repository(
    models_dir: Path | None = None, model_source: str | None = None
) -> ModelRepository:
    """Constrói a implementação de `ModelRepository` usada pela aplicação.

    Ponto único de escolha da origem dos artefatos. A API depende apenas do
    Protocol; trocar o carregamento local por outra fonte (por exemplo, o MLflow
    Model Registry) significa retornar outra implementação daqui, sem alterar
    `src/api/app.py`.

    Args:
        models_dir: Diretório de artefatos. Usa `settings.models_dir` se omitido.

    Returns:
        Implementação de `ModelRepository` pronta para uso.
    """
    artifact_dir = models_dir or Path(settings.models_dir)
    source = model_source or settings.model_source
    if source == "champion":
        return ChampionModelRepository(artifact_dir)
    return LocalModelRepository(artifact_dir)
