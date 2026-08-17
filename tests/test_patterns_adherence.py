"""Testes das correções de aderência aos padrões documentados no README.

Cobrem duas afirmações que, sem estes pontos, seriam apenas intenção:

- **Repository / DIP**: existe um ponto único de troca da implementação, e a
  API não instancia a classe concreta.
- **Continued Model Evaluation**: o pré-processamento persiste a distribuição
  de referência que `src/monitoring/drift_detection.py` consome.
"""
import numpy as np
import pytest

from src.api.model_loader import (
    LocalModelRepository,
    build_model_repository,
)
from src.monitoring.drift_detection import load_reference_stats
from src.pipeline.preprocess import _feature_names


class _PipelineComNomes:
    """Pipeline que expõe nomes de features, como um ColumnTransformer ajustado."""

    def get_feature_names_out(self):
        return np.array(["tenure", "monthly_charges", "contract_Two_year"])


class _PipelineSemNomes:
    """Transformer customizado sem introspecção de nomes."""


def test_factory_retorna_implementacao_do_protocol():
    """A factory devolve algo que cumpre o contrato de ModelRepository.

    A verificação é estrutural: `Protocol` sem `@runtime_checkable` não suporta
    isinstance, e a conformidade formal já é garantida em tempo de análise
    estática pela anotação de retorno da factory.
    """
    repo = build_model_repository()

    assert callable(getattr(repo, "load", None))
    assert isinstance(repo, LocalModelRepository)


def test_factory_aceita_diretorio_explicito(tmp_path):
    """Permite apontar para artefatos de teste sem alterar a configuração global."""
    repo = build_model_repository(tmp_path)

    assert repo._dir == tmp_path


def test_api_nao_instancia_repositorio_concreto():
    """A API depende do Protocol e da factory — não da implementação.

    Garante que a inversão de dependência descrita no README seja verdadeira:
    trocar a origem dos artefatos não deve exigir edição em app.py.
    """
    import src.api.app as app_module

    assert hasattr(app_module, "build_model_repository")
    assert not hasattr(app_module, "LocalModelRepository")


def test_feature_names_usa_o_pipeline_quando_disponivel():
    nomes = _feature_names(_PipelineComNomes(), n_features=3)

    assert nomes == ["tenure", "monthly_charges", "contract_Two_year"]


def test_feature_names_cai_em_nomes_posicionais():
    """Transformer sem get_feature_names_out não pode inviabilizar o artefato."""
    nomes = _feature_names(_PipelineSemNomes(), n_features=3)

    assert nomes == ["feature_0", "feature_1", "feature_2"]


def test_reference_stats_sao_recarregaveis(tmp_path):
    """O artefato salvo alimenta ks_test/psi com a distribuição de treino."""
    from src.monitoring.drift_detection import save_reference_stats

    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    destino = str(tmp_path / "reference_stats.npz")

    save_reference_stats(X_train, ["tenure", "monthly_charges"], destino)
    recarregado = load_reference_stats(destino)

    assert set(recarregado) == {"tenure", "monthly_charges"}
    np.testing.assert_array_equal(recarregado["tenure"], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("funcao", ["ks_test", "psi", "analyze_drift"])
def test_toolkit_de_drift_continua_exposto(funcao):
    """Guarda contra o módulo voltar a ficar órfão em refatorações futuras."""
    import src.monitoring.drift_detection as drift

    assert callable(getattr(drift, funcao))
