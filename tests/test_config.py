"""Testes de configuração — separação entre segredo e configuração.

Cobrem a garantia central do Gap 3: a aplicação não sobe em produção usando os
segredos de desenvolvimento versionados no repositório.

`_env_file=None` isola cada caso do `.env` da máquina que roda os testes.
"""
import pytest
from pydantic import ValidationError

from src.utils.config import DEV_API_KEY, DEV_JWT_SECRET_KEY, Settings

REAL_JWT = "a3f1c9d2b7e04856a1c3f9d2b7e04856a1c3f9d2b7e04856a1c3f9d2b7e04856"
REAL_API_KEY = "9f2c7a1e5b3d8046af12c7e5b3d80461"


def test_development_aceita_segredos_padrao():
    """Uso local e CI funcionam sem .env — os placeholders são suficientes."""
    settings = Settings(_env_file=None, app_env="development")

    assert settings.jwt_secret_key == DEV_JWT_SECRET_KEY
    assert settings.api_key == DEV_API_KEY


def test_production_recusa_jwt_secret_padrao():
    """Container sem JWT_SECRET_KEY real deve falhar no startup, não servir tokens."""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            app_env="production",
            jwt_secret_key=DEV_JWT_SECRET_KEY,
            api_key=REAL_API_KEY,
        )

    assert "JWT_SECRET_KEY" in str(exc.value)


def test_production_recusa_api_key_padrao():
    """Mesma garantia para a API Key usada na comunicação entre serviços."""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            app_env="production",
            jwt_secret_key=REAL_JWT,
            api_key=DEV_API_KEY,
        )

    assert "API_KEY" in str(exc.value)


def test_production_reporta_os_dois_segredos_de_uma_vez():
    """A mensagem lista tudo que falta — evita corrigir um segredo por deploy."""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            app_env="production",
            jwt_secret_key=DEV_JWT_SECRET_KEY,
            api_key=DEV_API_KEY,
        )

    message = str(exc.value)
    assert "JWT_SECRET_KEY" in message
    assert "API_KEY" in message


def test_production_aceita_segredos_reais():
    """Com segredos definidos, o startup em produção é normal."""
    settings = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret_key=REAL_JWT,
        api_key=REAL_API_KEY,
    )

    assert settings.app_env == "production"
    assert settings.jwt_secret_key == REAL_JWT


@pytest.mark.parametrize("valor_vazio", ["", "   "])
def test_segredo_vazio_no_env_cai_no_padrao_de_desenvolvimento(valor_vazio):
    """`JWT_SECRET_KEY=` no .env significa ausente, não chave em branco.

    Sem isso, copiar o .env.example sem preencher produziria tokens assinados
    com string vazia em desenvolvimento.
    """
    settings = Settings(
        _env_file=None,
        app_env="development",
        jwt_secret_key=valor_vazio,
        api_key=valor_vazio,
    )

    assert settings.jwt_secret_key == DEV_JWT_SECRET_KEY
    assert settings.api_key == DEV_API_KEY


def test_app_env_invalido_e_rejeitado():
    """Typo em APP_ENV não pode degradar silenciosamente para desenvolvimento."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="prod")
