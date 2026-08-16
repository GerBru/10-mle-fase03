"""
Configuração centralizada do projeto.
Carrega variáveis do arquivo .env e expõe um objeto `settings`
com todas as configurações necessárias para treino, API e MLflow.
Uso:
    from src.utils.config import settings
    print(settings.seed)                   # 42
    print(settings.mlflow_tracking_uri)    # http://localhost:5001

Segredos (`jwt_secret_key`, `api_key`) têm valor padrão apenas para
desenvolvimento e testes. Em `APP_ENV=production` o startup falha se eles não
forem sobrescritos por variável de ambiente — ver `_reject_placeholder_secrets`.
"""
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto — sempre absoluta, independente de onde o código é executado
# src/utils/config.py → src/utils → src → raiz
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Valores de desenvolvimento. São públicos por definição (estão versionados) e
# servem apenas para que testes e execução local funcionem sem `.env`.
# `APP_ENV=production` recusa qualquer um deles no startup.
DEV_JWT_SECRET_KEY = "dev-insecure-jwt-secret-change-me"
DEV_API_KEY = "dev-insecure-api-key-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ambiente de execução — controla a exigência de segredos reais
    app_env: Literal["development", "production"] = Field(
        default="development",
        description="Use 'production' em qualquer ambiente exposto; exige segredos reais.",
    )

    # Reprodutibilidade
    seed: int = Field(default=42, description="Seed global para reprodutibilidade")

    # Caminhos — sempre relativos à raiz do projeto
    data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
    )
    log_path: Path = Field(default=PROJECT_ROOT / "logs" / "churn_prediction.log")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    mlflow_experiment_name: str = Field(default="churn-prediction")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_version: str = Field(default="1.0.0")
    models_dir: str = Field(default="models")

    # Segurança — JWT
    jwt_secret_key: str = Field(
        default=DEV_JWT_SECRET_KEY,
        description="Chave secreta para assinar tokens JWT. Gere com: openssl rand -hex 32",
    )
    jwt_expire_minutes: int = Field(
        default=60,
        description="Tempo de expiração do token JWT em minutos",
    )

    # Segurança — API Key
    api_key: str = Field(
        default=DEV_API_KEY,
        description="API Key para autenticação entre serviços. Gere com: openssl rand -hex 16",
    )

    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Número máximo de requisições por janela de tempo",
    )
    rate_limit_window: int = Field(
        default=60,
        description="Janela de tempo em segundos para o rate limiting",
    )

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("jwt_secret_key", mode="before")
    @classmethod
    def _default_jwt_secret(cls, value: str | None) -> str:
        """Trata `JWT_SECRET_KEY=` (vazio no .env) como ausente, não como chave em branco."""
        return value if value and value.strip() else DEV_JWT_SECRET_KEY

    @field_validator("api_key", mode="before")
    @classmethod
    def _default_api_key(cls, value: str | None) -> str:
        """Trata `API_KEY=` (vazio no .env) como ausente, não como chave em branco."""
        return value if value and value.strip() else DEV_API_KEY

    @model_validator(mode="after")
    def _reject_placeholder_secrets(self) -> "Settings":
        """Impede que a aplicação suba em produção com segredos de desenvolvimento.

        Sem esta verificação, um container sem `.env` inicia normalmente assinando
        JWT com uma chave versionada no repositório — falha silenciosa e explorável.

        Raises:
            ValueError: se `app_env` for 'production' e algum segredo continuar
                com o valor padrão de desenvolvimento ou estiver vazio.
        """
        if self.app_env != "production":
            return self

        insecure = {
            "JWT_SECRET_KEY": (self.jwt_secret_key, DEV_JWT_SECRET_KEY, "openssl rand -hex 32"),
            "API_KEY": (self.api_key, DEV_API_KEY, "openssl rand -hex 16"),
        }
        problems = [
            f"{var} não foi definida (gere com: {command})"
            for var, (value, placeholder, command) in insecure.items()
            if not value.strip() or value == placeholder
        ]
        if problems:
            raise ValueError(
                "APP_ENV=production exige segredos reais. " + "; ".join(problems)
            )
        return self


def set_global_seed(seed: int) -> None:
    """
    Fixa o seed em todas as bibliotecas para garantir reprodutibilidade.
    Deve ser chamado no início de qualquer script de treino.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Instância global — importar de qualquer módulo
settings = Settings()
