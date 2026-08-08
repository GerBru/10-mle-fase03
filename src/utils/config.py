"""
Configuração centralizada do projeto.
Carrega variáveis do arquivo .env e expõe um objeto `settings`
com todas as configurações necessárias para treino, API e MLflow.
Uso:
    from src.utils.config import settings
    print(settings.seed)           # 42
    print(settings.model_path)     # models/mlp_churn.pt
    print(settings.mlflow_uri)     # http://localhost:5001
"""
import random
from pathlib import Path

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto — sempre absoluta, independente de onde o código é executado
# src/utils/config.py → src/utils → src → raiz
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Reprodutibilidade
    seed: int = Field(default=42, description="Seed global para reprodutibilidade")

    # Caminhos — sempre relativos à raiz do projeto
    model_path: Path = Field(default=PROJECT_ROOT / "models" / "mlp_churn.pt")
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
        default="churn-secret-key-fiap-tech-challenge-2026",
        description="Chave secreta para assinar tokens JWT. Em produção use: openssl rand -hex 32",
    )
    jwt_expire_minutes: int = Field(
        default=60,
        description="Tempo de expiração do token JWT em minutos",
    )

    # Segurança — API Key
    api_key: str = Field(
        default="churn-api-key-fiap-2026",
        description="API Key para autenticação entre serviços. Em produção use valor aleatório forte.",
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
