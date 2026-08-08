"""
Logging estruturado centralizado usando loguru.

Uso:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Treinamento iniciado")
    logger.warning("Classe minoritária com menos de 500 amostras")
    logger.error("Falha ao carregar modelo: {path}", path=model_path)
"""

import sys
from pathlib import Path

from loguru import logger as _logger

from src.utils.config import settings

# Remove o handler padrão do loguru
_logger.remove()

# Handler para console (stdout)
_logger.add(
    sys.stdout,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Handler para arquivo (rotativo — máximo 10MB, mantém 7 dias)
log_path = Path(__file__).parent.parent.parent / "logs" / "churn_prediction.log"
log_path.parent.mkdir(exist_ok=True)

_logger.add(
    log_path,
    level=settings.log_level,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    ),
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)


def get_logger(name: str):
    """Retorna um logger com o contexto do módulo chamador."""
    return _logger.bind(name=name)
