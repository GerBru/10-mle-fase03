"""
Segurança da API: JWT, API Key, rate limiting e repositório de usuários.
"""

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.utils.config import settings

JWT_SECRET_KEY = settings.jwt_secret_key
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = settings.jwt_expire_minutes
API_KEY = settings.api_key
RATE_LIMIT_REQUESTS = settings.rate_limit_requests
RATE_LIMIT_WINDOW = settings.rate_limit_window

http_bearer = HTTPBearer(auto_error=False)

# Estado de rate limiting (por IP)
_request_history: dict = defaultdict(deque)


class InMemoryUserRepository:
    """Repositório de usuários em memória com senhas bcrypt."""

    def __init__(self) -> None:
        self._db: dict[str, dict] = {
            "admin": {"password": bcrypt.hashpw(b"admin123", bcrypt.gensalt()), "role": "admin"},
            "user": {"password": bcrypt.hashpw(b"user123", bcrypt.gensalt()), "role": "user"},
        }

    def get(self, username: str) -> dict | None:
        return self._db.get(username)

    def authenticate(self, username: str, password: str) -> dict | None:
        user = self.get(username)
        if user and bcrypt.checkpw(password.encode(), user["password"]):
            return user
        return None


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ausente. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token inválido ou expirado: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key ausente.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida.")
    return x_api_key


def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Verifica rate limit para o IP. Retorna (bloqueado, retry_after_seconds)."""
    now = time.time()
    history = _request_history[client_ip]
    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()
    if len(history) >= RATE_LIMIT_REQUESTS:
        return True, int(RATE_LIMIT_WINDOW - (now - history[0]))
    history.append(now)
    return False, 0


def remaining_requests(client_ip: str) -> int:
    history = _request_history[client_ip]
    return max(0, RATE_LIMIT_REQUESTS - len(history))
