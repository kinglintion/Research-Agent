from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import create_engine, text

from ..config import get_settings


@lru_cache
def mysql_engine():
    return create_engine(get_settings().mysql_url, pool_pre_ping=True, pool_size=5)


@lru_cache
def redis_client() -> Redis:
    return Redis.from_url(get_settings().redis_url, socket_connect_timeout=2)


@lru_cache
def qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_url:
        kwargs = {"url": settings.qdrant_url, "timeout": 3}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        return QdrantClient(**kwargs)

    path = Path(settings.qdrant_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def check_mysql() -> tuple[str, str | None]:
    try:
        with mysql_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok", None
    except Exception as exc:
        return "unavailable", exc.__class__.__name__


def check_redis() -> tuple[str, str | None]:
    try:
        redis_client().ping()
        return "ok", None
    except Exception as exc:
        return "unavailable", exc.__class__.__name__


def check_qdrant() -> tuple[str, str | None]:
    try:
        qdrant_client().get_collections()
        return "ok", None
    except Exception as exc:
        return "unavailable", exc.__class__.__name__
