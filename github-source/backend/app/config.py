from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Research Copilot API"
    app_environment: str = "development"
    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:3000"

    llm_api_base: str = "https://api.example.com/v1"
    llm_api_key: str = "replace-with-provider-api-key"
    llm_model: str = "replace-with-provider-model-id"
    llm_context_window: int = 128000
    llm_timeout_seconds: float = 60.0

    # Optional OpenAI-compatible multimodal endpoint for figure/diagram analysis.
    # Keep empty until the user selects a provider and supplies credentials.
    vision_api_base: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    public_api_base_url: str = "http://127.0.0.1:8765"

    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    expected_embedding_dimension: int = 1024

    mysql_url: str = (
        "mysql+pymysql://research:replace-with-local-mysql-password@127.0.0.1:3306/"
        "research_copilot?charset=utf8mb4"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    qdrant_path: str = ".data/qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    paper_storage_dir: str = ".data/papers"
    pdf_parser: str = "auto"
    grobid_url: str = ""
    formula_ocr_enabled: bool = True
    formula_ocr_model: str = "PP-FormulaNet_plus-M"
    max_pdf_bytes: int = 104857600
    qdrant_papers_collection: str = "research_paper_chunks"

    github_token: str = ""
    scholarly_search_cache_seconds: int = 3600
    scholarly_http_timeout_seconds: float = 15.0
    arxiv_refresh_interval_seconds: int = 86400

    enable_model_diagnostics: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
