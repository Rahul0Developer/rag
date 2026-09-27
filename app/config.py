"""
DocuMind AI – Centralised application configuration.

All runtime knobs are sourced from environment variables (12-factor app style)
with sensible defaults so the project runs out-of-the-box for local development.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = two levels up from this file (…/documind/app/config.py)
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "DocuMind AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"          # development | production
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------- Database
    # Default is asyncpg against PostgreSQL.  For zero-setup local dev you can
    # set DATABASE_URL=sqlite+aiosqlite:///./documind.db
    DATABASE_URL: str = "postgresql+asyncpg://documind:documind@localhost:5432/documind"
    DATABASE_ECHO: bool = False

    # ---------------------------------------------------------------- Files
    DATA_DIR: Path = BASE_DIR / "data"
    FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

    # -------------------------------------------------------------- LLM/AI
    LLM_PROVIDER: str = "groq"                # groq | openai-compatible
    GROQ_API_KEY: str = "your_groq_api_key_here"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0              # deterministic => fewer hallucinations
    LLM_MAX_TOKENS: int = 1024
    LLM_REQUEST_TIMEOUT_S: float = 60.0

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ------------------------------------------------------------ Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # ----------------------------------------------------------- Retrieval
    TOP_K: int = 5
    # Minimum cosine similarity to consider a chunk "relevant".
    MIN_RELEVANCE_SCORE: float = 0.15

    # ------------------------------------------------------------ Security
    CORS_ORIGINS: list[str] = ["*"]

    # ------------------------------------------------------------- Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/documind.log"

    def ensure_dirs(self) -> None:
        """Create runtime directories if they do not exist."""
        for directory in (self.DATA_DIR, self.FAISS_INDEX_DIR, Path(self.LOG_FILE).parent):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor – import ``get_settings()`` everywhere."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings: Settings = get_settings()
