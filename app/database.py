"""
DocuMind AI – Async SQLAlchemy 2.0 database layer.

Exposes:
    * engine / async_session_factory
    * Base (declarative base)
    * get_db() FastAPI dependency yielding an AsyncSession
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from loguru import logger

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


def _build_engine() -> AsyncEngine:
    """Create the async engine with driver-appropriate pool arguments."""
    kwargs: dict = {"echo": settings.DATABASE_ECHO, "future": True}
    if settings.DATABASE_URL.startswith("postgresql"):
        # Conservative pool sizing – safe for free-tier Postgres (Neon).
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
    else:
        # SQLite (aiosqlite) ignores pooling args; keep check_same_thread off.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(settings.DATABASE_URL, **kwargs)


engine: AsyncEngine = _build_engine()

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a request-scoped AsyncSession."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (dev convenience; production uses Alembic)."""
    # Import models so they are registered on Base.metadata before create_all.
    import app.models  # noqa: F401  (side-effect import)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialised (create_all).")


async def dispose_db() -> None:
    """Gracefully close the connection pool on shutdown."""
    await engine.dispose()
    logger.info("Database connection pool disposed.")
