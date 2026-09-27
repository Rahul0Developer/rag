"""
DocuMind AI – FastAPI application entrypoint.

Run (dev):   uvicorn app.main:app --reload --port 8000
Run (prod):  uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy import text

from app.api import classify, documents, metrics, qa, summarize
from app.config import settings
from app.core.vectorstore import vector_store
from app.database import async_session_factory, dispose_db, init_db
from app.schemas import HealthResponse

# --------------------------------------------------------------- logging
logger.remove()  # default stderr handler with less structure
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>",
)
logger.add(settings.LOG_FILE, rotation="10 MB", retention="14 days", level="DEBUG")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle: DB schema + FAISS warm-up."""
    settings.ensure_dirs()
    try:
        await init_db()
    except Exception as exc:
        # Free hosts sometimes boot before Postgres is reachable; keep serving.
        logger.error(f"Database initialisation failed: {exc}")
    logger.info(
        f"{settings.APP_NAME} v{settings.APP_VERSION} started "
        f"(provider={settings.LLM_PROVIDER}, model={settings.LLM_MODEL})"
    )
    yield
    await dispose_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Document Intelligence Platform — grounded RAG over your "
        "documents with citations, classification, summarization and evaluation metrics."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Attach X-Process-Time header + structured access logs."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}"
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f} ms)")
    return response


# ------------------------------------------------------------------ routers
for module in (documents, qa, classify, summarize, metrics):
    app.include_router(module.router, prefix=settings.API_PREFIX)


# ------------------------------------------------------------------ health
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> dict:
    """Liveness/readiness probe used by Render and the dashboard footer."""
    db_status = "unknown"
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.warning(f"Health check: database unreachable ({exc.__class__.__name__})")

    llm_configured = bool(
        settings.GROQ_API_KEY
        and settings.GROQ_API_KEY != "your_groq_api_key_here"
        or (settings.LLM_PROVIDER != "groq" and settings.OPENAI_API_KEY)
    )
    return {
        "status": "ok" if db_status == "connected" else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER + (" (key set)" if llm_configured else " (NO KEY)"),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL.split("/")[-1],
        "database": db_status,
        "faiss_vectors": vector_store.count,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log full traceback, return a clean JSON error."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
