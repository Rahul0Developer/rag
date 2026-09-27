"""
DocuMind AI – Embedding provider (sentence-transformers / all-MiniLM-L6-v2).

The model is loaded lazily once and cached process-wide.  Embedding is a CPU
-bound synchronous operation, so callers run it via ``asyncio.to_thread`` to
keep the FastAPI event loop responsive.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from app.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None


def get_model() -> "SentenceTransformer":
    """Load (once) and return the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts_sync(texts: list[str]) -> np.ndarray:
    """Synchronously embed a list of texts → (n, dim) float32 L2-normalised."""
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # unit length => dot product == cosine sim
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Async wrapper around :func:`embed_texts_sync`."""
    return await asyncio.to_thread(embed_texts_sync, texts)


async def embed_query(query: str) -> np.ndarray:
    """Embed a single query; returns shape (dim,)."""
    vec = await embed_texts([query])
    return vec[0]
