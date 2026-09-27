"""
DocuMind AI – FAISS vector store wrapper.

Design notes
------------
* One flat ``IndexIDMap(IndexFlatIP)`` index over L2-normalised vectors →
  inner product == cosine similarity.
* The index is **rebuildable from PostgreSQL chunks** (see ``rebuild_from_db``),
  which matters on ephemeral free hosts (Render) where disk does not persist.
* All mutating operations are persisted to disk immediately.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.embeddings import embed_texts
from app.models import Chunk, Document


class VectorStore:
    """Thread-safe async FAISS store keyed by chunk ``faiss_id``."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir: Path = index_dir or settings.FAISS_INDEX_DIR
        self.index_path: Path = self.index_dir / "documind.faiss"
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None

    # ------------------------------------------------------------ lifecycle
    def _new_index(self) -> faiss.Index:
        # IndexFlatIP + IDMap: exact cosine search with explicit integer ids.
        base = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
        return faiss.IndexIDMap(base)

    @property
    def index(self) -> faiss.Index:
        """Lazily load (or create) the underlying FAISS index."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    if self.index_path.exists():
                        try:
                            self._index = faiss.read_index(str(self.index_path))
                            logger.info(
                                f"Loaded FAISS index ({self._index.ntotal} vectors) "
                                f"from {self.index_path}"
                            )
                        except Exception as exc:  # corrupted file → start fresh
                            logger.error(f"Failed to read FAISS index: {exc}. Recreating.")
                            self._index = self._new_index()
                    else:
                        self._index = self._new_index()
                        logger.info("Created new empty FAISS index.")
        return self._index

    def _persist(self) -> None:
        """Write the current index to disk (call while holding the lock)."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))

    # ------------------------------------------------------------- mutation
    async def add_vectors(self, ids: list[int], vectors: np.ndarray) -> None:
        """Add L2-normalised vectors with explicit integer ids and persist."""
        if vectors.size == 0:
            return
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        id_arr = np.asarray(ids, dtype=np.int64)

        def _add() -> None:
            with self._lock:
                self.index.add_with_ids(arr, id_arr)
                self._persist()

        await asyncio.to_thread(_add)
        logger.debug(f"Added {len(ids)} vectors to FAISS index (total={self.index.ntotal})")

    async def remove_document(self, document_id: int, db: AsyncSession) -> None:
        """Remove every vector belonging to a document, then persist."""
        result = await db.execute(select(Chunk.faiss_id).where(Chunk.document_id == document_id))
        ids = [int(i) for i in result.scalars().all()]
        if not ids:
            return

        def _remove() -> None:
            with self._lock:
                self.index.remove_ids(np.asarray(ids, dtype=np.int64))
                self._persist()

        await asyncio.to_thread(_remove)
        logger.info(f"Removed {len(ids)} vectors for document {document_id}")

    async def reset(self) -> None:
        """Drop the entire index (used by rebuild flows)."""
        def _reset() -> None:
            with self._lock:
                self._index = self._new_index()
                self._persist()

        await asyncio.to_thread(_reset)
        logger.warning("FAISS index reset to empty.")

    # -------------------------------------------------------------- queries
    async def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k ``(faiss_id, cosine_score)`` pairs, best first."""
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        def _search() -> tuple[np.ndarray, np.ndarray]:
            with self._lock:
                return self.index.search(q, max(1, k))

        scores, ids = await asyncio.to_thread(_search)
        results: list[tuple[int, float]] = []
        for fid, score in zip(ids[0], scores[0]):
            if fid == -1:  # faiss sentinel for "no more results"
                continue
            results.append((int(fid), float(score)))
        return results

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    # ------------------------------------------------------------- rebuild
    async def rebuild_from_db(self, db: AsyncSession) -> int:
        """Re-create the FAISS index from the ``chunks`` table.

        Embeddings are deterministic for a fixed model, so re-embedding every
        stored chunk reproduces an equivalent index.  This makes the system
        resilient on free hosting tiers without persistent disks.

        Returns the number of vectors indexed.
        """
        await self.reset()
        result = await db.execute(
            select(Chunk.id, Chunk.content, Chunk.faiss_id)
            .join(Document, Document.id == Chunk.document_id)
            .order_by(Chunk.faiss_id)
        )
        rows = result.all()
        if not rows:
            logger.info("No chunks found – rebuilt index is empty.")
            return 0

        contents = [r.content for r in rows]
        vectors = await embed_texts(contents)
        ids = [int(r.faiss_id) for r in rows]
        await self.add_vectors(ids, vectors)
        logger.info(f"Rebuilt FAISS index with {len(ids)} vectors from PostgreSQL.")
        return len(ids)


# Module-level singleton shared by the API layer.
vector_store = VectorStore()
