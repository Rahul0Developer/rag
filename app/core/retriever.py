"""
DocuMind AI – Retriever.

Maps FAISS hits back to database chunks and assembles a numbered context
block for the generator.  Confidence is derived from retrieval scores:
the stronger the top matches, the more we trust the eventual answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.embeddings import embed_query
from app.core.vectorstore import vector_store
from app.models import Chunk, Document


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk with its relevance metadata, ready for prompting."""

    chunk_id: int
    faiss_id: int
    document_id: int
    document_name: str
    content: str
    score: float  # cosine similarity in [-1, 1] (typically 0..1)


@dataclass(slots=True)
class RetrievalResult:
    """Full retrieval outcome including prompt-ready context."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: int = 0

    @property
    def has_context(self) -> bool:
        return len(self.chunks) > 0

    def build_context_block(self) -> str:
        """Numbered context block injected into the strict system prompt."""
        parts: list[str] = []
        for i, rc in enumerate(self.chunks, start=1):
            parts.append(
                f"[Source {i} | document: {rc.document_name} | chunk: {rc.chunk_id}]\n{rc.content}"
            )
        return "\n\n---\n\n".join(parts)


async def retrieve(
    question: str,
    db: AsyncSession,
    top_k: int | None = None,
) -> RetrievalResult:
    """Embed ``question`` → FAISS ANN search → hydrate chunks from Postgres."""
    import time

    started = time.perf_counter()
    k = top_k or settings.TOP_K

    query_vector = await embed_query(question)
    hits = await vector_store.search(query_vector, k=k)

    if not hits:
        logger.info("Retrieval returned zero candidates (empty index?).")
        return RetrievalResult(chunks=[], confidence=0.0, latency_ms=_ms(started))

    faiss_ids = [fid for fid, _ in hits]
    score_map = dict(hits)

    result = await db.execute(
        select(Chunk, Document.original_name)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.faiss_id.in_(faiss_ids))
    )
    rows = result.all()

    retrieved: list[RetrievedChunk] = []
    for chunk, doc_name in rows:
        score = score_map.get(chunk.faiss_id, 0.0)
        # Relevance gate: drop weak matches so the model never sees noise.
        if score < settings.MIN_RELEVANCE_SCORE:
            continue
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                faiss_id=chunk.faiss_id,
                document_id=chunk.document_id,
                document_name=doc_name,
                content=chunk.content,
                score=round(score, 4),
            )
        )

    retrieved.sort(key=lambda r: r.score, reverse=True)

    # Confidence heuristic: blend of top-1 relevance and mean top-k relevance.
    if retrieved:
        top1 = retrieved[0].score
        mean_topk = sum(r.score for r in retrieved) / len(retrieved)
        confidence = round(min(1.0, max(0.0, 0.6 * top1 + 0.4 * mean_topk)), 3)
    else:
        confidence = 0.0

    latency = _ms(started)
    logger.info(
        f"Retrieved {len(retrieved)}/{k} relevant chunks "
        f"(top_score={retrieved[0].score if retrieved else 0:.3f}) in {latency} ms"
    )
    return RetrievalResult(chunks=retrieved, confidence=confidence, latency_ms=latency)


def _ms(started: float) -> int:
    import time

    return int((time.perf_counter() - started) * 1000)
