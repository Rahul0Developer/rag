"""
DocuMind AI – Document ingestion & management service.

Orchestrates the full pipeline:
    upload → save → extract text → chunk → embed → FAISS → PostgreSQL
Also owns document deletion (DB rows + vectors + files) and index rebuilds.
"""
from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.chunking import chunk_document, estimate_tokens
from app.core.classifier import classify_document
from app.core.embeddings import embed_texts
from app.core.vectorstore import vector_store
from app.models import Chunk, Document, DocumentStatus, MetricEvent
from app.utils.file_utils import (
    extract_text,
    save_upload,
    unique_storage_name,
    validate_extension,
)


async def _next_faiss_id(db: AsyncSession) -> int:
    """Monotonic vector id allocator (max existing + 1)."""
    result = await db.execute(select(func.max(Chunk.faiss_id)))
    current = result.scalar() or 0
    return int(current) + 1


async def ingest_file(
    db: AsyncSession,
    *,
    original_name: str,
    content: bytes,
    auto_classify: bool = True,
) -> Document:
    """Full ingestion pipeline for a single uploaded file.

    Raises on invalid type / oversize / extraction failure; the caller records
    the failure and continues with remaining files.
    """
    started = time.perf_counter()
    ext = validate_extension(original_name)               # raises UnsupportedFileTypeError
    storage_name = unique_storage_name(original_name)
    path = await save_upload(content, storage_name)       # raises FileTooLargeError

    doc = Document(
        filename=storage_name,
        original_name=original_name,
        file_type=ext.lstrip("."),
        file_size_bytes=len(content),
        status=DocumentStatus.PROCESSING,
    )
    db.add(doc)
    await db.flush()  # obtain doc.id

    try:
        # 1. Text extraction (blocking CPU/IO → thread pool).
        text = await _run_blocking(extract_text, path)
        if not text.strip():
            raise ValueError("No readable text extracted from document.")

        # 2. Intelligent chunking.
        chunks = chunk_document(text)
        logger.info(f"[doc {doc.id}] '{original_name}' → {len(chunks)} chunks")

        # 3. Embed + index in FAISS.
        vectors = await embed_texts(chunks)
        base_id = await _next_faiss_id(db)
        faiss_ids = [base_id + i for i in range(len(chunks))]
        await vector_store.add_vectors(faiss_ids, vectors)

        # 4. Persist chunk rows (index is rebuildable from these).
        for i, (chunk_text, fid) in enumerate(zip(chunks, faiss_ids)):
            db.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    token_estimate=estimate_tokens(chunk_text),
                    faiss_id=fid,
                )
            )
        doc.chunk_count = len(chunks)

        # 5. Best-effort classification (never blocks ingestion).
        if auto_classify:
            try:
                cls = await classify_document(text)
                doc.category = cls.category
                doc.category_confidence = cls.confidence
            except Exception as exc:
                logger.warning(f"Classification failed for doc {doc.id}: {exc}")

        doc.status = DocumentStatus.COMPLETED
        latency = int((time.perf_counter() - started) * 1000)
        db.add(MetricEvent(event_type="ingest", success=True, latency_ms=latency))
        await db.flush()
        logger.info(f"[doc {doc.id}] ingested in {latency} ms")
        return doc

    except Exception as exc:
        doc.status = DocumentStatus.FAILED
        doc.error = str(exc)[:1000]
        db.add(MetricEvent(event_type="ingest", success=False, latency_ms=None))
        await db.flush()
        logger.error(f"Ingestion failed for '{original_name}': {exc}")
        raise


async def delete_document(db: AsyncSession, doc: Document) -> None:
    """Remove a document's vectors, chunk rows, DB record and stored file."""
    await vector_store.remove_document(doc.id, db)
    await db.delete(doc)          # chunks cascade-delete via relationship
    await db.flush()
    file_path = Path(doc.filename)
    try:
        (Path().resolve() / "data" / file_path).unlink(missing_ok=True)
    except OSError:
        pass
    logger.info(f"Deleted document {doc.id} ('{doc.original_name}')")


async def get_document_text(db: AsyncSession, document_id: int) -> tuple[Document | None, str]:
    """Reassemble full document text from persisted chunks (ordered)."""
    doc = await db.get(Document, document_id)
    if doc is None:
        return None, ""
    result = await db.execute(
        select(Chunk.content).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    )
    return doc, "\n\n".join(result.scalars().all())


async def reclassify_document(db: AsyncSession, doc: Document) -> None:
    """Run classification for an existing document and persist the result."""
    _, text = await get_document_text(db, doc.id)
    cls = await classify_document(text)
    doc.category = cls.category
    doc.category_confidence = cls.confidence
    db.add(MetricEvent(event_type="classify", success=True))
    await db.flush()


async def _run_blocking(fn, *args):
    """Run a blocking function in the default thread executor."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)
