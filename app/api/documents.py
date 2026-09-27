"""
DocuMind AI – Document API routes.

POST   /documents/upload      multi-file ingestion
GET    /documents             list (filter by status/category, paginated)
GET    /documents/{id}        detail incl. summary + category
DELETE /documents/{id}        remove doc + vectors + file
POST   /documents/rebuild-index  rebuild FAISS from PostgreSQL chunks
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Chunk, Document, DocumentStatus
from app.schemas import DocumentListOut, DocumentOut, UploadResponse
from app.services.document_service import ingest_file, delete_document
from app.core.vectorstore import vector_store
from app.utils.file_utils import FileTooLargeError, UnsupportedFileTypeError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF, DOCX, TXT or MD files"),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Ingest one or more documents. Partial success is allowed."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded: list[Document] = []
    failed: list[dict] = []

    for file in files:
        try:
            content = await file.read()
            doc = await ingest_file(
                db, original_name=file.filename or "unnamed", content=content
            )
            uploaded.append(doc)
        except (UnsupportedFileTypeError, FileTooLargeError) as exc:
            failed.append({"file": file.filename, "reason": str(exc)})
        except Exception as exc:
            logger.error(f"Upload failed for {file.filename}: {exc}")
            failed.append({"file": file.filename, "reason": "Processing error."})

    if not uploaded and failed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "All uploads failed.", "failures": failed},
        )
    return UploadResponse(uploaded=uploaded, failed=failed)


@router.get("", response_model=DocumentListOut)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    status_filter: DocumentStatus | None = Query(default=None, alias="status"),
    category: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentListOut:
    """List documents with optional filters and pagination."""
    query = select(Document).order_by(Document.created_at.desc())
    count_query = select(func.count(Document.id))
    if status_filter:
        query = query.where(Document.status == status_filter)
        count_query = count_query.where(Document.status == status_filter)
    if category:
        query = query.where(Document.category == category)
        count_query = count_query.where(Document.category == category)

    total = (await db.execute(count_query)).scalar_one()
    docs = (await db.execute(query.limit(limit).offset(offset))).scalars().all()
    return DocumentListOut(total=total, documents=list(docs))


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: int, db: AsyncSession = Depends(get_db)) -> Document:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_route(document_id: int, db: AsyncSession = Depends(get_db)) -> None:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    await delete_document(db, doc)


@router.get("/{document_id}/chunks")
async def list_chunks(document_id: int, db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Inspect a document's chunks (useful for debugging retrieval quality)."""
    rows = (
        await db.execute(
            select(Chunk.id, Chunk.chunk_index, Chunk.faiss_id, Chunk.content)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
    ).all()
    return [
        {"chunk_id": r.id, "index": r.chunk_index, "faiss_id": r.faiss_id, "content": r.content}
        for r in rows
    ]


@router.post("/rebuild-index")
async def rebuild_index(db: AsyncSession = Depends(get_db)) -> dict:
    """Rebuild the persisted FAISS index from PostgreSQL chunks.

    Essential on ephemeral free hosts: if ``faiss_index/`` is lost, call this
    once and retrieval resumes with identical results (deterministic model).
    """
    n = await vector_store.rebuild_from_db(db)
    logger.info(f"Index rebuild requested → {n} vectors")
    return {"status": "ok", "vectors": n}
