"""
DocuMind AI – Summarization API routes.

POST /summarize   short | medium | detailed summary of a stored document
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.generator import LLMNotConfiguredError
from app.core.summarizer import summarize_document
from app.database import get_db
from app.models import Document, MetricEvent
from app.schemas import SummarizeRequest, SummarizeResponse
from app.services.document_service import get_document_text

router = APIRouter(prefix="/summarize", tags=["summarization"])


@router.post("", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest, db: AsyncSession = Depends(get_db)) -> SummarizeResponse:
    """Generate a summary at the requested granularity and cache it on the doc."""
    started = time.perf_counter()
    doc, text = await get_document_text(db, payload.document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not text:
        raise HTTPException(status_code=422, detail="Document has no extractable text.")

    try:
        result = await summarize_document(text, payload.mode)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        db.add(MetricEvent(event_type="summarize", success=False))
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")

    # Cache the latest summary on the document record.
    doc.summary = result.summary
    db.add(
        MetricEvent(
            event_type="summarize",
            success=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    )
    await db.flush()
    return SummarizeResponse(document_id=doc.id, mode=result.mode, summary=result.summary)
