"""
DocuMind AI – Classification API routes.

POST /classify           classify an existing document by id
GET  /classify/categories list supported categories
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.classifier import classify_document
from app.database import get_db
from app.models import Category, Document, MetricEvent
from app.schemas import ClassifyRequest, ClassifyResponse
from app.services.document_service import get_document_text

router = APIRouter(prefix="/classify", tags=["classification"])


@router.get("/categories", response_model=list[str])
async def list_categories() -> list[str]:
    return [c.value for c in Category]


@router.post("", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, db: AsyncSession = Depends(get_db)) -> ClassifyResponse:
    """Classify a stored document into one of the fixed enterprise categories."""
    doc = await db.get(Document, payload.document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    _, text = await get_document_text(db, doc.id)
    if not text:
        raise HTTPException(status_code=422, detail="Document has no extractable text.")

    result = await classify_document(text)
    doc.category = result.category
    doc.category_confidence = result.confidence
    db.add(MetricEvent(event_type="classify", success=True))
    await db.flush()

    logger.info(f"Document {doc.id} classified as {result.category}")
    return ClassifyResponse(
        document_id=doc.id,
        category=result.category,
        confidence=result.confidence,
        reasoning=result.reasoning,
    )
