"""
DocuMind AI – Grounded Q&A API routes.

POST /qa                ask a grounded question (creates/continues conversation)
GET  /qa/conversations  list conversations
GET  /qa/conversations/{id}  full message history with citations
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.generator import NO_ANSWER, LLMNotConfiguredError, generate_answer
from app.core.retriever import retrieve
from app.database import get_db
from app.models import Conversation, Message, MessageRole, MetricEvent
from app.schemas import (
    Citation,
    ConversationDetailOut,
    ConversationOut,
    MessageOut,
    QARequest,
    QAResponse,
)

router = APIRouter(prefix="/qa", tags=["question-answering"])


@router.post("", response_model=QAResponse)
async def ask(payload: QARequest, db: AsyncSession = Depends(get_db)) -> QAResponse:
    """Answer strictly from retrieved document context, with citations."""
    started = time.perf_counter()

    # 1. Resolve or create the conversation.
    if payload.conversation_id is not None:
        conversation = await db.get(Conversation, payload.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = Conversation(title=payload.question[:80])
        db.add(conversation)
        await db.flush()

    # Persist the user turn immediately (history survives even on failure).
    db.add(Message(conversation_id=conversation.id, role=MessageRole.USER, content=payload.question))
    await db.flush()

    # 2. Retrieve relevant chunks.
    retrieval = await retrieve(payload.question, db, top_k=payload.top_k)

    # 3. Generate a grounded answer.
    try:
        answer, gen_ms, grounded = await generate_answer(payload.question, retrieval)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("QA generation error")
        db.add(MetricEvent(event_type="qa", success=False))
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")

    total_ms = int((time.perf_counter() - started) * 1000)
    answered_from_context = grounded and answer != NO_ANSWER

    # 4. Build citations for the UI (top 5 sources actually shown to the model).
    citations = [
        Citation(
            chunk_id=rc.chunk_id,
            document_id=rc.document_id,
            document_name=rc.document_name,
            snippet=rc.content[:280] + ("…" if len(rc.content) > 280 else ""),
            score=rc.score,
        )
        for rc in retrieval.chunks[:5]
    ]

    # 5. Persist assistant turn + telemetry.
    db.add(
        Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=json.dumps([c.model_dump() for c in citations]),
            confidence=retrieval.confidence,
            answered_from_context=answered_from_context,
            latency_ms=total_ms,
        )
    )
    db.add(
        MetricEvent(
            event_type="qa",
            success=True,
            latency_ms=total_ms,
            payload=json.dumps({"grounded": answered_from_context, "sources": len(citations)}),
        )
    )
    await db.flush()

    logger.info(
        f"QA answered in {total_ms} ms (retrieval={retrieval.latency_ms} ms, "
        f"generation={gen_ms} ms, grounded={answered_from_context})"
    )
    return QAResponse(
        conversation_id=conversation.id,
        answer=answer,
        citations=citations,
        confidence=retrieval.confidence,
        answered_from_context=answered_from_context,
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval.latency_ms,
        generation_latency_ms=gen_ms,
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(db: AsyncSession = Depends(get_db)) -> list[Conversation]:
    rows = await db.execute(select(Conversation).order_by(Conversation.created_at.desc()).limit(50))
    return list(rows.scalars().all())


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = (
        await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    ).scalars().all()
    return {"conversation": conversation, "messages": list(messages)}
