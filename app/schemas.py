"""
DocuMind AI – Pydantic v2 request/response schemas.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------- Docs
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_name: str
    file_type: str
    file_size_bytes: int
    status: str
    category: str | None = None
    category_confidence: float | None = None
    summary: str | None = None
    chunk_count: int
    error: str | None = None
    created_at: datetime


class DocumentListOut(BaseModel):
    total: int
    documents: list[DocumentOut]


# ----------------------------------------------------------------------- QA
class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    snippet: str
    score: float


class QARequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: int | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class QAResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[Citation]
    confidence: float
    answered_from_context: bool
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    confidence: float | None = None
    latency_ms: int | None = None
    created_at: datetime


class ConversationDetailOut(BaseModel):
    conversation: ConversationOut
    messages: list[MessageOut]


# --------------------------------------------------------------- Classify/Sum
class ClassifyRequest(BaseModel):
    document_id: int


class ClassifyResponse(BaseModel):
    document_id: int
    category: str
    confidence: float
    reasoning: str | None = None


class SummarizeRequest(BaseModel):
    document_id: int
    mode: str = Field(default="medium", pattern="^(short|medium|detailed)$")


class SummarizeResponse(BaseModel):
    document_id: int
    mode: str
    summary: str


# ------------------------------------------------------------------- Upload
class UploadResponse(BaseModel):
    uploaded: list[DocumentOut]
    failed: list[dict] = []


# ------------------------------------------------------------------ Metrics
class BenchmarkMetrics(BaseModel):
    """Offline evaluation results (see scripts/run_evaluation.py)."""

    hallucination_rate_before_pct: float
    hallucination_rate_after_pct: float
    hallucination_reduction_pct: float
    answer_faithfulness_pct: float
    retrieval_precision_at_5: float
    classification_f1: float
    summarization_rouge_l: float


class LiveMetrics(BaseModel):
    total_documents: int
    completed_documents: int
    total_chunks: int
    total_questions: int
    grounded_answer_rate_pct: float
    p95_latency_ms: int
    avg_latency_ms: int
    events_last_7_days: list[dict]


class MetricsResponse(BaseModel):
    benchmark: BenchmarkMetrics
    live: LiveMetrics


# ------------------------------------------------------------------- Health
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    database: str
    faiss_vectors: int
