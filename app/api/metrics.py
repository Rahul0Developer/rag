"""
DocuMind AI – Metrics API routes.

GET /metrics  benchmark (offline evaluation) + live telemetry metrics
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Chunk, Document, DocumentStatus, Message, MessageRole, MetricEvent
from app.schemas import LiveMetrics, MetricsResponse
from app.utils.metrics import BENCHMARK_METRICS, percentile

router = APIRouter(prefix="/metrics", tags=["metrics"])


async def _live_metrics(db: AsyncSession) -> LiveMetrics:
    """Aggregate live counters from PostgreSQL."""
    total_docs = (await db.execute(select(func.count(Document.id)))).scalar_one()
    completed_docs = (
        await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.COMPLETED)
        )
    ).scalar_one()
    total_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar_one()

    total_questions = (
        await db.execute(
            select(func.count(Message.id)).where(Message.role == MessageRole.USER)
        )
    ).scalar_one()

    # Grounded rate: assistant messages that answered from context.
    grounded = (
        await db.execute(
            select(func.count(Message.id)).where(
                Message.role == MessageRole.ASSISTANT,
                Message.answered_from_context.is_(True),
            )
        )
    ).scalar_one()
    assistant_msgs = (
        await db.execute(
            select(func.count(Message.id)).where(Message.role == MessageRole.ASSISTANT)
        )
    ).scalar_one()
    grounded_rate = round(100.0 * grounded / assistant_msgs, 1) if assistant_msgs else 100.0

    # Latency distribution from QA telemetry events.
    lat_rows = (
        await db.execute(
            select(MetricEvent.latency_ms).where(
                MetricEvent.event_type == "qa",
                MetricEvent.success.is_(True),
                MetricEvent.latency_ms.is_not(None),
            )
        )
    ).scalars().all()
    latencies = [float(v) for v in lat_rows]
    p95 = int(percentile(latencies, 95)) if latencies else 0
    avg = int(sum(latencies) / len(latencies)) if latencies else 0

    # Daily event counts for the last 7 days (chart data).
    since = datetime.now(timezone.utc) - timedelta(days=7)
    daily_rows = (
        await db.execute(
            select(
                func.date(MetricEvent.created_at).label("day"),
                MetricEvent.event_type,
                func.count(MetricEvent.id),
            )
            .where(MetricEvent.created_at >= since)
            .group_by("day", MetricEvent.event_type)
            .order_by("day")
        )
    ).all()
    events_last_7_days = [
        {"date": str(day), "event_type": etype, "count": count}
        for day, etype, count in daily_rows
    ]

    return LiveMetrics(
        total_documents=total_docs,
        completed_documents=completed_docs,
        total_chunks=total_chunks,
        total_questions=total_questions,
        grounded_answer_rate_pct=grounded_rate,
        p95_latency_ms=p95,
        avg_latency_ms=avg,
        events_last_7_days=events_last_7_days,
    )


@router.get("", response_model=MetricsResponse)
async def get_metrics(db: AsyncSession = Depends(get_db)) -> MetricsResponse:
    """Full metrics payload consumed by the dashboard Metrics page."""
    return MetricsResponse(benchmark=BENCHMARK_METRICS, live=await _live_metrics(db))
