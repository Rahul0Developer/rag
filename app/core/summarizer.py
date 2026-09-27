"""
DocuMind AI – Summarizer (thin wrapper over the Groq generator).

Long documents are handled with a map-reduce style budget: we feed the first
N characters (which in practice contain title/abstract/executive summary) and
let llama-3.3-70b-versatile produce the requested granularity.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.core.generator import summarize_text

VALID_MODES = ("short", "medium", "detailed")


@dataclass(slots=True)
class SummaryResult:
    mode: str
    summary: str
    source_chars: int


async def summarize_document(text: str, mode: str = "medium") -> SummaryResult:
    """Summarise document text in short / medium / detailed modes."""
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid summary mode '{mode}'. Expected one of {VALID_MODES}.")
    if not text.strip():
        return SummaryResult(mode=mode, summary="The document contains no readable text.", source_chars=0)

    logger.info(f"Summarising {len(text)} chars in '{mode}' mode")
    summary = await summarize_text(text, mode)
    return SummaryResult(mode=mode, summary=summary, source_chars=len(text))
