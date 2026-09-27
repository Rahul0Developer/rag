"""
DocuMind AI – Document classifier (thin wrapper over the Groq generator).

Kept as its own module so classification policy (categories, heuristics) is
isolated from raw LLM plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.core.generator import classify_text


@dataclass(slots=True)
class ClassificationResult:
    category: str
    confidence: float
    reasoning: str


# Cheap keyword pre-checks used to (a) give the LLM a hint and (b) provide a
# graceful fallback when the LLM is unavailable.
_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "Invoice": ("invoice", "bill to", "amount due", "tax id", "po number", "payment terms"),
    "Contract": ("agreement", "terms and conditions", "parties", "clause", "signature", "hereinafter"),
    "Resume": ("experience", "education", "skills", "objective", "resume", "curriculum vitae"),
    "Research Paper": ("abstract", "methodology", "references", "results", "doi", "peer-reviewed"),
    "Policy": ("policy", "compliance", "effective date", "guidelines", "regulation", "amendment"),
    "Report": ("report", "findings", "quarterly", "executive summary", "analysis", "recommendations"),
}


def heuristic_classify(text: str) -> tuple[str, float]:
    """Zero-dependency keyword scorer; returns (category, pseudo-confidence)."""
    lowered = text.lower()[:6000]
    best_cat, best_hits = "Other", 0
    for category, keywords in _KEYWORD_HINTS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_cat, best_hits = category, hits
    confidence = min(0.9, 0.3 + 0.1 * best_hits) if best_hits else 0.2
    return best_cat, confidence


async def classify_document(text: str, use_llm: bool = True) -> ClassificationResult:
    """Classify document text. Falls back to heuristics if the LLM errors."""
    if not text.strip():
        return ClassificationResult(category="Other", confidence=0.0, reasoning="Empty document.")

    if use_llm:
        try:
            category, confidence, reasoning = await classify_text(text)
            logger.info(f"LLM classification → {category} ({confidence:.2f})")
            return ClassificationResult(category, confidence, reasoning or "")
        except Exception as exc:
            logger.warning(f"LLM classification failed ({exc}); using keyword fallback.")

    category, confidence = heuristic_classify(text)
    return ClassificationResult(
        category=category,
        confidence=confidence,
        reasoning="Keyword-based fallback classification (LLM unavailable).",
    )
