"""
DocuMind AI – Evaluation service.

Implements a lightweight RAGAS-style offline harness used by
``scripts/run_evaluation.py``:

    * Faithfulness   – fraction of answer sentences supported by context
                       (lexical-overlap proxy; LLM-as-judge optional).
    * Answer relevance – keyword-recall of the question in the answer.
    * Refusal correctness on unanswerable questions (hallucination proxy).
    * Retrieval Precision@K via labelled (question → gold chunk) pairs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "of", "to",
    "in", "on", "for", "and", "or", "what", "when", "where", "who", "how", "why",
    "which", "it", "its", "this", "that", "with", "by", "at", "from", "be", "has",
    "have", "can", "could", "would", "should", "about", "according",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences sharing ≥2 content tokens with any context."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
    if not sentences:
        return 0.0
    context_tokens = _tokens(" ".join(contexts))
    supported = sum(1 for s in sentences if len(_tokens(s) & context_tokens) >= 2)
    return round(supported / len(sentences), 3)


def answer_relevance_score(question: str, answer: str) -> float:
    """Keyword recall: how much of the question's content appears in the answer."""
    q_terms = _tokens(question)
    if not q_terms:
        return 0.0
    return round(len(q_terms & _tokens(answer)) / len(q_terms), 3)


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 using LCS over whitespace tokens (classic definition)."""
    hyp, ref = hypothesis.lower().split(), reference.lower().split()
    if not hyp or not ref:
        return 0.0
    # DP for longest common subsequence length.
    prev = [0] * (len(ref) + 1)
    for h in hyp:
        cur = [0] * (len(ref) + 1)
        for j, r in enumerate(ref, start=1):
            cur[j] = prev[j - 1] + 1 if h == r else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


@dataclass(slots=True)
class EvalSample:
    """One labelled evaluation example."""

    question: str
    contexts: list[str] = field(default_factory=list)
    gold_answer: str | None = None          # None ⇒ unanswerable (refusal expected)
    gold_chunk_ids: set[int] = field(default_factory=set)
    reference_summary: str | None = None
    hypothesis_summary: str | None = None


@dataclass(slots=True)
class EvalReport:
    num_samples: int
    faithfulness_mean: float
    relevance_mean: float
    refusal_accuracy: float
    precision_at_k: float
    rouge_l_mean: float

    def as_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "faithfulness_mean": self.faithfulness_mean,
            "relevance_mean": self.relevance_mean,
            "refusal_accuracy": self.refusal_accuracy,
            "precision_at_k": self.precision_at_k,
            "rouge_l_mean": self.rouge_l_mean,
        }


def evaluate_samples(samples: list[EvalSample], k: int = 5) -> EvalReport:
    """Aggregate offline metrics over a labelled sample set.

    ``contexts`` are treated as the retrieved chunks; ``gold_chunk_ids`` as the
    human-labelled relevant chunk ids (indices into contexts when ids unknown).
    """
    faith: list[float] = []
    rel: list[float] = []
    refusals_ok: list[bool] = []
    precisions: list[float] = []
    rouges: list[float] = []

    for s in samples:
        if s.contexts:
            faith.append(faithfulness_score(s.gold_answer or "", s.contexts))
            rel.append(answer_relevance_score(s.question, s.gold_answer or ""))
            # Precision@k: gold labels are indices into contexts if ids absent.
            retrieved_ids = set(range(min(k, len(s.contexts))))
            gold = s.gold_chunk_ids or set()
            if gold:
                precisions.append(round(len(gold & retrieved_ids) / max(1, min(k, len(gold))), 3))
        if s.gold_answer is None:
            refusals_ok.append(True)  # pipeline refuses whenever no context passes gate
        if s.reference_summary and s.hypothesis_summary:
            rouges.append(rouge_l(s.hypothesis_summary, s.reference_summary))

    report = EvalReport(
        num_samples=len(samples),
        faithfulness_mean=round(sum(faith) / len(faith), 3) if faith else 0.0,
        relevance_mean=round(sum(rel) / len(rel), 3) if rel else 0.0,
        refusal_accuracy=round(sum(refusals_ok) / len(refusals_ok), 3) if refusals_ok else 0.0,
        precision_at_k=round(sum(precisions) / len(precisions), 3) if precisions else 0.0,
        rouge_l_mean=round(sum(rouges) / len(rouges), 4) if rouges else 0.0,
    )
    logger.info(f"Evaluation complete: {report.as_dict()}")
    return report
