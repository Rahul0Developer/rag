"""
DocuMind AI – Intelligent text chunking.

Uses LangChain's RecursiveCharacterTextSplitter (splitting only – no chains).
The splitter tries paragraph boundaries first and falls back to sentence /
word boundaries, which keeps semantic units intact for better retrieval.
"""
from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import settings

# Separators ordered from "most semantic" to "least semantic".
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,      # default 800 chars
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,  # default 150 chars
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace produced by PDF/DOCX extraction."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse excessive blank lines
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token heuristic (good enough for telemetry)."""
    return max(1, len(text) // 4)


def chunk_document(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split raw document text into overlapping retrieval chunks.

    Returns an empty list when the input contains no usable content.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        logger.warning("chunk_document received empty text after normalisation")
        return []

    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c.strip() for c in splitter.split_text(cleaned) if c and c.strip()]
    logger.debug(f"Chunked {len(cleaned)} chars into {len(chunks)} chunks")
    return chunks
