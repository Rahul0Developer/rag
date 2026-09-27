"""
DocuMind AI – File utilities: safe upload handling + text extraction.

Extraction backends (LangChain document loaders only – no chains):
    PDF  → PyPDFLoader
    DOCX → TextLoader via python-docx fallback? We use UnstructuredWordDocumentLoader
    TXT/MD → TextLoader
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import aiofiles
from loguru import logger

from app.config import settings


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file has a disallowed extension."""


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds MAX_UPLOAD_SIZE_MB."""


def sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters from a user filename."""
    name = Path(name).name                       # defend against ../ traversal
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name).strip() or "document"
    return name[:200]


def unique_storage_name(original_name: str) -> str:
    """Generate a collision-free storage filename: <uuid8>-<sanitised>."""
    safe = sanitize_filename(original_name)
    return f"{uuid.uuid4().hex[:8]}-{safe}"


def validate_extension(filename: str) -> str:
    """Return the lowercase extension; raise if unsupported."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    return ext


async def save_upload(content: bytes, storage_name: str) -> Path:
    """Persist raw upload bytes under the configured data directory."""
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise FileTooLargeError(f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.")
    path = settings.DATA_DIR / storage_name
    async with aiofiles.open(path, "wb") as fh:
        await fh.write(content)
    logger.debug(f"Saved upload to {path} ({len(content)} bytes)")
    return path


def extract_text(path: Path) -> str:
    """Extract plain text from a stored document using LangChain loaders."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(path))
        elif ext == ".docx":
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader

            loader = UnstructuredWordDocumentLoader(str(path))
        else:  # .txt / .md
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

        pages = loader.load_and_split() if ext == ".pdf" else loader.load()
        text = "\n".join(page.page_content for page in pages)
        logger.info(f"Extracted {len(text)} chars from {path.name}")
        return text
    except Exception as exc:
        logger.exception(f"Text extraction failed for {path}")
        raise RuntimeError(f"Could not extract text from {path.name}: {exc}") from exc
