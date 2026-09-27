"""
DocuMind AI – Flask dashboard server.

Flask's only job is to serve the UI (HTML/JS/CSS) and proxy nothing:
the browser talks directly to the FastAPI JSON API (CORS enabled).

Run:  python -m app.flask_app          (http://localhost:5000)
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from app.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


# ------------------------------------------------------------------ pages
@app.route("/")
def index() -> str:
    """Dashboard home: KPI cards + charts."""
    return render_template("index.html", page="dashboard", api_base=_api_base())


@app.route("/documents")
def documents_page() -> str:
    return render_template("documents.html", page="documents", api_base=_api_base())


@app.route("/chat")
def chat_page() -> str:
    return render_template("chat.html", page="chat", api_base=_api_base())


@app.route("/metrics")
def metrics_page() -> str:
    return render_template("metrics.html", page="metrics", api_base=_api_base())


@app.route("/settings")
def settings_page() -> str:
    return render_template(
        "settings.html",
        page="settings",
        api_base=_api_base(),
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
        groq_key_set=bool(
            settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here"
        ),
    )


# ------------------------------------------------------- small server utils
@app.route("/_health")
def health() -> "jsonify":  # type: ignore[valid-type]
    """UI-side ping so the footer can show backend status without CORS noise."""
    return jsonify({"status": "ok"})


def _api_base() -> str:
    """Base URL the browser JS should call (configurable for split hosting)."""
    return os.environ.get("PUBLIC_API_BASE", "http://localhost:8000")


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api"):
        return jsonify({"detail": "Not found"}), 404
    return render_template("base.html", page="404", api_base=_api_base()), 404


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=settings.DEBUG)
