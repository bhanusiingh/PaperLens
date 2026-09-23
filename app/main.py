"""
app/main.py

FastAPI application entry point for PaperLens.

Serves:
    GET /           → frontend/paperlens_ui_prototype.html (the UI)
    GET /static/*   → static assets (if any)
    /api/*          → NLP pipeline API routes

Start with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import papers, compare, evaluation

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PaperLens API",
    description="Structured research paper summarisation using LED Transformer and TF-IDF.",
    version="2.0.0",
)

# Allow browser fetch() calls from the same origin or localhost during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------
app.include_router(papers.router)
app.include_router(compare.router)
app.include_router(evaluation.router)

# ---------------------------------------------------------------------------
# Frontend — serve the HTML prototype
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
HTML_FILE = FRONTEND_DIR / "paperlens_ui_prototype.html"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the PaperLens UI prototype."""
    return FileResponse(str(HTML_FILE))


# Mount any static assets (fonts, icons) if needed in future
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")),
              name="static")
