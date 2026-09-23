"""
app/routers/papers.py

Routes:
    POST /api/analyze        — upload 1-5 PDFs, run pipeline, save sessions
    GET  /api/papers         — list all completed sessions
    GET  /api/papers/{id}    — get one paper session
    POST /api/rouge          — compute ROUGE given a reference abstract
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.pipeline import process_paper, run_led_summary
from src.evaluation.rouge_evaluator import ROUGEEvaluator

router = APIRouter(prefix="/api")

SESSIONS_DIR = Path("data/sessions")
RAW_DIR = Path("data/raw")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------
@router.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    """Upload 1–5 PDF files and run the pipeline on each."""
    if not (1 <= len(files) <= 5):
        raise HTTPException(status_code=400,
                            detail="Upload between 1 and 5 PDF files.")

    results = []
    for upload in files:
        if not upload.filename.endswith(".pdf"):
            raise HTTPException(status_code=400,
                                detail=f"{upload.filename} is not a PDF.")

        # Save uploaded file
        dest = RAW_DIR / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        # Run pipeline
        result = process_paper(str(dest))

        # Store session (exclude private _clean_text from JSON)
        session = {k: v for k, v in result.items() if not k.startswith("_")}
        session_path = SESSIONS_DIR / f"{result['paper_id']}.json"
        session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        # Also keep clean_text separately for LED on-demand
        ct_path = SESSIONS_DIR / f"{result['paper_id']}.txt"
        ct_path.write_text(result.get("_clean_text", ""), encoding="utf-8")

        results.append(session)

    return JSONResponse(content={"papers": results})


# ---------------------------------------------------------------------------
# GET /api/papers
# ---------------------------------------------------------------------------
@router.get("/papers")
async def list_papers():
    """Return all completed paper sessions."""
    papers = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            papers.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return JSONResponse(content={"papers": papers})


# ---------------------------------------------------------------------------
# GET /api/papers/{paper_id}
# ---------------------------------------------------------------------------
@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Return a single paper session by ID."""
    path = SESSIONS_DIR / f"{paper_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Paper not found.")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# POST /api/rouge
# ---------------------------------------------------------------------------
@router.post("/rouge")
async def compute_rouge(payload: dict):
    """
    Compute ROUGE scores for a paper.

    Body: { "paper_id": "...", "reference_abstract": "..." }

    Runs TF-IDF ROUGE immediately.
    Runs LED ROUGE only if led_summary already exists in the session;
    otherwise computes it (slow — requires torch + model).
    """
    paper_id = payload.get("paper_id", "")
    reference = payload.get("reference_abstract", "").strip()

    if not reference:
        raise HTTPException(status_code=400,
                            detail="reference_abstract must not be empty.")

    session_path = SESSIONS_DIR / f"{paper_id}.json"
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found.")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    evaluator = ROUGEEvaluator()

    # TF-IDF ROUGE
    tfidf_scores = evaluator.score(session["tfidf_summary"], reference).to_dict()

    # LED ROUGE — compute LED summary if not already done
    led_summary = session.get("led_summary")
    if not led_summary:
        ct_path = SESSIONS_DIR / f"{paper_id}.txt"
        if ct_path.exists():
            clean_text = ct_path.read_text(encoding="utf-8")
            led_summary = run_led_summary(clean_text)
            session["led_summary"] = led_summary
            session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    led_scores = {}
    if led_summary:
        led_scores = evaluator.score(led_summary, reference).to_dict()

    rouge_result = {"led": led_scores, "tfidf": tfidf_scores}

    # Persist ROUGE scores into the session
    session["rouge"] = rouge_result
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    return JSONResponse(content=rouge_result)
