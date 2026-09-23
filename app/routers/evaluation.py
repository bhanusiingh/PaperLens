"""
app/routers/evaluation.py

Route:
    GET /api/evaluation   — aggregate ROUGE stats from all completed sessions
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")

SESSIONS_DIR = Path("data/sessions")


@router.get("/evaluation")
async def get_evaluation():
    """
    Aggregate ROUGE-1/2/L across all sessions that have rouge scores.
    Returns per-model averages and the individual session scores.
    """
    led_scores = []
    tfidf_scores = []
    session_list = []

    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rouge = session.get("rouge")
        if not rouge:
            continue

        entry = {
            "paper_id": session["paper_id"],
            "filename": session["filename"],
            "rouge": rouge,
        }
        session_list.append(entry)

        if rouge.get("led"):
            led_scores.append(rouge["led"])
        if rouge.get("tfidf"):
            tfidf_scores.append(rouge["tfidf"])

    def _avg(scores: list[dict], key: str) -> float:
        vals = [s[key] for s in scores if key in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_led = {
        "rouge1": _avg(led_scores, "rouge1"),
        "rouge2": _avg(led_scores, "rouge2"),
        "rougeL": _avg(led_scores, "rougeL"),
    }
    avg_tfidf = {
        "rouge1": _avg(tfidf_scores, "rouge1"),
        "rouge2": _avg(tfidf_scores, "rouge2"),
        "rougeL": _avg(tfidf_scores, "rougeL"),
    }

    return JSONResponse(content={
        "total_papers_evaluated": len(session_list),
        "average_rouge": {
            "led": avg_led,
            "tfidf": avg_tfidf,
        },
        "sessions": session_list,
    })
