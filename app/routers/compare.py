"""
app/routers/compare.py

Route:
    POST /api/compare   — compare 2-5 already-analyzed papers
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.comparison.paper_comparator import PaperComparator

router = APIRouter(prefix="/api")

SESSIONS_DIR = Path("data/sessions")


@router.post("/compare")
async def compare_papers(payload: dict):
    """
    Body: { "paper_ids": ["id1", "id2", ...] }
    Returns: ComparisonReport as JSON
    """
    paper_ids: list[str] = payload.get("paper_ids", [])
    if not (2 <= len(paper_ids) <= 5):
        raise HTTPException(status_code=400,
                            detail="Provide between 2 and 5 paper IDs.")

    sessions = []
    titles = []
    for pid in paper_ids:
        path = SESSIONS_DIR / f"{pid}.json"
        if not path.exists():
            raise HTTPException(status_code=404,
                                detail=f"Paper {pid} not found.")
        session = json.loads(path.read_text(encoding="utf-8"))
        sessions.append(session)
        titles.append(session.get("filename", pid))

    # PaperComparator.compare() expects list of section dicts.
    # We pass the digest dict — map our 6 digest keys back to
    # the canonical section labels the comparator expects.
    _DIM_TO_SECTION = {
        "problem":     "introduction",
        "methodology": "methodology",
        "dataset":     "dataset",
        "results":     "results",
        "limitations": "limitations",
        "conclusion":  "conclusion",
    }
    digests = []
    for session in sessions:
        digest = session.get("digest", {})
        section_dict: dict[str, str] = {}
        for dim, section_key in _DIM_TO_SECTION.items():
            section_dict[section_key] = digest.get(dim, "")
        digests.append(section_dict)

    comparator = PaperComparator()
    report = comparator.compare(digests, titles)

    return JSONResponse(content=report.to_dict())
