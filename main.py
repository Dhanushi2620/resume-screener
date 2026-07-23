"""FastAPI application for AI-powered resume screening."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.explainer import explain_match
from app.extractor import extract_skills, extract_text_from_pdf, process_resume
from app.matcher import index_resume, rank_resumes_against_jd, score_match

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
RESUMES_DIR = BASE_DIR / "data" / "resumes"
JD_DIR = BASE_DIR / "data" / "jd"

RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Resume Screener",
    description="AI-powered resume screening using MiniLM + spaCy + ChromaDB + Ollama",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Resume Screener",
        "docs": "/docs",
        "endpoints": [
            "POST /upload/resume",
            "POST /upload/jd",
            "POST /screen",
            "POST /explain",
            "GET /health",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume PDF and extract text + skills."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

    resume_id = f"{uuid.uuid4().hex}_{file.filename}"
    dest = RESUMES_DIR / resume_id
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = process_resume(dest)
        index_resume(resume_id, result["text"], metadata={"filename": file.filename})
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {exc}") from exc

    return {
        "resume_id": resume_id,
        "filename": file.filename,
        "skills": result["skills"],
        "text_preview": result["text"][:500],
    }


@app.post("/upload/jd")
async def upload_jd(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """Upload a job description as plain text or a PDF/TXT file."""
    jd_id = uuid.uuid4().hex
    content = (text or "").strip()

    if file is not None and file.filename:
        dest = JD_DIR / f"{jd_id}_{file.filename}"
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if file.filename.lower().endswith(".pdf"):
            content = extract_text_from_pdf(dest)
        else:
            content = dest.read_text(encoding="utf-8", errors="ignore")
    else:
        if not content:
            raise HTTPException(status_code=400, detail="Provide JD text or a file")
        dest = JD_DIR / f"{jd_id}.txt"
        dest.write_text(content, encoding="utf-8")

    skills = extract_skills(content)
    return {
        "jd_id": jd_id,
        "path": str(dest),
        "skills": skills,
        "text_preview": content[:500],
    }


@app.post("/screen")
async def screen(
    jd_text: str = Form(...),
    resume_ids: Optional[str] = Form(default=None),
):
    """
    Screen uploaded resumes against a job description.

    Optionally pass comma-separated resume_ids; otherwise all PDFs in data/resumes are used.
    """
    if resume_ids:
        paths = []
        for rid in [r.strip() for r in resume_ids.split(",") if r.strip()]:
            path = RESUMES_DIR / rid
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Resume not found: {rid}")
            paths.append(path)
    else:
        paths = sorted(RESUMES_DIR.glob("*.pdf"))

    if not paths:
        raise HTTPException(status_code=400, detail="No resumes available to screen")

    rankings = rank_resumes_against_jd(jd_text, paths)
    return {"count": len(rankings), "results": rankings}


@app.post("/explain")
async def explain(
    jd_text: str = Form(...),
    resume_id: str = Form(...),
):
    """Score one resume against a JD and return an Ollama-generated explanation."""
    path = RESUMES_DIR / resume_id
    if not path.exists():
        # also allow bare filename match
        matches = list(RESUMES_DIR.glob(f"*{resume_id}*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Resume not found: {resume_id}")
        path = matches[0]

    processed = process_resume(path)
    scores = score_match(processed["text"], jd_text, resume_skills=processed["skills"])
    try:
        explanation = explain_match(scores, jd_text=jd_text, resume_text=processed["text"])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama explanation failed (is Ollama running?): {exc}",
        ) from exc

    return {"resume": str(path), "scores": scores, "explanation": explanation}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
