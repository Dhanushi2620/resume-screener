"""FastAPI application for AI-powered resume screening."""

import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.explainer import explain_match
from app.extractor import extract_skills, extract_text_from_pdf
from app.matcher import match_resume_to_jd

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

app = FastAPI(
    title="Resume Screener",
    description="AI-powered resume screening using MiniLM + spaCy + Ollama",
    version="1.0.0",
)


class ScreenTextRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=1)
    job_title: str = "Open Position"


def _run_screen_pipeline(resume_text: str, job_description: str, job_title: str = "Open Position"):
    """Shared matching + explanation pipeline for text or PDF flows."""
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is empty")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is empty")

    extracted = extract_skills(resume_text)
    match_result = match_resume_to_jd(resume_text, job_description)

    try:
        explanation = explain_match(match_result, job_title)
    except Exception as exc:
        explanation = f"Explanation unavailable (is Ollama running with qwen2.5:3b?): {exc}"

    return {
        "raw_text_preview": extracted["raw_text"][:500],
        "match": match_result,
        "explanation": explanation,
    }


@app.get("/")
def root():
    """Serve the single-page frontend UI."""
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/screen-text")
def screen_text(payload: ScreenTextRequest):
    """
    Screen raw resume text against a job description (no PDF required).
    Useful for testing without uploading a file.
    """
    return _run_screen_pipeline(
        payload.resume_text,
        payload.job_description,
        payload.job_title,
    )


@app.post("/screen")
async def screen(
    resume: UploadFile = File(...),
    jd_text: Optional[str] = Form(default=None),
    job_description: Optional[str] = Form(default=None),
    job_title: str = Form(default="Open Position"),
):
    """
    Screen a resume PDF against a job description.

    Accepts either form field: jd_text or job_description.
    Pipeline: extractor → embedder → matcher → explainer
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

    jd = (jd_text or job_description or "").strip()
    if not jd:
        raise HTTPException(
            status_code=400,
            detail="Provide job_description or jd_text",
        )

    suffix = Path(resume.filename).suffix or ".pdf"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await resume.read()
            tmp.write(content)
            tmp_path = tmp.name

        resume_text = extract_text_from_pdf(tmp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        return _run_screen_pipeline(resume_text, jd, job_title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
