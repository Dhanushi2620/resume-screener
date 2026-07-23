"""FastAPI application for AI-powered resume screening."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.explainer import OllamaUnavailableError, explain_match
from app.extractor import extract_skills, extract_text_from_pdf
from app.matcher import match_resume_to_jd

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

app = FastAPI(
    title="Resume Screener",
    description="AI-powered resume screening using MiniLM + spaCy + Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenTextRequest(BaseModel):
    resume_text: str = Field(default="")
    job_description: str = Field(default="")
    job_title: str = "Open Position"


def _run_screen_pipeline(resume_text: str, job_description: str, job_title: str = "Open Position"):
    """Shared matching + explanation pipeline for text or PDF flows."""
    if not resume_text or not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="resume_text is empty. Provide resume content to screen.",
        )
    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="job_description is empty. Provide a job description to compare against.",
        )

    try:
        extracted = extract_skills(resume_text)
        match_result = match_resume_to_jd(resume_text, job_description)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Matching pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score resume against job description: {exc}",
        ) from exc

    explanation = None
    explanation_error = None
    try:
        explanation = explain_match(match_result, job_title)
    except OllamaUnavailableError as exc:
        explanation_error = str(exc)
        logger.warning("Ollama unavailable; returning score without explanation: %s", exc)
    except Exception as exc:
        explanation_error = f"Explanation failed: {exc}"
        logger.warning("Explanation failed; returning score without explanation: %s", exc)

    result = {
        "raw_text_preview": extracted["raw_text"][:500],
        "match": match_result,
        "explanation": explanation,
    }
    if explanation is None:
        result["explanation_skipped"] = True
        result["explanation_error"] = explanation_error or (
            "Ollama is not running. Scores were computed without an AI explanation."
        )
    return result


@app.get("/")
def root():
    """Serve the single-page frontend UI."""
    try:
        if not FRONTEND_INDEX.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to serve frontend")
        raise HTTPException(status_code=500, detail=f"Failed to serve UI: {exc}") from exc


@app.get("/health")
def health():
    try:
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})


@app.post("/screen-text")
def screen_text(payload: ScreenTextRequest):
    """
    Screen raw resume text against a job description (no PDF required).
    Useful for testing without uploading a file.
    """
    try:
        return _run_screen_pipeline(
            payload.resume_text,
            payload.job_description,
            payload.job_title,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("screen-text failed")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/screen")
async def screen(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(default=None),
    jd_text: Optional[str] = Form(default=None),
    job_title: str = Form(default="Open Position"),
):
    """
    Screen a resume PDF against a job description.

    Accepts form fields: resume (PDF) + job_description (or jd_text).
    Pipeline: extractor → embedder → matcher → explainer
    """
    tmp_path = None
    try:
        logger.info("Received file: %s", resume.filename)
        logger.info("Content-Type: %s", resume.content_type)
        logger.info(
            "JD length: %s",
            len((job_description or jd_text or "")),
        )

        if not resume.filename or not resume.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF resumes are supported")

        jd = (job_description or jd_text or "").strip()
        if not jd:
            raise HTTPException(
                status_code=400,
                detail="Provide job_description or jd_text",
            )

        suffix = Path(resume.filename).suffix or ".pdf"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await resume.read()
                logger.info("File size: %s bytes", len(content))
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded PDF file is empty")
                tmp.write(content)
                tmp_path = tmp.name
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save uploaded PDF: {exc}",
            ) from exc

        try:
            resume_text = extract_text_from_pdf(tmp_path)
            logger.info("Extracted resume text length: %s", len(resume_text))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"PDF extraction failed. Ensure the file is a valid, text-based PDF. ({exc})",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"PDF extraction failed. Ensure the file is a valid, text-based PDF. ({exc})",
            ) from exc

        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The file may be image-only or corrupted.",
            )

        return _run_screen_pipeline(resume_text, jd, job_title)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("screen failed")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
