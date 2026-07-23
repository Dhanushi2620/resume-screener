"""Cosine similarity matching and resume–JD scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import chromadb
import numpy as np
from chromadb.config import Settings

from app.embedder import embed_text
from app.extractor import extract_skills, process_resume

COLLECTION_NAME = "resume_embeddings"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def skill_overlap_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """Fraction of JD skills present in the resume (0–1)."""
    if not jd_skills:
        return 0.0
    resume_set = {s.lower() for s in resume_skills}
    matched = sum(1 for s in jd_skills if s.lower() in resume_set)
    return matched / len(jd_skills)


def score_match(
    resume_text: str,
    jd_text: str,
    resume_skills: Optional[List[str]] = None,
    jd_skills: Optional[List[str]] = None,
    embedding_weight: float = 0.7,
) -> dict:
    """
    Score a resume against a job description.

    Combines MiniLM cosine similarity with skill-overlap scoring.
    """
    resume_skills = resume_skills if resume_skills is not None else extract_skills(resume_text)
    jd_skills = jd_skills if jd_skills is not None else extract_skills(jd_text)

    resume_vec = embed_text(resume_text)
    jd_vec = embed_text(jd_text)
    sim = cosine_similarity(resume_vec, jd_vec)
    skill_score = skill_overlap_score(resume_skills, jd_skills)

    skill_weight = 1.0 - embedding_weight
    overall = embedding_weight * sim + skill_weight * skill_score

    matched = sorted(set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills))
    missing = sorted(set(s.lower() for s in jd_skills) - set(s.lower() for s in resume_skills))

    return {
        "overall_score": round(overall * 100, 2),
        "embedding_similarity": round(sim * 100, 2),
        "skill_score": round(skill_score * 100, 2),
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def get_chroma_collection():
    """Get or create the ChromaDB collection for resume embeddings."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_resume(resume_id: str, text: str, metadata: Optional[dict] = None) -> None:
    """Store a resume embedding in ChromaDB."""
    collection = get_chroma_collection()
    vector = embed_text(text).tolist()
    collection.upsert(
        ids=[resume_id],
        embeddings=[vector],
        documents=[text[:2000]],
        metadatas=[metadata or {}],
    )


def rank_resumes_against_jd(jd_text: str, resume_paths: List[Union[str, Path]]) -> List[dict]:
    """Score and rank multiple resume PDFs against a job description."""
    results = []
    for path in resume_paths:
        processed = process_resume(path)
        scores = score_match(processed["text"], jd_text, resume_skills=processed["skills"])
        results.append(
            {
                "resume": str(path),
                **scores,
            }
        )
    results.sort(key=lambda r: r["overall_score"], reverse=True)
    return results
