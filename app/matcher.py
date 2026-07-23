"""Cosine similarity matching and resume–JD scoring."""

import numpy as np

from app.embedder import embed_text
from app.extractor import extract_skills


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors; returns float score."""
    a = np.asarray(vec1, dtype=np.float64).ravel()
    b = np.asarray(vec2, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_resume_to_jd(resume_text, jd_text):
    """
    Embed resume and JD, score similarity, and compare extracted skills.

    Returns match details including matched/missing skills.
    """
    resume_vec = embed_text(resume_text)
    jd_vec = embed_text(jd_text)
    similarity = cosine_similarity(resume_vec, jd_vec)

    resume_skills = extract_skills(resume_text)["skills"]
    jd_skills = extract_skills(jd_text)["skills"]

    resume_set = {s.lower(): s for s in resume_skills}
    jd_set = {s.lower(): s for s in jd_skills}

    matched_keys = sorted(set(resume_set) & set(jd_set))
    missing_keys = sorted(set(jd_set) - set(resume_set))

    matched_skills = [jd_set[k] for k in matched_keys]
    missing_skills = [jd_set[k] for k in missing_keys]

    return {
        "similarity_score": round(similarity, 4),
        "match_percentage": int(round(similarity * 100)),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
    }
