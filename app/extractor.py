"""Extract text from PDF resumes and identify skills via spaCy."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text
import spacy

# Load spaCy model once; download with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Common technical / soft skills lexicon for rule-based extraction
SKILL_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "nodejs",
    "fastapi",
    "django",
    "flask",
    "sql",
    "postgresql",
    "mongodb",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "machine learning",
    "deep learning",
    "nlp",
    "spacy",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
    "scikit-learn",
    "chromadb",
    "ollama",
    "fastapi",
    "rest api",
    "graphql",
    "ci/cd",
    "agile",
    "scrum",
    "communication",
    "leadership",
    "problem solving",
}


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract raw text content from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    text = extract_text(str(path))
    return text.strip() if text else ""


def extract_skills(text: str) -> list[str]:
    """
    Extract skills from resume text using spaCy NER + keyword matching.

    Returns a sorted, de-duplicated list of skill strings.
    """
    if not text:
        return []

    found: set[str] = set()
    lower = text.lower()

    # Keyword / phrase matching
    for skill in SKILL_KEYWORDS:
        if skill in lower:
            found.add(skill)

    # spaCy entity / noun-chunk hints when model is available
    if nlp is not None:
        doc = nlp(text[:100_000])  # cap for very long docs
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "LANGUAGE", "SKILL"}:
                candidate = ent.text.strip().lower()
                if candidate in SKILL_KEYWORDS or len(candidate) > 2:
                    if candidate in SKILL_KEYWORDS:
                        found.add(candidate)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if phrase in SKILL_KEYWORDS:
                found.add(phrase)

    return sorted(found)


def process_resume(pdf_path: Union[str, Path]) -> dict:
    """Full pipeline: PDF → text → skills."""
    text = extract_text_from_pdf(pdf_path)
    skills = extract_skills(text)
    return {"text": text, "skills": skills, "path": str(pdf_path)}
