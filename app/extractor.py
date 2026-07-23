"""Extract text from PDF resumes and identify skills via spaCy."""

import re
from pathlib import Path

from pdfminer.high_level import extract_text

_nlp = None

SKILLS_LIST = [
    "Python",
    "JavaScript",
    "TypeScript",
    "React",
    "Node.js",
    "FastAPI",
    "SQL",
    "PostgreSQL",
    "MongoDB",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "scikit-learn",
    "pandas",
    "NumPy",
    "Git",
    "REST API",
    "GraphQL",
    "Redis",
    "ChromaDB",
    "LangChain",
    "Ollama",
    "HuggingFace",
]

_SKILLS_LOOKUP = {s.lower(): s for s in SKILLS_LIST}


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF using pdfminer.six."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    text = extract_text(str(path))
    return (text or "").strip()


def extract_skills(text):
    """
    Extract skills from text using spaCy (nouns, proper nouns, entities)
    plus matching against a hardcoded skills list.

    Returns: {"raw_text": str, "skills": list}
    """
    raw_text = text or ""
    found = set()

    if raw_text.strip():
        lower = raw_text.lower()

        # Match hardcoded skills against full text (word-boundary aware
        # so "SQL" does not match inside "PostgreSQL")
        for skill_key, skill_name in _SKILLS_LOOKUP.items():
            pattern = r"(?<![a-z0-9])" + re.escape(skill_key) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                found.add(skill_name)

        # spaCy: nouns, proper nouns, named entities
        nlp = _get_nlp()
        doc = nlp(raw_text[:100_000])
        candidates = []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                candidates.append(token.text.strip())
        for ent in doc.ents:
            candidates.append(ent.text.strip())

        for candidate in candidates:
            key = candidate.lower()
            if key in _SKILLS_LOOKUP:
                found.add(_SKILLS_LOOKUP[key])

    return {"raw_text": raw_text, "skills": sorted(found, key=str.lower)}
