"""Extract text from PDF resumes and identify skills via spaCy."""

import logging
import re
from pathlib import Path

from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)

_nlp = None
_spacy_unavailable = False

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
    """Load spaCy model; return None if unavailable (keyword fallback)."""
    global _nlp, _spacy_unavailable
    if _spacy_unavailable:
        return None
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            _spacy_unavailable = True
            logger.warning(
                "spaCy model en_core_web_sm not available (%s); "
                "falling back to keyword skill matching only",
                exc,
            )
            return None
    return _nlp


def _match_keywords(text_lower: str) -> set:
    found = set()
    for skill_key, skill_name in _SKILLS_LOOKUP.items():
        pattern = r"(?<![a-z0-9])" + re.escape(skill_key) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill_name)
    return found


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF using pdfminer.six."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        text = extract_text(str(path))
    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {exc}") from exc
    return (text or "").strip()


def extract_skills(text):
    """
    Extract skills from text using spaCy (nouns, proper nouns, entities)
    plus matching against a hardcoded skills list.

    If spaCy / en_core_web_sm is missing, falls back to keyword matching only.

    Returns: {"raw_text": str, "skills": list}
    """
    raw_text = text or ""
    found = set()

    if raw_text.strip():
        lower = raw_text.lower()
        found |= _match_keywords(lower)

        nlp = _get_nlp()
        if nlp is not None:
            try:
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
            except Exception as exc:
                logger.warning("spaCy processing failed; using keywords only: %s", exc)

    return {"raw_text": raw_text, "skills": sorted(found, key=str.lower)}
