"""Generate MiniLM sentence embeddings for resumes and job descriptions."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Lazy-load and cache the MiniLM embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """Embed a single string; returns a 1-D float32 vector."""
    if not text or not text.strip():
        dim = get_model().get_sentence_embedding_dimension()
        return np.zeros(dim, dtype=np.float32)
    vector = get_model().encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings; returns an (n, dim) float32 array."""
    if not texts:
        dim = get_model().get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    cleaned = [t if t and t.strip() else " " for t in texts]
    vectors = get_model().encode(cleaned, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)
