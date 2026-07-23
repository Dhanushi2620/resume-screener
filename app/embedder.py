"""Generate MiniLM sentence embeddings for resumes and job descriptions."""

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_text(text):
    """Embed a single string; returns a float list vector."""
    model = _get_model()
    if not text or not str(text).strip():
        dim = model.get_sentence_embedding_dimension()
        return [0.0] * dim
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts):
    """Embed a batch of strings; returns a list of float vectors."""
    model = _get_model()
    if not texts:
        return []
    cleaned = [t if t and str(t).strip() else " " for t in texts]
    vectors = model.encode(cleaned, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
