"""Generate natural-language match explanations via Ollama."""

import json
import os
import urllib.error
import urllib.request


def _ollama_endpoint():
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    if base.endswith("/api/generate"):
        return base
    return f"{base}/api/generate"


def _ollama_model():
    return os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an error."""


def explain_match(match_result, job_title):
    """
    Call Ollama to explain fit in 3 sentences.

    Raises OllamaUnavailableError if the server/model is unavailable.
    """
    model = _ollama_model()
    url = _ollama_endpoint()

    prompt = f"""You are a recruiting assistant.
Job title: {job_title}

Match result:
- Similarity score: {match_result.get('similarity_score')}
- Match percentage: {match_result.get('match_percentage')}%
- Matched skills: {', '.join(match_result.get('matched_skills') or []) or 'none'}
- Missing skills: {', '.join(match_result.get('missing_skills') or []) or 'none'}
- Resume skills: {', '.join(match_result.get('resume_skills') or []) or 'none'}
- JD skills: {', '.join(match_result.get('jd_skills') or []) or 'none'}

In exactly 3 sentences, explain why this candidate is or is not a good fit for the role.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {url}: {exc}"
        ) from exc
    except Exception as exc:
        raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    text = (body.get("response") or "").strip()
    if not text:
        raise OllamaUnavailableError("Ollama returned an empty explanation")
    return text
