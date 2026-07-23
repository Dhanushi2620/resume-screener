"""Generate natural-language match explanations via Ollama."""

import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def explain_match(match_result, job_title):
    """
    Call Ollama (qwen2.5:3b) to explain fit in 3 sentences.

    Returns explanation string.
    """
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
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))

    return (body.get("response") or "").strip()
