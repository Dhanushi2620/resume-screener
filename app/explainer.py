"""Generate natural-language match explanations via Ollama."""

from __future__ import annotations

from typing import Any, Dict

import ollama


DEFAULT_MODEL = "llama3.2"


def build_prompt(match_result: Dict[str, Any], jd_excerpt: str = "", resume_excerpt: str = "") -> str:
    """Build an explanation prompt from matcher output."""
    return f"""You are a recruiting assistant. Explain how well this resume matches the job description.

Overall score: {match_result.get('overall_score', 'N/A')}%
Embedding similarity: {match_result.get('embedding_similarity', 'N/A')}%
Skill score: {match_result.get('skill_score', 'N/A')}%
Matched skills: {', '.join(match_result.get('matched_skills') or []) or 'none'}
Missing skills: {', '.join(match_result.get('missing_skills') or []) or 'none'}

Job description excerpt:
{jd_excerpt[:1500] or '(not provided)'}

Resume excerpt:
{resume_excerpt[:1500] or '(not provided)'}

Write 2–4 concise paragraphs covering:
1. Overall fit
2. Strengths (matched skills / experience signals)
3. Gaps (missing skills)
4. A short recommendation (strong / moderate / weak fit)
"""


def explain_match(
    match_result: Dict[str, Any],
    jd_text: str = "",
    resume_text: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Ask Ollama to explain a resume–JD match result.

    Requires a running Ollama server with the chosen model pulled.
    """
    prompt = build_prompt(match_result, jd_excerpt=jd_text, resume_excerpt=resume_text)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    message = response.get("message") or {}
    return (message.get("content") or "").strip()
