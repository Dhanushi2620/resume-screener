# Resume Screener — Complete Project Documentation

## 1. Problem Statement

Recruiters spend roughly 6–7 seconds reviewing each resume on average, which leaves little room for careful evaluation. Manual screening is inconsistent and can introduce unconscious bias across reviewers and days. Matching candidate skills to a job description objectively becomes harder as applicant volume grows. This project automates resume screening with AI so shortlisting is faster, more consistent, and easier to explain.

## 2. Solution Overview

Resume Screener lets a recruiter upload a resume PDF (or paste resume text) and paste a job description. The system embeds both texts, compares them with cosine similarity, extracts and compares skills, and returns a match percentage plus a short AI explanation. Recruiters get an objective skill gap view and a readable rationale, which helps them shortlist candidates faster without replacing human judgment.

## 3. Tech Stack — with WHY each was chosen

| Component | Technology | Why chosen |
|-----------|------------|------------|
| Embeddings | MiniLM (`all-MiniLM-L6-v2`) | Semantic similarity out of the box; ~87MB; free; runs fully local |
| NLP / skills | spaCy (`en_core_web_sm`) | Fast NER + POS tagging for skill hints; ~12MB model |
| Explanation | Ollama `qwen2.5:3b` | Local LLM; $0 inference cost; turns scores into plain English |
| API | FastAPI | Async Python API; auto OpenAPI docs at `/docs`; low latency |
| Vectors (optional) | ChromaDB | Local vector store with HNSW search for future resume indexing |
| Frontend | Pure HTML/CSS/JS | No framework overhead; single-file UI; easy to demo |

## 4. Architecture

```
Resume PDF  →  pdfminer extracts text
                      ↓
                spaCy extracts skills  (+ keyword skill list match)
                      ↓
                MiniLM embeds full text  →  float vector (384-d)
                      ↓
Job Description → MiniLM embeds → float vector (384-d)
                      ↓
                Cosine similarity computed
                      ↓
                Skills intersection + gap analysis
                      ↓
                Ollama generates explanation (if available)
                      ↓
                JSON response + UI display
```

**Alternate path:** `POST /screen-text` skips PDF extraction and starts from pasted resume text.

## 5. API Endpoints — with request/response examples

Base URL (local): `http://localhost:8010`

### GET /health

**Request**
```http
GET http://localhost:8010/health
```

**Response**
```json
{"status": "ok"}
```

### GET /

Serves `frontend/index.html` (the web UI).

### POST /screen

Screen a resume **PDF** against a job description.

**Request:** `multipart/form-data`
- `resume` — PDF file
- `job_description` **or** `jd_text` — job description string
- `job_title` (optional) — defaults to `"Open Position"`

```bash
curl -X POST http://localhost:8010/screen \
  -F "resume=@resume.pdf" \
  -F "job_description=Looking for a Python developer with FastAPI and Docker"
```

**Response (shape)**
```json
{
  "raw_text_preview": "John Doe - Software Engineer...",
  "match": {
    "similarity_score": 0.77,
    "match_percentage": 77,
    "matched_skills": ["Python", "FastAPI", "Docker"],
    "missing_skills": ["Kubernetes"],
    "resume_skills": ["Python", "FastAPI", "Docker", "Git"],
    "jd_skills": ["Python", "FastAPI", "Docker", "Kubernetes"]
  },
  "explanation": "Strong candidate because..."
}
```

If Ollama is down, scores are still returned with:
```json
{
  "explanation": null,
  "explanation_skipped": true,
  "explanation_error": "Ollama is not reachable..."
}
```

### POST /screen-text

Screen **raw text** (no PDF) — useful for demos and automated tests.

**Request**
```json
{
  "resume_text": "Python developer with FastAPI, PostgreSQL, Docker...",
  "job_description": "Looking for Python backend developer...",
  "job_title": "Backend Engineer"
}
```

**Example**
```bash
curl -X POST http://localhost:8010/screen-text \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer...",
    "job_description": "Looking for Python..."
  }'
```

**Response**
```json
{
  "raw_text_preview": "Python developer...",
  "match": {
    "similarity_score": 0.87,
    "match_percentage": 87,
    "matched_skills": ["Python", "FastAPI", "Docker"],
    "missing_skills": ["Kubernetes"],
    "resume_skills": ["Python", "FastAPI", "Docker", "Git"],
    "jd_skills": ["Python", "FastAPI", "Docker", "Kubernetes"]
  },
  "explanation": "Strong candidate because..."
}
```

**Error example (empty resume)**
```json
{"detail": "resume_text is empty. Provide resume content to screen."}
```
HTTP status: `400`

## 6. How MiniLM Works (simple explanation)

MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) converts a piece of text into a **384-dimensional vector** — a list of 384 floating-point numbers that represent meaning. Texts with similar meanings produce similar vectors and therefore a high cosine similarity. Cosine similarity measures the angle between two vectors: **1.0** means nearly identical direction (very related), **0.0** means orthogonal / unrelated. The resume and job description are embedded separately, then compared. No task-specific training is required for this project — the model is pre-trained on a large text corpus for semantic similarity.

## 7. How spaCy Skill Extraction Works

The pipeline loads `en_core_web_sm`, a compact (~12MB) English NLP model. spaCy tokenizes the text and labels parts of speech (nouns, proper nouns, etc.) and named entities. Those candidates are checked against a hardcoded skills list of 30+ technologies (Python, FastAPI, Docker, Kubernetes, …). Matching is word-boundary aware so `"SQL"` does not falsely match inside `"PostgreSQL"`. If spaCy or the model is missing, the system **falls back to keyword matching only** so screening still works.

## 8. How Scoring Works

```
match_percentage = round(cosine_similarity × 100)
similarity_score = cosine_similarity   # float in [0, 1]
```

**Score interpretation (UI)**

| Range | Meaning | UI color |
|-------|---------|----------|
| **> 70%** | Strong match | Green |
| **40–70%** | Partial match | Orange |
| **< 40%** | Weak match | Red |

**Skills scoring**

```
matched_skills = resume_skills ∩ jd_skills
missing_skills = jd_skills − resume_skills
```

Semantic score (MiniLM) and skill lists are both returned so recruiters can see *overall fit* and *concrete gaps*.

## 9. How Ollama Explanation Works

After scoring, the API sends the match summary to a local Ollama server (`qwen2.5:3b` by default). The prompt asks for **exactly 3 sentences** explaining why the candidate is or is not a good fit, using matched/missing skills and the percentage. The model returns a short, human-readable explanation for the UI. If Ollama is not running or the request fails, the API **still returns scores** and sets `explanation_skipped: true` with an error message — no hard failure. Base URL and model are configurable via `.env` (`OLLAMA_URL`, `OLLAMA_MODEL`).

## 10. Project Structure

```
resume-screener/
  app/
    __init__.py        — package marker
    extractor.py       — PDF text extraction + spaCy / keyword skill extraction
    embedder.py        — MiniLM text → vector conversion
    matcher.py         — cosine similarity + skill gap analysis
    explainer.py       — Ollama explanation generation
  frontend/
    index.html         — complete UI (upload, results, score display)
  samples/
    sample_resume.txt  — test resume for demo
    sample_jd.txt      — test job description for demo
  data/
    resumes/           — local upload storage (gitignored)
    jd/                — local JD storage (gitignored)
  docs/
    PROJECT_DOCUMENTATION.md — this document
  main.py              — FastAPI app + all endpoints
  requirements.txt     — all Python dependencies
  setup.sh             — one-command setup script
  README.md            — quick start guide
  .env.example         — environment variables template
  .gitignore           — Python / env / PDF ignores
  screenshot.png       — UI screenshot for README demo
```

## 11. Setup Instructions

```bash
git clone https://github.com/Dhanushi2620/resume-screener
cd resume-screener

# Option A — setup script
chmod +x setup.sh
./setup.sh

# Option B — manual
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional: copy env template and start Ollama
cp .env.example .env
ollama pull qwen2.5:3b
ollama serve   # if not already running

# Run API + UI
uvicorn main:app --reload --port 8010
```

Open **http://localhost:8010**  
Interactive API docs: **http://localhost:8010/docs**

**Quick text test**
```bash
curl -X POST http://localhost:8010/screen-text \
  -H "Content-Type: application/json" \
  -d "{
    \"resume_text\": \"$(cat samples/sample_resume.txt)\",
    \"job_description\": \"$(cat samples/sample_jd.txt)\"
  }" | python3 -m json.tool
```

## 12. Sample Output

**Input resume:** Python developer with FastAPI, PostgreSQL, Docker, Redis  
**Input JD:** Python, FastAPI, PostgreSQL, Docker, Kubernetes

**Output (illustrative / representative)**
```json
{
  "match_percentage": 80,
  "matched_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
  "missing_skills": ["Kubernetes"],
  "explanation": "This candidate is a strong match with 80% skill alignment. They have all core required skills. Only missing Kubernetes which could be learned quickly given their Docker experience."
}
```

**Real run with `samples/`** (observed during project testing): ~77% match; matched Docker, FastAPI, Git, Machine Learning, PostgreSQL, Python, Redis, REST API; missing AWS and Kubernetes; Ollama returned a 3-sentence rationale.

## 13. Key Technical Decisions — Interview Questions

**Q: Why MiniLM over other embedding models?**  
**A:** It is ~87MB, free, and runs locally with no API key. It is trained for semantic similarity, uses 384 dimensions (enough for resume↔JD matching), and embeds a document in tens of milliseconds on CPU — a good accuracy/cost/latency trade-off vs larger models.

**Q: Why spaCy over BERT NER?**  
**A:** `en_core_web_sm` is ~12MB versus hundreds of MB for BERT-style NER. It is fast enough for interactive screening, easy to install, and combined with a curated skills list gives reliable tech-skill hits without a GPU.

**Q: Why cosine similarity instead of only keyword overlap?**  
**A:** Keywords miss paraphrases (“built REST services” vs “REST API”). Cosine similarity on MiniLM embeddings captures semantic overlap. Skill intersection then explains *what* matched or is missing in recruiter-friendly terms.

**Q: Why Ollama locally instead of a cloud LLM API?**  
**A:** Zero per-request cost, no resume data leaves the machine, and offline demos work. If Ollama is down, scoring still succeeds — explanation is optional, not a single point of failure.

**Q: Why FastAPI + a single HTML file instead of a full SPA framework?**  
**A:** FastAPI gives typed endpoints and `/docs` for free. A pure HTML/CSS/JS UI keeps the repo small, installs nothing on the frontend, and is enough for a clear upload → results demo suitable for resume submission.

## 14. Error Handling & Resilience

| Failure | Behavior |
|---------|----------|
| Empty resume / JD | HTTP `400` with a clear `detail` message |
| Invalid / empty / image-only PDF | HTTP `400` explaining extraction failure |
| spaCy model missing | Keyword skill matching only; screening continues |
| Ollama not running | Scores returned; `explanation` is `null` + `explanation_skipped` |
| Unexpected server errors | HTTP `500` with logged exception |

## 15. Environment Variables

From `.env.example`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model used for explanations |
| `PORT` | `8010` | Suggested local app port |

## 16. Repository

- **GitHub:** https://github.com/Dhanushi2620/resume-screener  
- **Visibility:** Public  
- **Local UI:** http://localhost:8010  

---

*This document describes the Resume Screener as implemented for portfolio / resume submission: local MiniLM embeddings, spaCy + keyword skills, FastAPI API, optional Ollama explanations, and a lightweight web UI.*
