# Resume Screener

AI-powered resume screening using MiniLM + spaCy + ChromaDB + Ollama

## How it works

1. Upload resume PDF or paste resume text
2. Paste job description
3. MiniLM embeds both texts into vectors
4. Cosine similarity computes match score
5. spaCy extracts and compares skills
6. Ollama explains the match in plain English

## Stack

- MiniLM (sentence-transformers) — text embeddings
- spaCy — skill extraction
- ChromaDB — vector storage
- Ollama qwen2.5:3b — match explanation
- FastAPI — REST API
- Pure HTML/CSS/JS — frontend UI

## Run locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload --port 8010
```

Open http://localhost:8010

## Demo

![Resume Screener UI](screenshot.png)

### Sample Output

```json
{
  "match_percentage": 85,
  "matched_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST API", "Machine Learning", "Redis"],
  "missing_skills": ["Kubernetes"],
  "explanation": "Strong candidate — matches 7 of 8 required skills..."
}
```

## API Endpoints

- `POST /screen` — upload PDF resume + job description
- `POST /screen-text` — paste resume text + job description
- `GET /health` — health check
