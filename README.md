# Resume Screener

AI-powered resume screening using MiniLM + spaCy + ChromaDB + Ollama

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. (Optional) Ensure [Ollama](https://ollama.com) is running and pull a model:

```bash
ollama pull llama3.2
```

3. Start the API:

```bash
uvicorn main:app --reload
```

Open interactive docs at `http://127.0.0.1:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/upload/resume` | Upload a resume PDF; extract text/skills and index embeddings |
| `POST` | `/upload/jd` | Upload a job description (text form field or file) |
| `POST` | `/screen` | Rank resumes against JD text (MiniLM + skill overlap) |
| `POST` | `/explain` | Score one resume and generate an Ollama explanation |

## How it works

1. **Extract** — `pdfminer.six` pulls text from resume PDFs; spaCy plus a skills lexicon identifies candidate skills.
2. **Embed** — `sentence-transformers` (all-MiniLM-L6-v2) turns resume and JD text into dense vectors.
3. **Match** — Cosine similarity between embeddings is combined with skill-overlap scoring; ChromaDB stores resume vectors for reuse.
4. **Explain** — Ollama turns the numeric match result into a short natural-language hiring rationale.
