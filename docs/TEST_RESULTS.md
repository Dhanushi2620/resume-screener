# End-to-End Test Results

Verified on local run (`uvicorn` port **8010**, Ollama `qwen2.5:3b`).

| Case | Expected | Result | Explanation |
|------|----------|--------|-------------|
| Strong match (Python backend ↔ Python backend) | > 70% | **81%** | Yes |
| Partial match (Frontend ↔ Python backend) | 40–70% | **41%** | Yes |
| Sample files (`samples/`) | valid JSON | **77%** | Yes |
| PDF upload (`test_resume.pdf`) | valid JSON | **53%** | Yes |

Also verified:
- `GET /health` → `{"status":"ok"}`
- `GET /` → UI HTML 200
- `GET /docs` → OpenAPI docs 200

All three primary screen-text cases returned `match_percentage` and a non-empty Ollama `explanation`.
