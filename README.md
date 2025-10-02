# Robot RAG (Multimodal, Role/Model Aware) - Internal Demo

A local-first RAG that unifies PDFs, PPTX, Excel, and images into consistent, retrievable elements with figures returned when useful. Designed for operators and technicians; respects robot model/version and boosts OEM releases.

> **Scope:** Internal demo on a Mac mini. See `docs/LICENSING.md` before any external use.

---

## Quickstart (macOS, Apple Silicon)

### 0) Prereqs

- macOS 13+ on Apple Silicon, Docker, Homebrew, Python 3.11.
- Install OCR engine:
  ```bash
  brew install tesseract
  ```
- Optional: create a local Python environment:
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Configure OpenAI access (for answer generation):
  ```bash
  export OPENAI_API_KEY="<your key>"
  # optional override if you prefer a different model
  export OPENAI_RAG_MODEL="gpt-4o-mini"
  ```

### 1) Bring up local services

```bash
cp .env.example .env   # edit if you like
docker compose up -d   # starts qdrant + meilisearch
```

### 2) Initialize indexes (one-time)

```bash
make init-meili
make init-qdrant
```

This applies filters, synonyms, and ranking rules in Meilisearch, and creates a Qdrant collection with named vectors (text=384, image=512).

### 3) Drop data

Put your files in:

```
data/raw/
  manuals/*.pdf
  slides/*.pptx
  sheets/*.xlsx
  images/*.png|jpg
```

### 4) Implement + run pipelines

Fill in the stub modules under `ingestion/`, `processing/`, `embeddings/`, `index/`, `retrieve/`, then:
- Ingest -> `data/processed/*.jsonl` (elements matching `schemas/element.schema.json`).
- Embed (BGE-small for text; OpenCLIP ViT-B/32 for images).
- Index to Qdrant + Meilisearch.
- Query using the policy in `configs/policy.yaml` (BM25+ANN, RRF, boosts, rerank).

### 5) Evaluate

See `evaluation/EVAL.md` for a tiny harness (nDCG/Recall) using `evaluation/gold.csv`.

---

### Command-Line Cheat Sheet

**Environment setup**
- `python3 -m venv Cobotiq && source Cobotiq/bin/activate` — create/activate the project virtualenv.
- `pip install -r requirements.txt` — install the mandatory dependencies.
- `pip install openpyxl` — enable XLSX ingestion (required for spreadsheets).

**Platform operations**
- `make up` — start Qdrant and Meilisearch.
- `make down` — stop the services.
- `make init-meili` — apply Meilisearch index + settings from `configs/`.
- `make init-qdrant` — create the Qdrant named-vector collection.

**Pipeline stages**
- `make ingest` — run the full ingestion pipeline (covers `ingestion/`, `processing/`, `embeddings/`, `index/`, `retrieve/` policies).
- `python ingest.py --skip-index` — generate `data/processed/*.jsonl` without pushing to Qdrant/Meilisearch.
- `python - <<'PY'` snippets — lightweight way to exercise individual modules, for example:
  ```bash
  python - <<'PY'
  from ingestion.pdf_parser import parse_pdf
  from ingestion.normalize import normalize
  elements = list(parse_pdf("path/to/doc.pdf"))
  normalized = list(normalize(elements))
  print(len(normalized))
  PY
  ```

**Index maintenance**
- `curl -H "Authorization: Bearer $MEILI_MASTER_KEY" -X DELETE http://localhost:7700/indexes/robot_elements` — reset Meilisearch (if you need a clean slate).
- `curl -X DELETE http://localhost:6333/collections/robot_elements` — reset Qdrant before re-running `make init-*`.

---

### Querying Locally

1. With services running and indexes populated, start the API:
   ```bash
   source Cobotiq/bin/activate
   uvicorn app.api:app --reload
   ```
2. Visit `http://localhost:8000/` for the lightweight chat UI (styled after ChatGPT). It supports audience/model selectors, streams answers, and shows citation links inline.
3. Open `http://localhost:8000/docs` if you prefer the Swagger interface for testing the raw `/search` endpoint.
4. Alternatively, use the command line:
   ```bash
   curl "http://localhost:8000/search?query=torque%20m3%20screw"
   ```
   The JSON response includes the fused + reranked hits, a generated answer (when configured), and citation snippets.

---

### Design in one minute

- Hybrid retrieval: BM25 (Meilisearch) + ANN (Qdrant) fused with RRF, then MiniLM cross-encoder rerank for procedures.
- Multimodal: store text & image vectors per element (Qdrant named vectors). Return figures when they co-rank.
- Metadata first: role/model/category filters; boosts for priority (OEM) and recency.

### Environment

- Qdrant: http://localhost:6333
- Meilisearch: http://localhost:7700 (API key in `.env`)

### References (docs)

- PyMuPDF, python-pptx, pandas read_excel, Tesseract (macOS), Qdrant named vectors & HNSW, Meilisearch synonyms & ranking, Reciprocal Rank Fusion, MiniLM cross-encoder.
