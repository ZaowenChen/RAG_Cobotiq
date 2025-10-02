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

### Design in one minute

- Hybrid retrieval: BM25 (Meilisearch) + ANN (Qdrant) fused with RRF, then MiniLM cross-encoder rerank for procedures.
- Multimodal: store text & image vectors per element (Qdrant named vectors). Return figures when they co-rank.
- Metadata first: role/model/category filters; boosts for priority (OEM) and recency.

### Environment

- Qdrant: http://localhost:6333
- Meilisearch: http://localhost:7700 (API key in `.env`)

### References (docs)

- PyMuPDF, python-pptx, pandas read_excel, Tesseract (macOS), Qdrant named vectors & HNSW, Meilisearch synonyms & ranking, Reciprocal Rank Fusion, MiniLM cross-encoder.
