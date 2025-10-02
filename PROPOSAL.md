# Multimodal RAG for Robotic Instructions (Mac-mini-first, Internal Demo)

## Goals & Constraints
- Unify PDFs (text + figures), PPT/PPTX (slides + speaker notes), Excel (tables), and standalone images into retrievable, citation-ready elements (text + metadata, optional image vectors).
- Local-first stack on a Mac mini (no external services): parsing, OCR, embeddings, lexical + vector search, fusion, rerank.
- Role/model aware: operator vs technician; S50 vs V40 vs generic.
- Return figures alongside answers when relevant.
- Keep OEM releases & latest versions on top via metadata boosts.

## Architecture (left -> right)
Ingest -> Normalize -> Chunk -> Embed (text + image) -> Index (vector + keyword) -> Retrieve (BM25 + ANN, RRF fuse) -> Priority/Freshness boosts -> Rerank (cross-encoder) -> Answer + citations + figures.

**Why this shape?**  
Parsing coverage (PyMuPDF, python-pptx, pandas), OCR via Tesseract (PaddleOCR optional), robust hybrid retrieval with RRF, cross-encoder rerank for procedures, and multi-vector records (text+image) using Qdrant named vectors.

## Data Model (Element-level payload)
Each retrievable element (paragraph, step, table row, figure, slide note) carries:
```json
{
  "id": "uuid-or-stable-id",
  "doc_id": "doc-uuid",
  "doc_title": "Beetle S50 Lidar Calibration v1.4",
  "doc_type": "pdf",
  "element_type": "step",
  "content_text": "Tighten the M3 screw to 0.5 N*m...",
  "robot_model": "S50",
  "software_version": "AIO 1.4.2",
  "hardware_rev": "H1.2",
  "audience_level": "technician",
  "category": "sop",
  "priority": "high",
  "effective_date": "2025-08-20",
  "replaces": "doc:abcd",
  "source_uri": "file:///.../manual.pdf#page=12&figure=2",
  "text_vector": "...",
  "image_vector": "...",
  "ocr_text": "caption/labels if any"
}
```

Why: Filters (role, model, category), boosts (priority, recency), and exact figure return via source_uri. Qdrant named vectors keep text & image together.

## Retrieval Policy (defaults)
1. Pre-filter by audience_level and robot_model when provided (defaults: operator, generic).
2. Parallel searches
   - BM25 (Meilisearch) over content_text (synonyms on), facet filters applied.
   - ANN (Qdrant) over text_vector, top_k=100.
   - (Optional) ANN over image_vector with CLIP text encoder when query hints "diagram/figure/wiring/port". top_k=40.
3. RRF fuse with k=60.
4. Boosts: priority (+0.15 for high) and recency (sigmoid over days since effective_date, max +0.10).
5. Rerank top-40 with cross-encoder/ms-marco-MiniLM-L6-v2; keep best 6-8 chunks.
6. Assemble answer with co-occurring figures; cite doc_title (page/slide/figure).

## Mac-Mini defaults
- Text embeddings: BAAI/bge-small-en-v1.5 (384-dim).
- Image embeddings: OpenCLIP ViT-B/32 (512-dim).
- Vector DB: Qdrant (HNSW; named vectors text=384, image=512).
- Keyword FTS: Meilisearch (synonyms + custom ranking).
- Reranker: MiniLM-L6 cross-encoder (top_k=40). Toggleable via configs/policy.yaml.

## Evaluation (offline)
- Maintain evaluation/gold.csv with 30-50 queries covering operator/technician, S50/V40, SOP/PNC/hotfix.
- Track nDCG@10 and Recall@20 across variants: BM25-only, ANN-only, RRF, RRF+rerank.

## Licensing (internal demo)
- PyMuPDF is AGPL/commercial; safe for internal demo. If you later expose externally, either acquire a license or swap to pdfplumber + pdfminer.six. See docs/LICENSING.md.

## References (docs)
- PyMuPDF docs (PDF blocks, images, tables), python-pptx docs (speaker notes), pandas read_excel, Tesseract (macOS), Qdrant named vectors & HNSW, Meilisearch synonyms & ranking, RRF, MiniLM cross-encoder.
