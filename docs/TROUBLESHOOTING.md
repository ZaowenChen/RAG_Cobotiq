# Troubleshooting

**PyTorch / MPS (Apple Silicon)**
- Ensure Python 3.11 and `torch>=2.3`. Check MPS:
  ```python
  import torch
  print(torch.backends.mps.is_available())
  ```
- If models fall back to CPU, set:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

**Tesseract not found**
- Install via Homebrew: `brew install tesseract`.
- Point `pytesseract.pytesseract.tesseract_cmd` to the binary if needed, for example `/opt/homebrew/bin/tesseract`.

**Meilisearch auth errors**
- API key comes from `.env` as `MEILI_MASTER_KEY` (default `dev_master_key`).
- Headers use `X-Meili-API-Key`.

**Qdrant collection shape mismatch**
- Confirm vector sizes: text=384 (BGE-small), image=512 (OpenCLIP ViT-B/32).
- Recreate the collection if you switch models.

**Reranker latency too high**
- Reduce `rerank.top_k` in `configs/policy.yaml` (for example, 24).
- Disable reranking for non-procedural queries via `enable_for_queries_matching`.

---

## Notes for implementation (non-blocking)
- Keep a small synonym list (`configs/synonyms.json`) current with operator vocabulary; load into Meilisearch regularly.
- For OCR on scanned tables, add `requirements-optional.txt` and implement a PaddleOCR fallback only if Tesseract proves insufficient.
- Start with RRF untuned; collect a few manual judgments before tinkering.

---

If you want, I can also draft tiny stubs for the Python modules (argument parsing, minimal glue) so you can run end-to-end with placeholder logic on day one.
