"""Cross-encoder reranking helpers."""

from typing import Dict, List

def rerank(query: str, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return reranked results using a cross-encoder."""
    raise NotImplementedError("Load cross-encoder/ms-marco-MiniLM-L6-v2 via sentence-transformers.")
