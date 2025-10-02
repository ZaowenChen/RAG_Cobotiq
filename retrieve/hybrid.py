"""Hybrid retrieval orchestrator (BM25 + ANN + RRF)."""

from typing import Dict, Iterable, List

def retrieve(query: str, filters: Dict[str, str]) -> List[Dict[str, object]]:
    """Return fused retrieval results for the query and filters."""
    raise NotImplementedError("Combine Meilisearch BM25, Qdrant ANN, and RRF fusion.")
