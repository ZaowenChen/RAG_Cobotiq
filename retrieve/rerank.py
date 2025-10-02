"""Cross-encoder reranking helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import torch
from sentence_transformers import CrossEncoder

from retrieve.policy import load_policy

_POLICY = load_policy()


def _device() -> str:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=_device())


def rerank(query: str, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return reranked results using a cross-encoder."""
    config = _POLICY.get("rerank", {})
    if not config.get("enabled", True):
        return candidates
    if not candidates:
        return []

    model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")
    top_k = int(config.get("top_k", 40))
    to_score = candidates[: top_k]

    model = _load_model(model_name)
    pairs = [(query, str(item.get("content_text", ""))) for item in to_score]
    scores = model.predict(pairs, convert_to_numpy=True)

    reranked = []
    for item, score in zip(to_score, scores):
        enriched = dict(item)
        enriched["rerank_score"] = float(score)
        reranked.append(enriched)

    reranked.sort(key=lambda entry: entry.get("rerank_score", 0.0), reverse=True)
    reranked.extend(candidates[top_k:])
    return reranked
