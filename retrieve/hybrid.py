"""Hybrid retrieval orchestrator (BM25 + ANN + RRF)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from meilisearch import Client as MeiliClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings.image_embed import embed_text_for_images
from embeddings.text_embed import embed_text
from retrieve.policy import load_policy

LOGGER = logging.getLogger(__name__)
_COLLECTION_NAME = "robot_elements"
_MEILI_INDEX_NAME = "robot_elements"
_POLICY = load_policy()


_FALLBACK_VALUES = {
    "robot_model": "generic",
    "audience_level": "operator",
}


@lru_cache(maxsize=1)
def _meili_client() -> MeiliClient:
    url = os.getenv("MEILI_URL", "http://localhost:7700")
    master_key = os.getenv("MEILI_MASTER_KEY", "dev_master_key")
    return MeiliClient(url, master_key)


@lru_cache(maxsize=1)
def _qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def _apply_policy_defaults(filters: Dict[str, str]) -> Dict[str, str]:
    defaults = _POLICY.get("prefilter", {})
    merged = dict(filters)
    if defaults.get("default_audience_level") and not merged.get("audience_level"):
        merged["audience_level"] = defaults["default_audience_level"]
    if defaults.get("default_robot_model") and not merged.get("robot_model"):
        merged["robot_model"] = defaults["default_robot_model"]
    return merged


def _build_meili_filter(filters: Dict[str, str]) -> List[List[str]] | None:
    clauses: List[List[str]] = []
    for key, value in filters.items():
        if value in (None, ""):
            continue
        values = [_value for _value in {value, _FALLBACK_VALUES.get(key, value)} if _value]
        clauses.append([f'{key} = "{item}"' for item in values])
    return clauses or None


def _meili_search(query: str, filters: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
    try:
        client = _meili_client()
        response = client.index(_MEILI_INDEX_NAME).search(
            query,
            {
                "limit": limit,
                "filter": _build_meili_filter(filters),
            },
        )
    except Exception as exc:  # pragma: no cover - service might be offline
        LOGGER.error("Meilisearch query failed: %s", exc)
        return []

    hits: List[Dict[str, Any]] = []
    for hit in response.get("hits", []):
        hits.append({
            "id": hit.get("id"),
            "score": float(hit.get("_rankingScore", 0.0)),
            "source": "meilisearch",
            "payload": hit,
        })
    return hits


def _qdrant_filter(filters: Dict[str, str]) -> qmodels.Filter | None:
    conditions: List[qmodels.FieldCondition] = []
    for key, value in filters.items():
        if value in (None, ""):
            continue
        fallback = _FALLBACK_VALUES.get(key)
        if fallback and fallback != value:
            match = qmodels.MatchAny(any=[value, fallback])
        else:
            match = qmodels.MatchValue(value=value)
        conditions.append(qmodels.FieldCondition(key=key, match=match))
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def _qdrant_search(query: str, filters: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
    query_embedding = embed_text([query])
    if not query_embedding:
        return []

    try:
        client = _qdrant_client()
        hits = client.search(
            collection_name=_COLLECTION_NAME,
            query_vector={"name": "text", "vector": query_embedding[0]},
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=_qdrant_filter(filters),
        )
    except Exception as exc:  # pragma: no cover - service might be offline
        LOGGER.error("Qdrant query failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for point in hits:
        payload = dict(point.payload or {})
        payload.setdefault("id", point.id)
        results.append({
            "id": str(point.id),
            "score": float(point.score or 0.0),
            "source": "qdrant",
            "payload": payload,
        })
    return results


def _image_search_params(query: str) -> tuple[bool, float]:
    config = _POLICY.get("image_search", {})
    if not config.get("enabled"):
        return False, 0.0

    primary_weight = float(config.get("rrf_weight", 0.6))
    fallback_weight = float(config.get("rrf_weight_fallback", primary_weight * 0.4))
    triggers = config.get("trigger_keywords") or []

    if not triggers:
        return True, primary_weight

    lowered = query.lower()
    for keyword in triggers:
        if keyword.lower() in lowered:
            return True, primary_weight

    if config.get("fallback_enabled", True):
        return True, fallback_weight
    return False, 0.0


def _qdrant_image_search(query: str, filters: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
    query_embedding = embed_text_for_images([query])
    if not query_embedding:
        return []

    try:
        client = _qdrant_client()
        hits = client.search(
            collection_name=_COLLECTION_NAME,
            query_vector={"name": "image", "vector": query_embedding[0]},
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=_qdrant_filter(filters),
        )
    except Exception as exc:  # pragma: no cover - service might be offline
        LOGGER.error("Qdrant image query failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for point in hits:
        payload = dict(point.payload or {})
        payload.setdefault("id", point.id)
        if not payload.get("image_path") and not payload.get("media_path"):
            continue
        results.append(
            {
                "id": str(point.id),
                "score": float(point.score or 0.0),
                "source": "qdrant_image",
                "payload": payload,
            }
        )
    return results


def _reciprocal_rank_fusion(
    result_sets: Iterable[List[Dict[str, Any]]],
    k: int,
    *,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for index, results in enumerate(result_sets):
        weight = 1.0
        if weights and index < len(weights):
            weight = weights[index]
        if not results:
            continue
        for rank, result in enumerate(results, start=1):
            doc_id = str(result["id"])
            entry = aggregated.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "payload": result["payload"],
                    "scores": defaultdict(float),
                    "rrf_score": 0.0,
                    "sources": set(),
                },
            )
            entry["rrf_score"] += weight * (1.0 / (k + rank))
            entry["scores"][result["source"]] = result["score"]
            entry["sources"].add(result["source"])

    fused = []
    for value in aggregated.values():
        payload = dict(value["payload"])
        payload["scores"] = {k: v for k, v in value["scores"].items()}
        payload["sources"] = sorted(value["sources"])
        payload["rrf_score"] = value["rrf_score"]
        fused.append(payload)

    fused.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return fused


def retrieve(query: str, filters: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return fused retrieval results for the query and filters."""
    merged_filters = _apply_policy_defaults(filters)
    rrf_k = int(_POLICY.get("rrf", {}).get("k", 60))
    limit = max(rrf_k, 50)

    meili_results = _meili_search(query, merged_filters, limit)
    qdrant_results = _qdrant_search(query, merged_filters, limit)

    image_config = _POLICY.get("image_search", {})
    run_images, image_weight = _image_search_params(query)
    image_results: List[Dict[str, Any]] = []
    if run_images:
        image_limit = int(image_config.get("top_k", limit))
        image_results = _qdrant_image_search(query, merged_filters, image_limit)

    if not meili_results and not qdrant_results and not image_results:
        return []

    result_sets: List[List[Dict[str, Any]]] = [meili_results, qdrant_results]
    weights: List[float] = [1.0, 1.0]
    if image_results:
        result_sets.append(image_results)
        weights.append(image_weight)

    fused = _reciprocal_rank_fusion(result_sets, rrf_k, weights=weights)
    return fused
