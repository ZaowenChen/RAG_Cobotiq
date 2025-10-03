"""Utilities for pushing elements into Qdrant."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import PointStruct

_COLLECTION_NAME = "robot_elements"
_TEXT_DIM = 384
_IMAGE_DIM = 512
_DEFAULT_BATCH_SIZE = 128

LOGGER = logging.getLogger(__name__)


def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", "60"))
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def _batch_size() -> int:
    try:
        value = int(os.getenv("QDRANT_UPSERT_BATCH", str(_DEFAULT_BATCH_SIZE)))
        return max(16, value)
    except ValueError:
        return _DEFAULT_BATCH_SIZE


def _flush(client: QdrantClient, batch: list[PointStruct]) -> None:
    if not batch:
        return
    try:
        client.upsert(collection_name=_COLLECTION_NAME, points=batch, wait=True)
    except ResponseHandlingException as exc:
        LOGGER.error("Qdrant upsert failed for batch of %d points: %s", len(batch), exc)
        raise


def index_qdrant(elements: Iterable[Mapping[str, object]]) -> None:
    """Write text and image vectors to the configured Qdrant collection."""
    client = _client()
    batch_limit = _batch_size()
    batch: list[PointStruct] = []

    for element in elements:
        text_vector = element.get("text_vector")
        image_vector = element.get("image_vector")
        if not text_vector:
            continue

        vector_payload = {
            "text": list(text_vector),
            "image": list(image_vector) if image_vector else [0.0] * _IMAGE_DIM,
        }

        if len(vector_payload["text"]) != _TEXT_DIM:
            LOGGER.debug("Skipping element %s due to unexpected text vector length", element.get("id"))
            continue
        if image_vector and len(vector_payload["image"]) != _IMAGE_DIM:
            LOGGER.debug("Skipping element %s due to unexpected image vector length", element.get("id"))
            continue

        payload = {key: value for key, value in element.items() if key not in {"text_vector", "image_vector"}}
        batch.append(PointStruct(id=payload["id"], payload=payload, vector=vector_payload))

        if len(batch) >= batch_limit:
            _flush(client, batch)
            batch = []

    _flush(client, batch)
