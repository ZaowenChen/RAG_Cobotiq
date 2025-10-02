"""Utilities for pushing elements into Qdrant."""

from __future__ import annotations

import os
from typing import Iterable, Mapping

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

_COLLECTION_NAME = "robot_elements"
_TEXT_DIM = 384
_IMAGE_DIM = 512


def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def index_qdrant(elements: Iterable[Mapping[str, object]]) -> None:
    """Write text and image vectors to the configured Qdrant collection."""
    client = _client()
    points: list[PointStruct] = []

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
            continue
        if image_vector and len(vector_payload["image"]) != _IMAGE_DIM:
            continue

        payload = {key: value for key, value in element.items() if key not in {"text_vector", "image_vector"}}
        points.append(PointStruct(id=payload["id"], payload=payload, vector=vector_payload))

    if not points:
        return
    client.upsert(collection_name=_COLLECTION_NAME, points=points)
