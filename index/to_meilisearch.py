"""Utilities for pushing elements into Meilisearch."""

from __future__ import annotations

import os
from typing import Iterable, Mapping

from meilisearch import Client

_INDEX_NAME = "robot_elements"


def _client() -> Client:
    url = os.getenv("MEILI_URL", "http://localhost:7700")
    master_key = os.getenv("MEILI_MASTER_KEY", "dev_master_key")
    return Client(url, master_key)


def index_meilisearch(elements: Iterable[Mapping[str, object]]) -> None:
    """Upload lexical documents and metadata facets to Meilisearch."""
    client = _client()
    documents = []
    for element in elements:
        document = {
            key: value
            for key, value in element.items()
            if key
            not in {
                "text_vector",
                "image_vector",
                "ocr_text",
                "image_path",
            }
        }
        document["ocr_text"] = element.get("ocr_text")
        documents.append(document)
    if not documents:
        return
    client.index(_INDEX_NAME).add_documents(documents)
