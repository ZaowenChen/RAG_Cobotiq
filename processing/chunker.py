"""Chunking utilities for breaking normalized elements into retrievable slices."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator

from uuid import uuid4

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 200


def chunk(
    elements: Iterable[Dict[str, object]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Dict[str, object]]:
    """Yield chunked elements ready for embedding."""
    for element in elements:
        content = str(element.get("content_text", ""))
        if len(content) <= max_chars:
            yield element
            continue

        start = 0
        chunk_index = 0
        while start < len(content):
            end = min(len(content), start + max_chars)
            segment = content[start:end]
            chunk_copy = dict(element)
            chunk_copy["id"] = str(uuid4())
            chunk_copy["content_text"] = segment
            chunk_copy["source_uri"] = f"{element.get('source_uri')}#chunk={chunk_index}"
            chunk_copy["chunk_of"] = element["id"]
            yield chunk_copy

            if end == len(content):
                break
            start = max(0, end - overlap)
            chunk_index += 1
