"""Duplicate detection utilities for processed elements."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Iterator


def drop_duplicates(elements: Iterable[Dict[str, object]]) -> Iterator[Dict[str, object]]:
    """Return elements with near-duplicates removed using exact hashing."""
    seen = set()
    for element in elements:
        key = str(element.get("content_text", ""))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        yield element
