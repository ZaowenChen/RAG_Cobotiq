"""Normalization helpers that map raw parser output to schema-compliant elements."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

DEFAULTS: Dict[str, Optional[str]] = {
    "robot_model": "generic",
    "audience_level": "operator",
    "category": None,
    "software_version": None,
    "hardware_rev": None,
    "priority": "normal",
    "effective_date": None,
}

_DOC_NAMESPACE = uuid.UUID("7c0856c3-3c5d-4c3f-857b-60c5fb8a8545")


def _doc_id_from_path(path: str) -> str:
    return str(uuid.uuid5(_DOC_NAMESPACE, str(Path(path).resolve())))


def normalize(
    elements: Iterable[Dict[str, object]],
    *,
    defaults: Optional[Dict[str, Optional[str]]] = None,
) -> Iterator[Dict[str, object]]:
    """Normalize parser outputs to match schemas/element.schema.json."""
    merged_defaults = {**DEFAULTS, **(defaults or {})}

    for element in elements:
        doc_path = str(element.get("doc_path", ""))
        doc_id = element.get("doc_id") or _doc_id_from_path(doc_path)
        metadata = {
            key: element.get(key) or element.get("metadata", {}).get(key) or merged_defaults.get(key)
            for key in (
                "robot_model",
                "software_version",
                "hardware_rev",
                "audience_level",
                "category",
                "priority",
                "effective_date",
                "replaces",
            )
        }

        normalized = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "doc_title": element.get("doc_title") or Path(doc_path).stem,
            "doc_type": element.get("doc_type"),
            "element_type": element.get("element_type"),
            "content_text": element.get("content_text", ""),
            "robot_model": metadata.get("robot_model"),
            "software_version": metadata.get("software_version"),
            "hardware_rev": metadata.get("hardware_rev"),
            "audience_level": metadata.get("audience_level"),
            "category": metadata.get("category"),
            "priority": metadata.get("priority"),
            "effective_date": metadata.get("effective_date"),
            "replaces": metadata.get("replaces"),
            "source_uri": element.get("source_uri"),
            "text_vector": None,
            "image_vector": None,
            "ocr_text": element.get("metadata", {}).get("ocr_text"),
        }

        if element.get("image_path"):
            normalized["image_path"] = element.get("image_path")

        if element.get("order") is not None:
            normalized["order"] = element.get("order")

        yield normalized
