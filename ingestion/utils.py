"""Shared ingestion utilities for document identifiers and media paths."""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
import hashlib
from pathlib import Path

_DOC_NAMESPACE = uuid.UUID("7c0856c3-3c5d-4c3f-857b-60c5fb8a8545")

_PREPARED_DOCS: set[str] = set()


def doc_id_from_path(path: str) -> str:
    """Return the deterministic UUID for a document based on its resolved path."""
    return str(uuid.uuid5(_DOC_NAMESPACE, str(Path(path).resolve())))


@lru_cache(maxsize=1)
def media_root() -> Path:
    """Return the filesystem root where derived media assets are stored."""
    root = Path(os.getenv("ROBOT_RAG_MEDIA_ROOT", "data/processed/media")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def reset_doc_media_dir(doc_id: str) -> Path:
    """Ensure the media directory for a document is empty and ready for new assets."""
    root = media_root() / doc_id
    if doc_id in _PREPARED_DOCS:
        # Directory already prepared within this process; avoid redundant wipes.
        return root

    if root.exists():
        for child in root.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                _remove_tree(child)
    root.mkdir(parents=True, exist_ok=True)
    _PREPARED_DOCS.add(doc_id)
    return root


def ensure_doc_media_dir(doc_id: str) -> Path:
    """Return the media directory for a document without clearing existing content."""
    root = media_root() / doc_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_media_bytes(doc_id: str, filename: str, data: bytes) -> dict[str, object]:
    """Persist image bytes for a document and return metadata about the stored asset."""
    directory = ensure_doc_media_dir(doc_id)
    target = directory / filename
    target.write_bytes(data)

    rel_path = target.relative_to(media_root())
    digest = hashlib.sha256(data).hexdigest()

    return {
        "image_path": str(target),
        "media_path": str(rel_path.as_posix()),
        "image_sha256": digest,
    }


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()
