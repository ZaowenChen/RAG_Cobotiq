"""Image parser that extracts OCR text and metadata."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import mimetypes

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional during tests
    pytesseract = None  # type: ignore

LOGGER = logging.getLogger(__name__)

from ingestion.utils import doc_id_from_path, write_media_bytes, reset_doc_media_dir


def _mime_type(path: Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def parse_image(path: str, *, metadata: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, object]]:
    """Yield a raw figure element from a standalone image."""
    meta = metadata or {}
    image_path = Path(path)
    doc_title = meta.get("doc_title", image_path.stem.replace("_", " "))
    doc_id = doc_id_from_path(str(image_path))

    ocr_text = ""
    if pytesseract is not None:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ocr_text = pytesseract.image_to_string(img).strip()
        except Exception as exc:  # pragma: no cover - best effort OCR
            LOGGER.debug("Image OCR failed for %s: %s", image_path, exc)

    try:
        data = image_path.read_bytes()
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.error("Failed to read image %s: %s", image_path, exc)
        return []

    reset_doc_media_dir(doc_id)
    stored = write_media_bytes(doc_id, image_path.name, data)
    mime = _mime_type(image_path)

    image_width = image_height = None
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_width, image_height = probe.size
    except Exception:  # pragma: no cover - best effort metadata
        image_width = image_height = None

    yield {
        "doc_id": doc_id,
        "doc_path": str(image_path.resolve()),
        "doc_title": doc_title,
        "doc_type": "image",
        "element_type": "figure",
        "order": 0,
        "content_text": ocr_text,
        "source_uri": f"file://{image_path.resolve()}",
        "metadata": {**meta, "ocr_text": ocr_text},
        "image_path": stored["image_path"],
        "media_path": stored["media_path"],
        "image_sha256": stored["image_sha256"],
        "image_mime_type": mime,
        "image_width": int(image_width) if image_width is not None else None,
        "image_height": int(image_height) if image_height is not None else None,
    }
