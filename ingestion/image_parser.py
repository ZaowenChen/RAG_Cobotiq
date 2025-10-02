"""Image parser that extracts OCR text and metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional during tests
    pytesseract = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def parse_image(path: str, *, metadata: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, object]]:
    """Yield a raw figure element from a standalone image."""
    meta = metadata or {}
    image_path = Path(path)
    doc_title = meta.get("doc_title", image_path.stem.replace("_", " "))

    ocr_text = ""
    if pytesseract is not None:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ocr_text = pytesseract.image_to_string(img).strip()
        except Exception as exc:  # pragma: no cover - best effort OCR
            LOGGER.debug("Image OCR failed for %s: %s", image_path, exc)

    yield {
        "doc_path": str(image_path.resolve()),
        "doc_title": doc_title,
        "doc_type": "image",
        "element_type": "figure",
        "order": 0,
        "content_text": ocr_text,
        "source_uri": f"file://{image_path.resolve()}",
        "metadata": {**meta, "ocr_text": ocr_text},
        "image_path": str(image_path.resolve()),
    }
