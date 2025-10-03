"""PDF parser that extracts basic text and figure elements."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import fitz  # type: ignore
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency during tests
    pytesseract = None  # type: ignore

from ingestion.utils import doc_id_from_path, reset_doc_media_dir, write_media_bytes

LOGGER = logging.getLogger(__name__)

PNG_MIME = "image/png"


def _page_paragraphs(page: fitz.Page) -> Iterator[str]:
    text = page.get_text("text") or ""
    for block in text.split("\n\n"):
        cleaned = block.strip()
        if cleaned:
            yield cleaned


def _extract_ocr_text(pix: fitz.Pixmap) -> Optional[str]:
    if pytesseract is None:
        return None
    try:
        image_bytes = pix.tobytes("png")
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            text = pytesseract.image_to_string(img)
            return text.strip() or None
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.debug("OCR failed: %s", exc)
        return None


def parse_pdf(path: str, *, metadata: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, object]]:
    """Yield raw element dictionaries produced from a PDF file."""
    meta = metadata or {}
    pdf_path = Path(path)
    doc_title = meta.get("doc_title", pdf_path.stem.replace("_", " "))
    doc_id = doc_id_from_path(str(pdf_path))

    reset_doc_media_dir(doc_id)

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            source_uri = f"file://{pdf_path.resolve()}#page={page_index + 1}"
            for paragraph_index, paragraph in enumerate(_page_paragraphs(page), start=1):
                yield {
                    "doc_id": doc_id,
                    "doc_path": str(pdf_path.resolve()),
                    "doc_title": doc_title,
                    "doc_type": "pdf",
                    "element_type": "paragraph",
                    "order": paragraph_index,
                    "content_text": paragraph,
                    "source_uri": f"{source_uri}&segment={paragraph_index}",
                    "metadata": {**meta, "page_number": page_index + 1},
                }

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                try:
                    xref = image_info[0]
                    pixmap = fitz.Pixmap(document, xref)
                except Exception as exc:  # pragma: no cover - extraction best effort
                    LOGGER.debug("Pixmap extraction failed: %s", exc)
                    continue

                try:
                    png_bytes = pixmap.tobytes("png")
                except Exception as exc:  # pragma: no cover - best effort
                    LOGGER.debug("Pixmap conversion failed: %s", exc)
                    pixmap = None
                    continue

                ocr_text = _extract_ocr_text(pixmap)
                stored = write_media_bytes(
                    doc_id,
                    f"page{page_index + 1:04d}_figure{image_index:03d}.png",
                    png_bytes,
                )

                element = {
                    "doc_id": doc_id,
                    "doc_path": str(pdf_path.resolve()),
                    "doc_title": doc_title,
                    "doc_type": "pdf",
                    "element_type": "figure",
                    "order": image_index,
                    "content_text": ocr_text or "",
                    "source_uri": f"{source_uri}&figure={image_index}",
                    "metadata": {
                        **meta,
                        "ocr_text": ocr_text,
                        "page_number": page_index + 1,
                    },
                    "image_width": int(getattr(pixmap, "width", 0)),
                    "image_height": int(getattr(pixmap, "height", 0)),
                    "image_mime_type": PNG_MIME,
                }
                element.update(stored)
                yield element
                pixmap = None  # help GC
