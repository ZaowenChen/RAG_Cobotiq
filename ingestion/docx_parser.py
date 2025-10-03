"""DOCX parser that extracts paragraphs, tables, and embedded figures."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from docx import Document  # type: ignore
from docx.document import Document as DocxDocument  # type: ignore
from docx.table import _Cell, Table  # type: ignore
from docx.text.paragraph import Paragraph  # type: ignore
from PIL import Image

from ingestion.utils import doc_id_from_path, reset_doc_media_dir, write_media_bytes

LOGGER = logging.getLogger(__name__)

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
}


def _iter_block_items(parent) -> Iterator[Paragraph | Table]:  # type: ignore[no-untyped-def]
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body  # type: ignore[attr-defined]
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc  # type: ignore[attr-defined]
    else:  # pragma: no cover - defensive programming
        return

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _figure_alt_text(shape) -> str:  # type: ignore[no-untyped-def]
    text = _clean_text(getattr(shape, "alternative_text", "") or "")
    if text:
        return text
    try:
        doc_pr = shape._inline.docPr  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - attribute availability varies
        return ""
    return _clean_text(doc_pr.get("descr") or doc_pr.get("title") or "")


def parse_docx(path: str, *, metadata: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, object]]:
    """Yield raw element dictionaries produced from a DOCX file."""
    meta = metadata or {}
    docx_path = Path(path)
    doc_title = meta.get("doc_title", docx_path.stem.replace("_", " "))
    doc_id = doc_id_from_path(str(docx_path))

    document = Document(str(docx_path))

    reset_doc_media_dir(doc_id)

    textual_order = 0
    table_counter = 0

    for block_index, block in enumerate(_iter_block_items(document), start=1):
        if isinstance(block, Paragraph):
            text = _clean_text(block.text or "")
            if not text:
                continue
            textual_order += 1
            yield {
                "doc_id": doc_id,
                "doc_path": str(docx_path.resolve()),
                "doc_title": doc_title,
                "doc_type": "docx",
                "element_type": "paragraph",
                "order": textual_order,
                "content_text": text,
                "source_uri": f"file://{docx_path.resolve()}#paragraph={block_index}",
                "metadata": meta,
            }
        elif isinstance(block, Table):
            table_counter += 1
            for row_index, row in enumerate(block.rows, start=1):
                cells = [_clean_text(cell.text or "") for cell in row.cells]
                cells = [cell for cell in cells if cell]
                if not cells:
                    continue
                textual_order += 1
                yield {
                    "doc_id": doc_id,
                    "doc_path": str(docx_path.resolve()),
                    "doc_title": doc_title,
                    "doc_type": "docx",
                    "element_type": "table_row",
                    "order": textual_order,
                    "content_text": " | ".join(cells),
                    "source_uri": (
                        f"file://{docx_path.resolve()}#table={table_counter}&row={row_index}"
                    ),
                    "metadata": {**meta, "table_index": table_counter},
                }

    # Embedded figures are processed after textual blocks to keep logic clear.
    figures = list(document.inline_shapes)  # type: ignore[attr-defined]
    if not figures:
        return

    for figure_index, shape in enumerate(figures, start=1):  # type: ignore[no-untyped-def]
        if not hasattr(shape, "image"):
            continue
        try:
            blob = shape.image.blob  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("Failed to extract image blob from %s: %s", docx_path, exc)
            continue

        ext = (getattr(shape.image, "ext", "png") or "png").lower()  # type: ignore[attr-defined]
        filename = f"figure{figure_index:03d}.{ext}"
        stored = write_media_bytes(doc_id, filename, blob)

        width = height = None
        try:
            with Image.open(io.BytesIO(blob)) as img:
                width, height = img.size
        except Exception as exc:  # pragma: no cover - image best effort
            LOGGER.debug("Failed to probe DOCX figure size: %s", exc)

        alt_text = _figure_alt_text(shape)

        figure_payload = {
            "doc_id": doc_id,
            "doc_path": str(docx_path.resolve()),
            "doc_title": doc_title,
            "doc_type": "docx",
            "element_type": "figure",
            "order": figure_index,
            "content_text": alt_text,
            "source_uri": f"file://{docx_path.resolve()}#figure={figure_index}",
            "metadata": {**meta, "alt_text": alt_text},
            "image_mime_type": _MIME_BY_EXT.get(ext, "application/octet-stream"),
            "alt_text": alt_text,
        }
        if width is not None and height is not None:
            figure_payload["image_width"] = int(width)
            figure_payload["image_height"] = int(height)
        figure_payload.update(stored)
        yield figure_payload
