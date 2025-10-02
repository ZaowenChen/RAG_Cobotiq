"""XLSX parser that converts rows into structured elements."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_xlsx(path: str, *, metadata: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, object]]:
    """Yield raw element dictionaries produced from spreadsheet rows."""
    meta = metadata or {}
    xlsx_path = Path(path)
    doc_title = meta.get("doc_title", xlsx_path.stem.replace("_", " "))

    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", xlsx_path, exc)
        return []

    elements = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue
        for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
            values = []
            for column_name, value in row.items():
                if pd.isna(value):
                    continue
                values.append(f"{column_name}: {value}")
            if not values:
                continue
            content = "\n".join(values)
            elements.append(
                {
                    "doc_path": str(xlsx_path.resolve()),
                    "doc_title": doc_title,
                    "doc_type": "xlsx",
                    "element_type": "table_row",
                    "order": row_index,
                    "content_text": content,
                    "source_uri": f"file://{xlsx_path.resolve()}#sheet={sheet_name}&row={row_index}",
                    "metadata": {**meta, "sheet_name": sheet_name},
                }
            )
    return elements
