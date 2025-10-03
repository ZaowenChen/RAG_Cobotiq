"""End-to-end ingestion command for Robot RAG."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from embeddings.image_embed import embed_images
from embeddings.text_embed import embed_text
from index.to_meilisearch import index_meilisearch
from index.to_qdrant import index_qdrant
from ingestion.docx_parser import parse_docx
from ingestion.image_parser import parse_image
from ingestion.normalize import normalize
from ingestion.pdf_parser import parse_pdf
from ingestion.pptx_parser import parse_pptx
from ingestion.xlsx_parser import parse_xlsx
from processing.chunker import chunk
from processing.captions import generate_caption
from processing.dedupe import drop_duplicates

LOGGER = logging.getLogger("robot_rag.ingest")

_SUPPORTED_SUFFIXES = {
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
    ".xlsx": parse_xlsx,
    ".xls": parse_xlsx,
    ".docx": parse_docx,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
}


def _iter_raw_files(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in _SUPPORTED_SUFFIXES:
            yield path


def _collect_elements(paths: Iterable[Path], metadata: Dict[str, str]) -> List[Dict[str, object]]:
    elements: List[Dict[str, object]] = []
    for path in paths:
        parser = _SUPPORTED_SUFFIXES.get(path.suffix.lower())
        if not parser:
            continue
        LOGGER.info("Parsing %s", path.name)
        parsed = list(parser(str(path), metadata=metadata))
        elements.extend(parsed)
    return elements


def _write_jsonl(processed_dir: Path, elements: Iterable[Dict[str, object]]) -> None:
    by_doc: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for element in elements:
        by_doc[element["doc_id"]].append(element)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, items in by_doc.items():
        output_path = processed_dir / f"{doc_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        LOGGER.info("Wrote %s", output_path)


def _apply_text_embeddings(elements: List[Dict[str, object]]) -> None:
    texts = [element.get("content_text", "") for element in elements]
    vectors = embed_text(texts)
    for element, vector in zip(elements, vectors):
        element["text_vector"] = vector


def _apply_image_embeddings(elements: List[Dict[str, object]]) -> None:
    image_elements = []
    for element in elements:
        path = element.get("image_path")
        if not path:
            continue
        resolved = Path(str(path))
        if not resolved.exists():
            LOGGER.warning("Skipping image embedding; file missing: %s", resolved)
            continue
        image_elements.append((element, str(resolved)))

    if not image_elements:
        return

    vectors = embed_images(path for _, path in image_elements)
    captions_cache: Dict[str, Dict[str, object]] = {}

    for element, path in image_elements:
        if path in vectors:
            element["image_vector"] = vectors[path]

        if path not in captions_cache:
            caption_payload = generate_caption(
                path,
                image_sha256=str(element.get("image_sha256") or "") or None,
                fallback_text=str(element.get("content_text") or "") or None,
            )
            captions_cache[path] = caption_payload

        caption_payload = captions_cache[path]
        for key in ("caption", "caption_model", "caption_source", "caption_confidence"):
            value = caption_payload.get(key)
            if value is not None:
                element[key] = value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot RAG ingestion pipeline")
    parser.add_argument("--raw-dir", default="data/raw", type=Path, help="Directory with raw documents")
    parser.add_argument(
        "--processed-dir",
        default=Path("data/processed"),
        type=Path,
        help="Output directory for normalized JSONL",
    )
    parser.add_argument("--robot-model", default="generic", help="Default robot model metadata")
    parser.add_argument("--audience-level", default="operator", help="Default audience level metadata")
    parser.add_argument("--category", default=None, help="Default category metadata")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Character overlap between chunks")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing into Qdrant/Meilisearch")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    raw_dir = args.raw_dir
    if not raw_dir.exists():
        LOGGER.error("Raw directory %s does not exist", raw_dir)
        raise SystemExit(1)

    metadata_defaults = {
        "robot_model": args.robot_model,
        "audience_level": args.audience_level,
        "category": args.category,
    }

    raw_files = list(_iter_raw_files(raw_dir))
    if not raw_files:
        LOGGER.warning("No supported documents found in %s", raw_dir)
        return

    raw_elements = _collect_elements(raw_files, metadata_defaults)
    normalized = list(normalize(raw_elements, defaults=metadata_defaults))
    chunked = list(chunk(normalized, max_chars=args.chunk_size, overlap=args.chunk_overlap))
    deduped = list(drop_duplicates(chunked))

    _apply_text_embeddings(deduped)
    _apply_image_embeddings(deduped)

    _write_jsonl(args.processed_dir, deduped)

    if args.skip_index:
        LOGGER.info("Skipping index push per --skip-index")
        return

    LOGGER.info("Indexing %d elements into Meilisearch", len(deduped))
    index_meilisearch(deduped)
    LOGGER.info("Indexing %d elements into Qdrant", len(deduped))
    index_qdrant(deduped)


if __name__ == "__main__":
    main()
