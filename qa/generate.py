"""Answer generation using OpenAI responses API with citation support."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("OPENAI_RAG_MODEL", "gpt-4o-mini")
_MAX_CONTEXT_CHARS = 1200
_TOP_CONTEXT_RESULTS = 6


def _get_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def _client() -> OpenAI:
    return OpenAI(api_key=_get_api_key())


def _format_context(results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []
    for idx, result in enumerate(results[:top_k], start=1):
        content = (
            (result.get("content_text") or "")
            or (result.get("caption") or "")
            or (result.get("ocr_text") or "")
        ).strip()
        if not content:
            continue
        snippet = content[: _MAX_CONTEXT_CHARS]
        citation_id = f"[{idx}]"
        formatted.append(
            {
                "citation_id": str(citation_id),
                "doc_title": str(result.get("doc_title", "")),
                "source_uri": str(result.get("source_uri") or ""),
                "content": str(snippet),
                "element_type": str(result.get("element_type") or ""),
                "caption": str(result.get("caption") or ""),
                "media_path": str(result.get("media_path") or ""),
                "image_mime_type": str(result.get("image_mime_type") or ""),
            }
        )
    return formatted


def _collect_figures(results: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    figures: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if str(result.get("element_type")) != "figure":
            continue
        media_path = result.get("media_path")
        image_sha = result.get("image_sha256")
        dedupe_key = str(image_sha or media_path or result.get("id"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        figure = {
            "id": str(result.get("id")),
            "doc_id": str(result.get("doc_id")),
            "doc_title": str(result.get("doc_title") or ""),
            "source_uri": str(result.get("source_uri") or ""),
            "caption": str(result.get("caption") or result.get("content_text") or ""),
            "media_path": str(media_path or ""),
            "image_path": str(result.get("image_path") or ""),
            "image_mime_type": str(result.get("image_mime_type") or ""),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "image_sha256": image_sha,
            "caption_model": result.get("caption_model"),
            "caption_source": result.get("caption_source"),
        }
        figures.append(figure)
        if limit and len(figures) >= limit:
            break
    return figures


def _build_prompt(query: str, context: List[Dict[str, str]]) -> str:
    context_lines = []
    for entry in context:
        context_lines.append(
            f"{entry['citation_id']} Title: {entry['doc_title']}\n"
            f"Source: {entry['source_uri']}\n"
            f"Excerpt: {entry['content']}\n"
        )
    context_block = "\n".join(context_lines) or "No supporting context available."

    instructions = (
        "You are a robotics support assistant. Answer the user question using the provided context. "
        "Cite supporting passages in square brackets matching their citation id (for example [1]). "
        "If you are unsure or the context is insufficient, say so clearly. Keep the response concise "
        "(2-3 sentences) unless the query requires step-by-step guidance."
    )

    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Instructions: " + instructions
    )
    return prompt


def generate_answer(
    query: str,
    results: List[Dict[str, Any]],
    *,
    top_k: int = _TOP_CONTEXT_RESULTS,
    model: str | None = None,
) -> Dict[str, Any]:
    """Generate an answer and citation payload from retrieval results."""
    if not results:
        return {
            "answer": None,
            "citations": [],
            "figures": [],
            "error": "No retrieval results available to build an answer.",
        }

    api_key = _get_api_key()
    if not api_key:
        LOGGER.warning("OPENAI_API_KEY is not configured; returning retrieval results only.")
        return {
            "answer": None,
            "citations": _format_context(results, top_k),
            "figures": _collect_figures(results, limit=top_k),
            "error": "OPENAI_API_KEY not configured.",
        }

    context = _format_context(results, top_k)
    if not context:
        return {
            "answer": None,
            "citations": [],
            "figures": _collect_figures(results, limit=top_k),
            "error": "Context snippets missing; cannot generate answer.",
        }

    prompt = _build_prompt(query, context)
    selected_model = model or _DEFAULT_MODEL

    try:
        client = _client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You answer strictly from supplied context."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_RAG_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("OPENAI_RAG_MAX_TOKENS", "400")),
        )
        answer_text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - network or API issues
        LOGGER.error("OpenAI answer generation failed: %s", exc)
        return {
            "answer": None,
            "citations": context,
            "figures": _collect_figures(results, limit=top_k),
            "error": f"OpenAI generation failed: {exc}",
        }

    return {
        "answer": answer_text,
        "citations": context,
        "figures": _collect_figures(results, limit=_TOP_CONTEXT_RESULTS),
        "model": selected_model,
    }
