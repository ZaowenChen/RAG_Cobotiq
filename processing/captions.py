"""Automatic and manual caption generation utilities for figures."""

from __future__ import annotations

import io
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("ROBOT_RAG_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
_MANUAL_DIR = Path(os.getenv("ROBOT_RAG_CAPTION_DIR", "data/captions"))

try:  # pragma: no cover - optional heavy dependency
    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore
except ImportError:  # pragma: no cover - allow runtime without transformers
    BlipForConditionalGeneration = None  # type: ignore
    BlipProcessor = None  # type: ignore


def _device() -> torch.device:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model():  # pragma: no cover - heavy to execute in tests
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        raise RuntimeError("transformers is not installed; cannot auto-caption images")
    LOGGER.info("Loading caption model %s", _DEFAULT_MODEL)
    processor = BlipProcessor.from_pretrained(_DEFAULT_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(_DEFAULT_MODEL)
    device = _device()
    model = model.to(device)
    model.eval()
    return processor, model, device


@lru_cache(maxsize=1)
def _manual_captions() -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    if not _MANUAL_DIR.exists():
        try:
            _MANUAL_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - best effort to prepare directory
            LOGGER.debug("Could not ensure manual caption directory %s: %s", _MANUAL_DIR, exc)
        return overrides

    for json_path in sorted(_MANUAL_DIR.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load manual captions from %s: %s", json_path, exc)
            continue

        source = json_path.name
        if isinstance(payload, dict):
            items = payload.items()
        elif isinstance(payload, list):
            items = []
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                key = (
                    str(entry.get("image_sha256") or entry.get("sha256") or entry.get("key") or "")
                )
                if not key:
                    continue
                value = entry.get("caption")
                if not isinstance(value, str):
                    continue
                metadata = dict(entry)
                metadata["caption"] = value
                metadata.setdefault("caption_model", "manual")
                metadata.setdefault("caption_source", "manual")
                metadata.setdefault("source_file", source)
                overrides[key.lower()] = metadata
            continue
        else:
            continue

        for key, value in items:
            if not isinstance(value, (str, dict)):
                continue
            caption = value if isinstance(value, str) else value.get("caption")
            if not isinstance(caption, str):
                continue
            metadata = value if isinstance(value, dict) else {}
            metadata = dict(metadata)
            metadata["caption"] = caption
            metadata.setdefault("caption_model", metadata.get("model", "manual"))
            metadata.setdefault("caption_source", "manual")
            metadata.setdefault("source_file", source)
            overrides[str(key).lower()] = metadata

    return overrides


def _lookup_manual(image_path: str, image_sha256: Optional[str]) -> Optional[Dict[str, object]]:
    overrides = _manual_captions()
    keys = []
    if image_sha256:
        keys.append(image_sha256.lower())
    path_obj = Path(image_path)
    keys.extend(
        [
            str(path_obj).lower(),
            path_obj.name.lower(),
            path_obj.stem.lower(),
        ]
    )
    for key in keys:
        if key in overrides:
            entry = dict(overrides[key])
            entry.setdefault("caption_model", entry.get("model", "manual"))
            entry.setdefault("caption_source", "manual")
            entry.setdefault("caption_confidence", entry.get("confidence"))
            return entry
    return None


def _auto_caption(image_path: str) -> Optional[Dict[str, object]]:
    try:
        processor, model, device = _load_model()
    except RuntimeError as exc:  # pragma: no cover - transformers missing
        LOGGER.debug("Auto caption unavailable: %s", exc)
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Failed to prepare image for captioning: %s", exc)
        return None

    with torch.no_grad():
        output = model.generate(**inputs, max_length=64)
    caption = processor.decode(
        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip()
    if not caption:
        return None
    return {
        "caption": caption,
        "caption_model": _DEFAULT_MODEL,
        "caption_source": "auto",
    }


def generate_caption(
    image_path: str,
    *,
    image_sha256: Optional[str] = None,
    fallback_text: Optional[str] = None,
) -> Dict[str, object]:
    """Return the best available caption payload for an image element."""
    override = _lookup_manual(image_path, image_sha256)
    if override:
        return override

    auto = _auto_caption(image_path)
    if auto:
        return auto

    if fallback_text:
        return {
            "caption": fallback_text,
            "caption_model": "fallback",
            "caption_source": "fallback",
        }

    return {
        "caption": f"Image located at {Path(image_path).name}",
        "caption_model": "placeholder",
        "caption_source": "placeholder",
    }
