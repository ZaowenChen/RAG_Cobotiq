"""Text embedding helpers using BGE-small."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import torch
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _device() -> str:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    model = SentenceTransformer(_MODEL_NAME, device=_device())
    return model


def embed_text(chunks: Iterable[str], *, batch_size: int = 64) -> List[List[float]]:
    """Return 384-d text embeddings for the provided chunks."""
    texts = [text.strip() for text in chunks]
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
