"""Image embedding helpers using OpenCLIP ViT-B/32."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"


def _device() -> torch.device:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
    model = model.to(_device())
    model.eval()
    return model, preprocess, tokenizer


def embed_images(image_paths: Iterable[str], *, batch_size: int = 16) -> Dict[str, List[float]]:
    """Return 512-d image embeddings keyed by image path."""
    paths = [str(path) for path in image_paths]
    if not paths:
        return {}

    model, preprocess, _ = _load_model()
    device = _device()
    encoded: Dict[str, List[float]] = {}

    batch: List[torch.Tensor] = []
    batch_paths: List[str] = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = preprocess(image)
        except Exception:
            continue
        batch.append(tensor)
        batch_paths.append(path)
        if len(batch) >= batch_size:
            _encode_batch(batch, batch_paths, model, device, encoded)
            batch = []
            batch_paths = []

    if batch:
        _encode_batch(batch, batch_paths, model, device, encoded)

    return encoded


def _encode_batch(
    tensors: List[torch.Tensor],
    paths: List[str],
    model,
    device: torch.device,
    output: Dict[str, List[float]],
) -> None:
    stack = torch.stack(tensors).to(device)
    with torch.no_grad():
        encoded = model.encode_image(stack)
        encoded = F.normalize(encoded, dim=-1)
    for path, vector in zip(paths, encoded.cpu().tolist()):
        output[path] = vector


def embed_text_for_images(queries: Iterable[str], *, batch_size: int = 32) -> List[List[float]]:
    """Encode text queries into the CLIP image embedding space."""
    texts = [query.strip() for query in queries if query and query.strip()]
    if not texts:
        return []

    model, _, tokenizer = _load_model()
    device = _device()
    encoded_queries: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            vectors = model.encode_text(tokens)
            vectors = F.normalize(vectors, dim=-1)
        encoded_queries.extend(vectors.cpu().tolist())

    return encoded_queries
