"""Optional image captioning helpers."""

from __future__ import annotations

from typing import Dict


def generate_caption(image_path: str) -> Dict[str, str]:
    """Return a placeholder caption payload for an image element."""
    return {
        "caption": f"Auto-generated description for {image_path}",
        "model": "placeholder",
    }
