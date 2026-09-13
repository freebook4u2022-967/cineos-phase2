"""Decoded pixel ingestion for CINEOS native neural training.

Neural components must learn from image content, not container/file bytes. Pillow is
kept behind the ``neural`` optional dependency so the base orchestration package
remains lightweight while GPU training gets a deterministic RGB preprocessing
contract.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def pillow_available() -> bool:
    """Return whether Pillow is available for decoded image ingestion."""
    return importlib.util.find_spec("PIL") is not None


def _load_pillow() -> tuple[Any, Any]:
    if not pillow_available():
        raise RuntimeError(
            "Pillow is required for neural image ingestion; install cineos[neural]"
        )
    from PIL import Image, ImageOps

    return Image, ImageOps


@dataclass(frozen=True, slots=True)
class DecodedRGBImage:
    """Deterministically resized RGB pixels normalized to [-1, 1]."""

    width: int
    height: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("decoded image dimensions must be positive")
        expected = self.width * self.height * 3
        if len(self.values) != expected:
            raise ValueError(
                f"decoded RGB image expected {expected} values, got {len(self.values)}"
            )


def decode_rgb_image(path: str | Path, *, image_size: int) -> DecodedRGBImage:
    """Decode, orient and resize an image into normalized RGB training pixels.

    EXIF orientation is applied before RGB conversion. A fixed square resize keeps
    encoder tensor shapes checkpoint-stable and provider-neutral.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    Image, ImageOps = _load_pillow()
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        rgb = oriented.convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
        raw = tuple(channel for pixel in resized.getdata() for channel in pixel)

    normalized = tuple((float(value) / 127.5) - 1.0 for value in raw)
    return DecodedRGBImage(
        width=image_size,
        height=image_size,
        values=normalized,
    )
