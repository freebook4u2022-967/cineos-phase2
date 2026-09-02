"""CINEOS-owned deterministic multi-reference conditioning board.

The external video foundation still receives a single image-conditioning input.
This module composes every approved identity reference into one auditable board so
multi-character shots do not silently drop identities. It is orchestration only:
no pretrained weights or external-model capability are represented as CINEOS-native.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .diffusers_video import DiffusersVideoError
from .native_request import NativeShotRequest
from .production_diffusers import MultiReferenceConditioningResult

REFERENCE_BOARD_ADAPTER_ID = "cineos.reference-board"
REFERENCE_BOARD_ADAPTER_VERSION = "1.2"
_MAX_REFERENCES = 4


def compose_reference_board(
    request: NativeShotRequest,
    references: Sequence[Any],
) -> MultiReferenceConditioningResult:
    """Compose 2-4 approved reference images into the foundation's one image slot.

    The board is deterministic for a fixed input sequence and target resolution.
    References retain request order, matching the production boundary's exact
    consumption attestation. Each approved reference id must be unique: repeating
    one identity would consume board capacity while falsely presenting the request
    as broader multi-reference conditioning. Each source is aspect-preserving
    letterboxed inside its tile rather than center-cropped: production identity
    conditioning must not discard face, hair, wardrobe, body-shape, or silhouette
    evidence merely to fill a tile. Pillow is imported lazily so base CINEOS installs
    do not acquire an image dependency unless neural/video execution is requested.
    """

    expected = tuple(request.approved_reference_ids)
    if len(references) != len(expected):
        raise DiffusersVideoError(
            "reference-board adapter received a different number of images than "
            "approved reference ids"
        )
    if len(expected) != len(set(expected)):
        raise DiffusersVideoError(
            "reference-board adapter requires unique approved reference ids"
        )
    if len(references) < 2:
        raise DiffusersVideoError(
            "reference-board adapter is reserved for multi-reference conditioning"
        )
    if len(references) > _MAX_REFERENCES:
        raise DiffusersVideoError(
            f"reference-board adapter supports at most {_MAX_REFERENCES} references"
        )

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - exercised without video extras
        raise DiffusersVideoError(
            "reference-board adapter requires Pillow; install cineos[video]"
        ) from exc

    width, height = request.camera.get("resolution", (0, 0))
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise DiffusersVideoError(
            "reference-board adapter requires a positive integer camera resolution"
        )

    count = len(references)
    columns = 2 if count > 1 else 1
    rows = 1 if count <= 2 else 2
    tile_width = width // columns
    tile_height = height // rows
    if tile_width <= 0 or tile_height <= 0:
        raise DiffusersVideoError("reference-board target resolution is too small")

    board = Image.new("RGB", (width, height), (0, 0, 0))
    for index, reference in enumerate(references):
        if not isinstance(reference, Image.Image):
            raise DiffusersVideoError(
                "reference-board adapter requires Pillow Image reference inputs"
            )
        source = reference.convert("RGB")
        tile = ImageOps.contain(
            source,
            (tile_width, tile_height),
            method=Image.Resampling.LANCZOS,
        )
        x0 = (index % columns) * tile_width
        y0 = (index // columns) * tile_height
        x = x0 + (tile_width - tile.width) // 2
        y = y0 + (tile_height - tile.height) // 2
        board.paste(tile, (x, y))

    return MultiReferenceConditioningResult(
        image=board,
        consumed_reference_ids=expected,
        adapter_id=REFERENCE_BOARD_ADAPTER_ID,
        adapter_version=REFERENCE_BOARD_ADAPTER_VERSION,
    )


__all__ = [
    "REFERENCE_BOARD_ADAPTER_ID",
    "REFERENCE_BOARD_ADAPTER_VERSION",
    "compose_reference_board",
]
