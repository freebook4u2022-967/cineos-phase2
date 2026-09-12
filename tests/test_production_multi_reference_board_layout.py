from __future__ import annotations

import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_multi_reference import ProductionReferenceBoardAdapter

Image = pytest.importorskip("PIL.Image")


def _request(
    reference_ids: tuple[str, ...], *, resolution: tuple[int, int]
) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-reference-board-layout",
        scene_id="scene-reference-board-layout",
        camera={"resolution": resolution, "fps": 24, "duration": 1.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(reference_ids),
        deterministic_seed=17,
        renderer_requirements={},
    )
    request.refresh_hash()
    return request


def test_three_reference_board_uses_full_bottom_row_for_third_identity() -> None:
    request = _request(("ref-a", "ref-b", "ref-c"), resolution=(120, 80))
    references = (
        Image.new("RGB", (15, 10), (255, 0, 0)),
        Image.new("RGB", (15, 10), (0, 255, 0)),
        Image.new("RGB", (30, 10), (0, 0, 255)),
    )

    result = ProductionReferenceBoardAdapter()(request, references)

    assert result.image.size == (120, 80)
    assert result.image.getpixel((1, 1)) == (255, 0, 0)
    assert result.image.getpixel((118, 1)) == (0, 255, 0)
    assert result.image.getpixel((1, 78)) == (0, 0, 255)
    assert result.image.getpixel((118, 78)) == (0, 0, 255)
    assert result.consumed_reference_ids == ("ref-a", "ref-b", "ref-c")


def test_three_reference_board_covers_odd_resolution_edges() -> None:
    request = _request(("ref-a", "ref-b", "ref-c"), resolution=(121, 81))
    references = (
        Image.new("RGB", (60, 40), (255, 0, 0)),
        Image.new("RGB", (61, 40), (0, 255, 0)),
        Image.new("RGB", (121, 41), (0, 0, 255)),
    )

    result = ProductionReferenceBoardAdapter()(request, references)

    assert result.image.getpixel((0, 0)) == (255, 0, 0)
    assert result.image.getpixel((120, 0)) == (0, 255, 0)
    assert result.image.getpixel((0, 80)) == (0, 0, 255)
    assert result.image.getpixel((120, 80)) == (0, 0, 255)
