from pathlib import Path

import pytest
from PIL import Image

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.reference_board import (
    REFERENCE_BOARD_ADAPTER_ID,
    REFERENCE_BOARD_ADAPTER_VERSION,
    compose_reference_board,
)


def _request(*references: str, resolution=(8, 4)) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={"resolution": resolution, "fps": 24, "duration": 1.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(references),
        deterministic_seed=7,
        renderer_requirements={},
        metadata={"prompt": "two characters interact"},
    )
    request.refresh_hash()
    return request


def _solid(rgb: tuple[int, int, int], size=(6, 6)) -> Image.Image:
    return Image.new("RGB", size, rgb)


def test_reference_board_consumes_every_reference_in_request_order():
    request = _request("hero", "partner")

    result = compose_reference_board(
        request,
        (_solid((255, 0, 0)), _solid((0, 255, 0))),
    )

    assert result.consumed_reference_ids == ("hero", "partner")
    assert result.adapter_id == REFERENCE_BOARD_ADAPTER_ID
    assert result.adapter_version == REFERENCE_BOARD_ADAPTER_VERSION
    assert result.image.size == (8, 4)
    assert result.image.getpixel((1, 1)) == (255, 0, 0)
    assert result.image.getpixel((6, 1)) == (0, 255, 0)


def test_reference_board_preserves_full_portrait_identity_evidence():
    request = _request("hero", "partner")
    portrait = Image.new("RGB", (2, 6), (0, 255, 0))
    portrait.putpixel((0, 0), (255, 0, 0))
    portrait.putpixel((1, 0), (255, 0, 0))
    portrait.putpixel((0, 5), (0, 0, 255))
    portrait.putpixel((1, 5), (0, 0, 255))

    result = compose_reference_board(request, (portrait, _solid((255, 255, 255))))
    left_tile = result.image.crop((0, 0, 4, 4))
    pixels = set(left_tile.getdata())

    assert any(red > green and red > blue for red, green, blue in pixels)
    assert any(blue > red and blue > green for red, green, blue in pixels)


def test_reference_board_rejects_duplicate_identity_references():
    request = _request("hero", "hero")

    with pytest.raises(DiffusersVideoError, match="unique approved reference ids"):
        compose_reference_board(
            request,
            (_solid((255, 0, 0)), _solid((255, 0, 0))),
        )


def test_reference_board_rejects_same_image_under_distinct_identity_ids():
    request = _request("hero", "partner")
    portrait = _solid((120, 80, 40))
    same_pixels_different_mode = portrait.convert("RGBA")

    with pytest.raises(DiffusersVideoError, match="distinct reference image content"):
        compose_reference_board(
            request,
            (portrait, same_pixels_different_mode),
        )


def test_reference_board_uses_stable_two_by_two_layout_for_three_characters():
    request = _request("hero", "partner", "villain")

    result = compose_reference_board(
        request,
        (
            _solid((255, 0, 0)),
            _solid((0, 255, 0)),
            _solid((0, 0, 255)),
        ),
    )

    assert result.image.getpixel((1, 0)) == (255, 0, 0)
    assert result.image.getpixel((6, 0)) == (0, 255, 0)
    assert result.image.getpixel((1, 3)) == (0, 0, 255)
    assert result.image.getpixel((6, 3)) == (0, 0, 0)


def test_reference_board_rejects_partial_or_excess_reference_sets():
    request = _request("hero", "partner")
    with pytest.raises(DiffusersVideoError, match="different number"):
        compose_reference_board(request, (_solid((255, 0, 0)),))

    request = _request("a", "b", "c", "d", "e")
    with pytest.raises(DiffusersVideoError, match="at most 4"):
        compose_reference_board(request, tuple(_solid((0, 0, 0)) for _ in range(5)))


def test_reference_board_rejects_non_image_reference():
    request = _request("hero", "partner")

    with pytest.raises(DiffusersVideoError, match="Pillow Image"):
        compose_reference_board(request, (_solid((255, 0, 0)), "not-an-image"))


def test_pinned_wan_profile_supplies_reference_board_by_default(tmp_path: Path):
    renderer = WAN22_TI2V_5B_PROFILE.renderer(output_dir=tmp_path)

    assert renderer.multi_reference_adapter is compose_reference_board


def test_pinned_wan_profile_allows_stronger_audited_adapter_override(tmp_path: Path):
    def custom_adapter(request, references):
        return compose_reference_board(request, references)

    renderer = WAN22_TI2V_5B_PROFILE.renderer(
        output_dir=tmp_path,
        multi_reference_adapter=custom_adapter,
    )

    assert renderer.multi_reference_adapter is custom_adapter
