import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_multi_reference import (
    MULTI_REFERENCE_RUNTIME_SCHEMA,
    ProductionMultiReferenceError,
    ProductionReferenceBoardAdapter,
    bind_production_multi_reference_runtime,
)


def _runtime():
    return {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "default",
        "production_default_runtime": True,
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": False,
            "pipeline_factory": False,
            "video_exporter": False,
        },
    }


def _request(reference_ids, *, resolution=(1280, 704), characters=None):
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={"resolution": resolution, "fps": 24, "duration": 1.0},
        characters=list(characters or []),
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(reference_ids),
        deterministic_seed=7,
        renderer_requirements={},
    )
    request.refresh_hash()
    return request


def test_first_party_multi_reference_adapter_remains_production_evidence():
    adapter = ProductionReferenceBoardAdapter()

    bound = bind_production_multi_reference_runtime(_runtime(), adapter)

    assert adapter.adapter_version == "0.1.4"
    assert bound["production_default_runtime"] is True
    assert bound["runtime_mode"] == "default"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is False
    assert bound["multi_reference_conditioning"]["schema"] == (
        MULTI_REFERENCE_RUNTIME_SCHEMA
    )
    assert bound["multi_reference_conditioning"]["adapter_id"] == adapter.adapter_id
    assert bound["multi_reference_conditioning"]["adapter_version"] == (
        adapter.adapter_version
    )
    assert bound["multi_reference_conditioning"]["layout_policy"] == (
        "equal_character_area_then_reference_area"
    )
    assert (
        bound["multi_reference_conditioning"]["prevents_reference_count_area_bias"]
        is True
    )
    assert (
        bound["multi_reference_conditioning"]["requires_unique_reference_ids"] is True
    )
    assert (
        bound["multi_reference_conditioning"]["attests_character_reference_ownership"]
        is True
    )


def test_arbitrary_multi_reference_adapter_downgrades_runtime_evidence():
    bound = bind_production_multi_reference_runtime(_runtime(), lambda *_args: None)

    assert bound["production_default_runtime"] is False
    assert bound["runtime_mode"] == "injected"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is True
    assert "multi_reference_conditioning" not in bound


def test_absent_multi_reference_adapter_preserves_default_runtime():
    bound = bind_production_multi_reference_runtime(_runtime(), None)

    assert bound["production_default_runtime"] is True
    assert bound["runtime_mode"] == "default"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is False


def test_first_party_adapter_attests_exact_multi_character_reference_ownership():
    image_module = pytest.importorskip("PIL.Image")
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("hero-front", "partner-front", "hero-profile"),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-front", "hero-profile"],
            },
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["partner-front"],
            },
        ),
    )
    images = (
        image_module.new("RGB", (8, 8), (255, 0, 0)),
        image_module.new("RGB", (8, 8), (0, 255, 0)),
        image_module.new("RGB", (8, 8), (0, 0, 255)),
    )

    result = adapter(request, images)

    assert result.consumed_reference_ids == (
        "hero-front",
        "partner-front",
        "hero-profile",
    )
    assert result.consumed_character_reference_ids == (
        ("hero", ("hero-front", "hero-profile")),
        ("partner", ("partner-front",)),
    )


def test_multi_character_board_balances_area_before_splitting_extra_views():
    image_module = pytest.importorskip("PIL.Image")
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("hero-front", "partner-front", "hero-profile"),
        resolution=(120, 80),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-front", "hero-profile"],
            },
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["partner-front"],
            },
        ),
    )
    images = (
        image_module.new("RGB", (30, 80), (255, 0, 0)),
        image_module.new("RGB", (60, 80), (0, 255, 0)),
        image_module.new("RGB", (30, 80), (0, 0, 255)),
    )

    result = adapter(request, images)

    # Hero owns the left half and its two views subdivide only that half. Partner
    # owns the entire right half even though it has one approved view. Reference
    # count therefore cannot silently give one character more conditioning area.
    assert result.image.getpixel((10, 40)) == (255, 0, 0)
    assert result.image.getpixel((40, 40)) == (0, 0, 255)
    assert result.image.getpixel((90, 40)) == (0, 255, 0)
    assert result.consumed_reference_ids == (
        "hero-front",
        "partner-front",
        "hero-profile",
    )


def test_first_party_adapter_does_not_infer_character_ownership_from_reference_order():
    image_module = pytest.importorskip("PIL.Image")
    adapter = ProductionReferenceBoardAdapter()
    request = _request(("hero-front", "partner-front"))
    images = (
        image_module.new("RGB", (8, 8), (255, 0, 0)),
        image_module.new("RGB", (8, 8), (0, 255, 0)),
    )

    result = adapter(request, images)

    assert result.consumed_character_reference_ids is None


def test_duplicate_reference_ids_fail_before_image_processing():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(("hero-front", "hero-front"))

    class ExplodingImage:
        def convert(self, _mode):
            raise AssertionError("duplicate ids must fail before image processing")

        def resize(self, _size, _resample):
            raise AssertionError("duplicate ids must fail before image processing")

    with pytest.raises(
        ProductionMultiReferenceError, match="unique approved reference ids"
    ):
        adapter(request, (ExplodingImage(), ExplodingImage()))


def test_duplicate_reference_error_lists_each_duplicate_once():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(("hero", "partner", "hero", "partner"))

    with pytest.raises(ProductionMultiReferenceError) as exc_info:
        adapter(request, (object(), object(), object(), object()))

    message = str(exc_info.value)
    assert "hero, partner" in message
    assert message.count("hero") == 1
    assert message.count("partner") == 1


def test_tiny_resolution_fails_closed_before_board_resize():
    image_module = pytest.importorskip("PIL.Image")
    adapter = ProductionReferenceBoardAdapter()
    request = _request(("hero", "partner"), resolution=(1, 1))
    images = (
        image_module.new("RGB", (8, 8), (255, 0, 0)),
        image_module.new("RGB", (8, 8), (0, 255, 0)),
    )

    with pytest.raises(ProductionMultiReferenceError, match="too small"):
        adapter(request, images)
