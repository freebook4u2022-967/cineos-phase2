"""Regression coverage for quality-first production foundation selection."""

from types import SimpleNamespace

import pytest

from cineos.atlas.foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.gpu_preflight import GPUDeviceProfile
from cineos.atlas.production_foundation_selection import (
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
)


def _gpu(total: float, free: float | None = None) -> GPUDeviceProfile:
    return GPUDeviceProfile(
        index=0,
        name="test-gpu",
        compute_capability=(9, 0),
        total_vram_gb=total,
        free_vram_gb=total if free is None else free,
        supports_bfloat16=True,
    )


def _request(
    *refs: str,
    resolution: tuple[int, int] = (832, 480),
    fps: float = 16.0,
    duration: float = 5.0,
    renderer_requirements: dict | None = None,
):
    request = SimpleNamespace(
        approved_reference_ids=tuple(refs),
        camera={"resolution": resolution, "fps": fps, "duration": duration},
    )
    if renderer_requirements is not None:
        request.renderer_requirements = renderer_requirements
    return request


def _a14b_request(*refs: str):
    return _request(*refs, resolution=(832, 480), fps=16.0)


def _five_b_request(*refs: str):
    return _request(*refs, resolution=(1280, 704), fps=24.0)


def test_prefers_a14b_when_every_shot_is_image_conditioned_and_resident_plan_is_validated():
    selection = select_strongest_production_foundation(
        (_gpu(100.0, 96.0),),
        (_a14b_request("hero"), _a14b_request("hero", "partner")),
    )

    assert selection.profile is WAN22_I2V_A14B_PROFILE
    assert selection.plan.memory_strategy == "resident"
    assert selection.fallback_used is False
    assert selection.rejected_profiles == ()
    assert selection.to_dict()["origin"] == "external_pretrained_foundation"


def test_selection_manifest_persists_exact_external_foundation_provenance():
    selection = select_strongest_production_foundation(
        (_gpu(100.0, 96.0),),
        (_a14b_request("hero"), _a14b_request("hero", "partner")),
    )

    payload = selection.to_dict()
    provenance = WAN22_I2V_A14B_PROFILE.provenance
    assert payload["model_id"] == provenance.model_id
    assert payload["revision"] == provenance.revision
    assert payload["license_id"] == provenance.license_id
    assert payload["source_url"] == provenance.source_url
    assert payload["foundation_name"] == provenance.foundation_name
    assert payload["origin"] == "external_pretrained_foundation"


def test_fallback_manifest_keeps_5b_external_provenance_explicit():
    selection = select_strongest_production_foundation(
        (_gpu(48.0, 44.0),),
        (_five_b_request("hero"), _five_b_request("hero")),
    )

    payload = selection.to_dict()
    provenance = WAN22_TI2V_5B_PROFILE.provenance
    assert payload["model_id"] == provenance.model_id
    assert payload["revision"] == provenance.revision
    assert payload["license_id"] == provenance.license_id
    assert payload["source_url"] == provenance.source_url
    assert payload["foundation_name"] == provenance.foundation_name
    assert payload["origin"] == "external_pretrained_foundation"


def test_falls_back_to_5b_below_unvalidated_a14b_vram_floor():
    selection = select_strongest_production_foundation(
        (_gpu(48.0, 44.0),),
        (_five_b_request("hero"), _five_b_request("hero")),
    )

    assert selection.profile is WAN22_TI2V_5B_PROFILE
    assert selection.fallback_used is True
    assert any("production floor" in reason for reason in selection.rejected_profiles)


def test_fails_closed_when_a14b_floor_is_met_but_only_unvalidated_offload_is_available():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="requires validated resident execution; generic planner selected model_cpu_offload",
    ):
        select_strongest_production_foundation(
            (_gpu(96.0, 90.0),),
            (_a14b_request("hero"), _a14b_request("hero", "partner")),
        )


def test_falls_back_to_5b_when_any_shot_lacks_image_conditioning():
    selection = select_strongest_production_foundation(
        (_gpu(96.0, 90.0),),
        (_five_b_request("hero"), _five_b_request()),
    )

    assert selection.profile is WAN22_TI2V_5B_PROFILE
    assert selection.fallback_used is True
    assert any("image conditioning" in reason for reason in selection.rejected_profiles)


def test_high_vram_runner_does_not_select_a14b_for_5b_generation_contract():
    selection = select_strongest_production_foundation(
        (_gpu(96.0, 90.0),),
        (_five_b_request("hero"), _five_b_request("hero", "partner")),
    )

    assert selection.profile is WAN22_TI2V_5B_PROFILE
    assert selection.fallback_used is True
    assert any(
        "unsupported resolution 1280x704" in reason
        for reason in selection.rejected_profiles
    )


def test_rejects_a14b_when_exact_request_fps_is_not_profile_supported():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="unsupported fps 24",
    ):
        select_strongest_production_foundation(
            (_gpu(96.0, 90.0),),
            (
                _request("hero", resolution=(832, 480), fps=24.0),
                _request("hero", resolution=(832, 480), fps=24.0),
            ),
        )


def test_rejects_all_profiles_for_unsupported_generation_resolution():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="unsupported resolution 1920x1080",
    ):
        select_strongest_production_foundation(
            (_gpu(96.0, 90.0),),
            (
                _request("hero", resolution=(1920, 1080), fps=24.0),
                _request("hero", resolution=(1920, 1080), fps=24.0),
            ),
        )


def test_rejects_camera_resolution_that_conflicts_with_native_renderer_requirements():
    request = _request(
        "hero",
        renderer_requirements={
            "supported_resolution": [1280, 720],
            "supported_fps": 16.0,
            "maximum_duration": 5.0,
        },
    )

    with pytest.raises(
        ProductionFoundationSelectionError,
        match="camera resolution conflicts with renderer_requirements",
    ):
        select_strongest_production_foundation((_gpu(96.0, 90.0),), (request,))


def test_rejects_camera_fps_that_conflicts_with_native_renderer_requirements():
    request = _request(
        "hero",
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 24.0,
            "maximum_duration": 5.0,
        },
    )

    with pytest.raises(
        ProductionFoundationSelectionError,
        match="camera fps conflicts with renderer_requirements",
    ):
        select_strongest_production_foundation((_gpu(96.0, 90.0),), (request,))


def test_rejects_camera_duration_above_native_renderer_requirement_envelope():
    request = _request(
        "hero",
        duration=5.0,
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 16.0,
            "maximum_duration": 4.0,
        },
    )

    with pytest.raises(
        ProductionFoundationSelectionError,
        match="camera duration exceeds renderer_requirements.maximum_duration",
    ):
        select_strongest_production_foundation((_gpu(96.0, 90.0),), (request,))


def test_accepts_consistent_native_renderer_requirements():
    request = _request(
        "hero",
        duration=4.0,
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 16.0,
            "maximum_duration": 5.0,
        },
    )

    selection = select_strongest_production_foundation((_gpu(100.0, 96.0),), (request,))

    assert selection.profile is WAN22_I2V_A14B_PROFILE
    assert selection.plan.memory_strategy == "resident"


def test_rejects_empty_connected_shot_set_before_gpu_or_model_acquisition():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="requires at least one native shot request",
    ):
        select_strongest_production_foundation((_gpu(100.0, 96.0),), ())


def test_fails_closed_when_no_approved_profile_can_fit():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="no approved pinned production video foundation can safely execute",
    ):
        select_strongest_production_foundation(
            (_gpu(4.0, 3.0),),
            (_five_b_request("hero"), _five_b_request("hero")),
        )
