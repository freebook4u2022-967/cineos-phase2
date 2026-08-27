import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image.backend import NativeImageResearchResult
from cineos.native_image.conditioning import compile_native_image_plan
from cineos.native_image.neural_decoder import DecodedRGBFrame
from cineos.native_image.spatial_evidence import (
    MeasuredSpatialContinuityObserver,
    SpatialContinuityMemory,
    describe_spatial_rgb_frame,
    spatial_similarity,
)


def _frame_from_rows(rows: list[list[int]]) -> DecodedRGBFrame:
    height = len(rows)
    width = len(rows[0])
    rgb = bytearray()
    for row in rows:
        assert len(row) == width
        for value in row:
            rgb.extend((value, value, value))
    return DecodedRGBFrame(width, height, bytes(rgb))


def _result(
    frame: DecodedRGBFrame, shot_id: str = "shot-001"
) -> NativeImageResearchResult:
    return NativeImageResearchResult(
        shot_id=shot_id,
        plan_hash="plan-hash",
        seed=123,
        identity_state={},
        scene_state={},
        image=frame,
    )


def _plan(shot_id: str = "shot-001"):
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-001",
        camera={"resolution": (4, 4)},
        characters=[],
        environment=None,
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=[],
        deterministic_seed=123,
        renderer_requirements={},
    )
    request.refresh_hash()
    return compile_native_image_plan(request)


def test_spatial_descriptor_detects_directional_edge_energy():
    vertical_split = _frame_from_rows(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]
    )
    descriptor = describe_spatial_rgb_frame(vertical_split)

    assert descriptor.horizontal_edge_energy > 0.0
    assert descriptor.vertical_edge_energy == 0.0
    assert descriptor.luma_grid[0] == 0.0
    assert descriptor.luma_grid[3] == pytest.approx(1.0)


def test_spatial_similarity_penalizes_composition_shift_with_same_global_luma():
    left_bright = _frame_from_rows(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ]
    )
    right_bright = _frame_from_rows(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]
    )

    baseline = describe_spatial_rgb_frame(left_bright)
    shifted = describe_spatial_rgb_frame(right_bright)

    assert spatial_similarity(baseline, baseline) == pytest.approx(1.0)
    assert spatial_similarity(baseline, shifted) < 0.6


def test_spatial_similarity_fails_closed_for_different_sampling_geometry():
    small = describe_spatial_rgb_frame(_frame_from_rows([[0, 255], [0, 255]]))
    large = describe_spatial_rgb_frame(
        _frame_from_rows(
            [
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
            ]
        )
    )

    assert spatial_similarity(small, large) == 0.0


def test_spatial_memory_round_trips_versioned_accepted_state():
    memory = SpatialContinuityMemory()
    descriptor = describe_spatial_rgb_frame(
        _frame_from_rows(
            [
                [64, 64, 128, 128],
                [64, 64, 128, 128],
                [64, 64, 128, 128],
                [64, 64, 128, 128],
            ]
        )
    )
    memory.accept("scene-001", descriptor)

    restored = SpatialContinuityMemory.restore(memory.snapshot())

    assert restored.latest("scene-001") == descriptor


def test_spatial_observer_commits_only_on_explicit_accept():
    observer = MeasuredSpatialContinuityObserver()
    plan = _plan()
    baseline_frame = _frame_from_rows(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ]
    )
    candidate_frame = _frame_from_rows(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]
    )
    baseline = _result(baseline_frame)
    candidate = _result(candidate_frame)
    observer.accept(baseline, plan)

    before = observer.memory.latest(plan.scene_id)
    report = observer.observe(candidate, plan)
    after_observe = observer.memory.latest(plan.scene_id)

    assert report.scores["environment"] < 0.6
    assert after_observe == before

    observer.accept(candidate, plan)
    assert observer.memory.latest(plan.scene_id) == describe_spatial_rgb_frame(
        candidate_frame
    )
