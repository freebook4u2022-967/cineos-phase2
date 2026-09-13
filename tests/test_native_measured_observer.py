from __future__ import annotations

import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image.backend import NativeImageResearchResult
from cineos.native_image.conditioning import compile_native_image_plan
from cineos.native_image.measured_observer import (
    MeasuredNativeFrameObserver,
    merge_visual_observations,
)
from cineos.native_image.neural_decoder import DecodedRGBFrame
from cineos.native_image.spatial_evidence import MeasuredSpatialContinuityObserver
from cineos.native_image.visual_qc import VisualContinuityObservation


class _EmptyIdentitySource:
    def observe_identity(self, result, plan):
        del result, plan
        return ()


class _FailAfterAcceptSpatialObserver(MeasuredSpatialContinuityObserver):
    def accept(self, result, plan):
        super().accept(result, plan)
        raise RuntimeError("simulated durable spatial commit failure")


def _frame_from_rows(rows: list[list[int]]) -> DecodedRGBFrame:
    height = len(rows)
    width = len(rows[0])
    rgb = bytearray()
    for row in rows:
        assert len(row) == width
        for value in row:
            rgb.extend((value, value, value))
    return DecodedRGBFrame(width, height, bytes(rgb))


def _result(frame: DecodedRGBFrame, shot_id: str) -> NativeImageResearchResult:
    return NativeImageResearchResult(
        shot_id=shot_id,
        plan_hash="plan-hash",
        seed=123,
        identity_state={},
        scene_state={},
        image=frame,
    )


def _plan(shot_id: str):
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


def test_merge_visual_observations_preserves_worst_overlapping_axis():
    global_pixels = VisualContinuityObservation(
        shot_id="shot-001",
        scores={"environment": 0.92, "lighting": 0.88},
        confidence=0.95,
    )
    spatial = VisualContinuityObservation(
        shot_id="shot-001",
        scores={"environment": 0.41},
        confidence=0.90,
    )

    merged = merge_visual_observations(global_pixels, spatial)

    assert merged.scores == {"environment": 0.41, "lighting": 0.88}
    assert merged.confidence == 0.90


def test_merge_visual_observations_rejects_mixed_shot_evidence():
    first = VisualContinuityObservation("shot-001", {"environment": 1.0})
    second = VisualContinuityObservation("shot-002", {"lighting": 1.0})

    with pytest.raises(ValueError, match="same shot"):
        merge_visual_observations(first, second)


def test_measured_observer_combines_pixel_and_dense_spatial_evidence():
    observer = MeasuredNativeFrameObserver(_EmptyIdentitySource())
    first_plan = _plan("shot-001")
    second_plan = _plan("shot-002")
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
    observer.accept_frame(_result(left_bright, "shot-001"), first_plan)

    candidate = _result(right_bright, "shot-002")
    pixel = observer.pixel_observer.observe(candidate, second_plan)
    spatial = observer.spatial_observer.observe(candidate, second_plan)
    combined = observer.observe_visual_continuity(candidate, second_plan)

    assert combined.scores["environment"] == min(
        pixel.scores["environment"], spatial.scores["environment"]
    )
    assert combined.scores["lighting"] == pixel.scores["lighting"]
    assert combined.scores["environment"] < 0.7


def test_measured_observer_checkpoint_round_trip_restores_both_memories():
    observer = MeasuredNativeFrameObserver(_EmptyIdentitySource())
    plan = _plan("shot-001")
    frame = _frame_from_rows(
        [
            [32, 64, 96, 128],
            [32, 64, 96, 128],
            [32, 64, 96, 128],
            [32, 64, 96, 128],
        ]
    )
    observer.accept_frame(_result(frame, "shot-001"), plan)
    checkpoint = observer.checkpoint_state()

    restored = MeasuredNativeFrameObserver(_EmptyIdentitySource())
    restored.restore_state(checkpoint)

    assert restored.checkpoint_state() == checkpoint


def test_measured_observer_rolls_back_all_memories_when_second_commit_fails():
    observer = MeasuredNativeFrameObserver(
        _EmptyIdentitySource(),
        spatial_observer=_FailAfterAcceptSpatialObserver(),
    )
    before = observer.checkpoint_state()
    plan = _plan("shot-001")
    frame = _frame_from_rows(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ]
    )

    with pytest.raises(RuntimeError, match="spatial commit failure"):
        observer.accept_frame(_result(frame, "shot-001"), plan)

    assert observer.checkpoint_state() == before
