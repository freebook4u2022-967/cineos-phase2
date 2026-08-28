import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image.backend import (
    NativeImageResearchBackend,
    NativeImageResearchResult,
)
from cineos.native_image.conditioning import compile_native_image_plan
from cineos.native_image.frame_runtime import NativeFrameRuntime
from cineos.native_image.neural_decoder import DecodedRGBFrame
from cineos.native_image.pixel_observer import (
    DecodedPixelContinuityObserver,
    PixelAwareNativeFrameObserver,
    describe_rgb_frame,
)
from cineos.native_image.rerender import AutomaticRerenderController


def _solid_frame(value: int) -> DecodedRGBFrame:
    return DecodedRGBFrame(4, 4, bytes([value, value, value] * 16))


def _result(frame: DecodedRGBFrame) -> NativeImageResearchResult:
    return NativeImageResearchResult(
        shot_id="shot-001",
        plan_hash="plan-hash",
        seed=123,
        identity_state={},
        scene_state={},
        image=frame,
    )


def _plan():
    request = NativeShotRequest(
        shot_id="shot-001",
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


class _NoIdentitySource:
    def observe_identity(self, result, plan):
        return ()


class _SingleFrameModel:
    def __init__(self, frame: DecodedRGBFrame):
        self.frame = frame

    def encode_identity(self, tokens):
        return {"tokens": tokens}

    def encode_scene(self, plan):
        return {"scene_id": plan.scene_id}

    def generate(self, *, identity_state, scene_state, seed):
        return self.frame


class _FailAfterPixelCommitObserver:
    """Exercise runtime rollback after observer state has already changed."""

    def __init__(self, inner: PixelAwareNativeFrameObserver):
        self.inner = inner

    def observe_identity(self, result, plan):
        return self.inner.observe_identity(result, plan)

    def observe_visual_continuity(self, result, plan):
        return self.inner.observe_visual_continuity(result, plan)

    def checkpoint_state(self):
        return self.inner.checkpoint_state()

    def restore_state(self, checkpoint):
        self.inner.restore_state(checkpoint)

    def accept_frame(self, result, plan):
        self.inner.accept_frame(result, plan)
        raise RuntimeError("simulated downstream acceptance failure")


def test_pixel_aware_observer_checkpoint_round_trips_memory():
    plan = _plan()
    pixel_observer = DecodedPixelContinuityObserver()
    observer = PixelAwareNativeFrameObserver(
        identity_source=_NoIdentitySource(), pixel_observer=pixel_observer
    )
    baseline = _result(_solid_frame(96))
    observer.accept_frame(baseline, plan)
    checkpoint = observer.checkpoint_state()

    observer.accept_frame(_result(_solid_frame(160)), plan)
    assert pixel_observer.memory.latest(plan.scene_id) == describe_rgb_frame(
        _solid_frame(160)
    )

    observer.restore_state(checkpoint)
    assert pixel_observer.memory.latest(plan.scene_id) == describe_rgb_frame(
        _solid_frame(96)
    )


def test_frame_runtime_rolls_back_pixels_when_acceptance_transaction_fails():
    plan = _plan()
    pixel_observer = DecodedPixelContinuityObserver()
    inner = PixelAwareNativeFrameObserver(
        identity_source=_NoIdentitySource(), pixel_observer=pixel_observer
    )
    baseline = _result(_solid_frame(128))
    inner.accept_frame(baseline, plan)
    observer = _FailAfterPixelCommitObserver(inner)
    runtime = NativeFrameRuntime(
        NativeImageResearchBackend(_SingleFrameModel(_solid_frame(129))),
        observer,
        controller=AutomaticRerenderController(max_attempts=1),
    )

    with pytest.raises(RuntimeError, match="downstream acceptance failure"):
        runtime.generate(plan)

    assert pixel_observer.memory.latest(plan.scene_id) == describe_rgb_frame(
        _solid_frame(128)
    )


def test_pixel_aware_observer_rejects_unknown_checkpoint_schema():
    observer = PixelAwareNativeFrameObserver(identity_source=_NoIdentitySource())

    with pytest.raises(
        ValueError, match="unsupported pixel-aware observer checkpoint schema"
    ):
        observer.restore_state({"schema": "cineos-pixel-aware-observer-checkpoint/999"})
