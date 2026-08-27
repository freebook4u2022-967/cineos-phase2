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
    PixelContinuityMemory,
    describe_rgb_frame,
)
from cineos.native_image.rerender import AutomaticRerenderController


def _solid_frame(value: int, *, width: int = 4, height: int = 4) -> DecodedRGBFrame:
    return DecodedRGBFrame(width, height, bytes([value, value, value] * width * height))


def _vertical_split_frame(*, bright_left: bool) -> DecodedRGBFrame:
    pixels = bytearray()
    for _y in range(4):
        for x in range(4):
            bright = x < 2 if bright_left else x >= 2
            value = 224 if bright else 32
            pixels.extend((value, value, value))
    return DecodedRGBFrame(4, 4, bytes(pixels))


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
        characters=[
            {
                "character_uuid": "hero",
                "cinedna_profile_id": "hero",
                "cinedna_profile_version": "1.0",
                "approved_reference_ids": ["ref-front"],
                "identity_invariants": ["same face"],
                "face_constraints": {},
                "body_constraints": {},
                "scene_specific_overrides": {"primary_reference_id": "ref-front"},
            }
        ],
        environment=None,
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=["ref-front"],
        deterministic_seed=123,
        renderer_requirements={"face_identity_support": True},
    )
    request.refresh_hash()
    return compile_native_image_plan(request)


def test_generated_rgb_descriptor_detects_black_and_mid_gray():
    black = describe_rgb_frame(_solid_frame(0))
    gray = describe_rgb_frame(_solid_frame(128))

    assert black.mean_luma == 0.0
    assert black.black_fraction == 1.0
    assert black.clipped_fraction == 0.0
    assert black.edge_energy == 0.0
    assert 0.49 < gray.mean_luma < 0.51
    assert gray.black_fraction == 0.0
    assert sum(gray.luma_histogram) == 1.0
    assert all(0.49 < value < 0.51 for value in gray.spatial_luma)


def test_generated_rgb_descriptor_measures_spatial_layout_and_edges():
    left = describe_rgb_frame(_vertical_split_frame(bright_left=True))
    right = describe_rgb_frame(_vertical_split_frame(bright_left=False))

    assert abs(left.mean_luma - right.mean_luma) < 1e-12
    assert left.luma_histogram == right.luma_histogram
    assert left.spatial_luma != right.spatial_luma
    assert left.spatial_luma[0] > left.spatial_luma[1]
    assert right.spatial_luma[0] < right.spatial_luma[1]
    assert left.edge_energy > 0.0
    assert right.edge_energy == left.edge_energy


def test_pixel_observer_scores_against_last_accepted_scene_baseline():
    memory = PixelContinuityMemory()
    observer = DecodedPixelContinuityObserver(memory)
    plan = _plan()
    baseline = _result(_solid_frame(128))
    observer.accept(baseline, plan)

    same = observer.observe(_result(_solid_frame(128)), plan)
    changed = observer.observe(_result(_solid_frame(0)), plan)

    assert same.scores["environment"] == 1.0
    assert same.scores["lighting"] == 1.0
    assert changed.scores["environment"] < 0.80
    assert changed.scores["lighting"] < 0.70


def test_pixel_observer_detects_layout_change_with_same_global_histogram():
    observer = DecodedPixelContinuityObserver()
    plan = _plan()
    observer.accept(_result(_vertical_split_frame(bright_left=True)), plan)

    observation = observer.observe(
        _result(_vertical_split_frame(bright_left=False)), plan
    )

    assert observation.scores["lighting"] == 1.0
    assert observation.scores["environment"] < 1.0


def test_pixel_continuity_memory_round_trips_versioned_state():
    memory = PixelContinuityMemory()
    descriptor = describe_rgb_frame(_vertical_split_frame(bright_left=True))
    memory.accept("scene-001", descriptor)

    snapshot = memory.snapshot()
    restored = PixelContinuityMemory.restore(snapshot)

    assert snapshot["schema"] == "cineos-pixel-continuity-memory/0.2"
    assert restored.latest("scene-001") == descriptor


def test_pixel_continuity_memory_restores_legacy_v01_state():
    descriptor = describe_rgb_frame(_solid_frame(96)).snapshot()
    descriptor.pop("spatial_luma")
    descriptor.pop("edge_energy")
    restored = PixelContinuityMemory.restore(
        {
            "schema": "cineos-pixel-continuity-memory/0.1",
            "accepted": {"scene-001": descriptor},
        }
    )

    legacy = restored.latest("scene-001")
    assert legacy is not None
    assert legacy.spatial_luma == (0.0, 0.0, 0.0, 0.0)
    assert legacy.edge_energy == 0.0


class _SequenceModel:
    def __init__(self):
        self.frames = [_solid_frame(0), _solid_frame(128)]

    def encode_identity(self, tokens):
        return {"tokens": tokens}

    def encode_scene(self, plan):
        return {"scene_id": plan.scene_id}

    def generate(self, *, identity_state, scene_state, seed):
        return self.frames.pop(0)


class _NoIdentitySource:
    def observe_identity(self, result, plan):
        return ()


def test_frame_runtime_never_commits_rejected_generated_pixels():
    plan = _plan()
    pixel_observer = DecodedPixelContinuityObserver()
    pixel_observer.accept(_result(_solid_frame(128)), plan)
    observer = PixelAwareNativeFrameObserver(
        identity_source=_NoIdentitySource(), pixel_observer=pixel_observer
    )
    runtime = NativeFrameRuntime(
        NativeImageResearchBackend(_SequenceModel()),
        observer,
        controller=AutomaticRerenderController(max_attempts=2),
    )

    result = runtime.generate(plan)

    assert result.accepted is True
    assert result.attempt_count == 2
    assert result.image == _solid_frame(128)
    assert pixel_observer.memory.latest(plan.scene_id) == describe_rgb_frame(
        _solid_frame(128)
    )
