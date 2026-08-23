from cineos.native_image import NativeImageResearchBackend, compile_native_image_plan
from cineos.native_image.frame_runtime import NativeFrameRuntime
from cineos.native_image.rerender import AutomaticRerenderController
from cineos.native_image.temporal_identity import IdentityObservation
from cineos.native_image.visual_qc import VisualContinuityObservation
from cineos.atlas.native_request import NativeShotRequest


class StubModel:
    def __init__(self):
        self.calls = 0
        self.corrections = []

    def encode_identity(self, tokens):
        return {"tokens": tokens}

    def encode_scene(self, plan):
        return {"shot_id": plan.shot_id}

    def generate(self, *, identity_state, scene_state, seed):
        self.calls += 1
        return {"frame": self.calls, "seed": seed}

    def apply_corrections(self, payload):
        self.corrections.append(payload)


class Observer:
    def __init__(self, fail_first=False, always_fail=False):
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.calls = 0

    def observe_identity(self, result, plan):
        self.calls += 1
        if self.always_fail or (self.fail_first and self.calls == 1):
            embedding = (-1.0, 0.0)
        else:
            embedding = (1.0, 0.0)
        return (
            IdentityObservation(
                character_uuid="hero",
                shot_id=plan.shot_id,
                embedding=embedding,
                approved_reference_ids=("ref-front",),
            ),
        )

    def observe_visual_continuity(self, result, plan):
        bad = self.always_fail or (self.fail_first and self.calls == 1)
        face = 0.40 if bad else 0.95
        return VisualContinuityObservation(
            shot_id=plan.shot_id,
            scores={
                "face_identity": face,
                "body_shape": 0.95,
                "wardrobe": 0.95,
                "hair": 0.95,
                "props": 0.95,
                "environment": 0.95,
                "lighting": 0.95,
                "screen_direction": 0.95,
            },
        )


def _plan():
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={"resolution": (1920, 1080)},
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


def test_native_frame_runtime_accepts_clean_first_frame():
    model = StubModel()
    runtime = NativeFrameRuntime(NativeImageResearchBackend(model), Observer())
    result = runtime.generate(_plan())
    assert result.accepted is True
    assert result.attempt_count == 1
    assert result.image["frame"] == 1


def test_native_frame_runtime_rerenders_then_accepts_corrected_frame():
    model = StubModel()
    runtime = NativeFrameRuntime(
        NativeImageResearchBackend(model),
        Observer(fail_first=True),
        controller=AutomaticRerenderController(max_attempts=3),
    )
    result = runtime.generate(_plan())
    assert result.accepted is True
    assert result.attempt_count == 2
    assert model.calls == 2
    assert len(model.corrections) == 1
    assert result.image["frame"] == 2


def test_native_frame_runtime_exhausts_budget_without_returning_bad_frame():
    model = StubModel()
    runtime = NativeFrameRuntime(
        NativeImageResearchBackend(model),
        Observer(always_fail=True),
        controller=AutomaticRerenderController(max_attempts=2),
    )
    result = runtime.generate(_plan())
    assert result.accepted is False
    assert result.final_decision == "reject"
    assert result.attempt_count == 2
    assert result.image is None
