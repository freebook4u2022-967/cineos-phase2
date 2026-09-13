from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image import NativeImageResearchBackend, compile_native_image_plan
from cineos.native_image.latent_model import CineosLatentFrameModel, NativePixelFrame


def _plan():
    request = NativeShotRequest(
        shot_id="shot-latent-001",
        scene_id="scene-001",
        camera={"resolution": (1920, 1080), "shot_type": "close-up"},
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
        environment={"description": "night interior"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=["ref-front"],
        deterministic_seed=321,
        renderer_requirements={"face_identity_support": True},
    )
    request.refresh_hash()
    return compile_native_image_plan(request)


def test_latent_model_produces_real_rgb_frame():
    backend = NativeImageResearchBackend(CineosLatentFrameModel(max_dimension=64))
    result = backend.render(_plan())

    assert isinstance(result.image, NativePixelFrame)
    assert result.image.width == 64
    assert result.image.height == 36
    assert len(result.image.rgb) == 64 * 36 * 3
    assert len(result.image.latent) == 16


def test_latent_frame_is_deterministic_for_same_seed():
    plan = _plan()
    first_backend = NativeImageResearchBackend(CineosLatentFrameModel(max_dimension=64))
    second_backend = NativeImageResearchBackend(
        CineosLatentFrameModel(max_dimension=64)
    )
    first = first_backend.render(plan)
    second = second_backend.render(plan)

    assert first.image.latent == second.image.latent
    assert first.image.rgb == second.image.rgb


def test_native_pixel_frame_writes_dependency_free_ppm(tmp_path):
    backend = NativeImageResearchBackend(CineosLatentFrameModel(max_dimension=32))
    result = backend.render(_plan())
    destination = result.image.save_ppm(tmp_path / "frame.ppm")
    payload = destination.read_bytes()

    assert payload.startswith(b"P6\n32 18\n255\n")
    assert len(payload) > 32 * 18 * 3
