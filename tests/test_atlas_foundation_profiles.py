import pytest

from cineos.atlas.diffusers_video import FoundationProvenance
from cineos.atlas.foundation_profiles import (
    EXTERNAL_PRETRAINED_FOUNDATION,
    WAN22_TI2V_5B_DIFFUSERS_REVISION,
    WAN22_TI2V_5B_PROFILE,
    FoundationExecutionProfile,
    build_wan22_ti2v_5b_renderer,
)


class FakePipeline:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


def test_wan22_profile_is_pinned_and_explicitly_external():
    profile = WAN22_TI2V_5B_PROFILE

    assert profile.origin == EXTERNAL_PRETRAINED_FOUNDATION
    assert profile.provenance.model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert profile.provenance.revision == WAN22_TI2V_5B_DIFFUSERS_REVISION
    assert profile.provenance.license_id == "Apache-2.0"
    assert profile.resolutions == ((1280, 704), (704, 1280))
    assert profile.fps == (24.0,)
    assert profile.minimum_gpu_vram_gb == 24.0
    assert profile.snapshot()["origin"] == EXTERNAL_PRETRAINED_FOUNDATION


def test_wan22_profile_forwards_pinned_revision_to_diffusers(tmp_path):
    calls = []
    pipeline = FakePipeline()

    def factory(model_id, **options):
        calls.append((model_id, options))
        return pipeline

    renderer = build_wan22_ti2v_5b_renderer(
        output_dir=tmp_path,
        pipeline_factory=factory,
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32", local_files_only=True)

    assert calls == [
        (
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            {
                "local_files_only": True,
                "revision": WAN22_TI2V_5B_DIFFUSERS_REVISION,
            },
        )
    ]
    assert pipeline.device == "cpu"
    supported = {
        (resolution.width, resolution.height)
        for resolution in renderer.capabilities.supported_resolution
    }
    assert supported == {(1280, 704), (704, 1280)}


def test_profile_rejects_unpinned_or_relabelled_foundation():
    provenance = FoundationProvenance(
        model_id="example/model",
        license_id="Apache-2.0",
        source_url="https://example.invalid/model",
    )
    with pytest.raises(ValueError, match="immutable"):
        FoundationExecutionProfile(
            profile_id="unpinned",
            provenance=provenance,
            resolutions=((1280, 704),),
            fps=(24.0,),
            duration_range=(1.0, 5.0),
            minimum_gpu_vram_gb=24.0,
        )

    pinned = FoundationProvenance(
        model_id="example/model",
        revision="a" * 40,
        license_id="Apache-2.0",
        source_url="https://example.invalid/model",
    )
    with pytest.raises(ValueError, match="explicitly external"):
        FoundationExecutionProfile(
            profile_id="mislabelled",
            provenance=pinned,
            resolutions=((1280, 704),),
            fps=(24.0,),
            duration_range=(1.0, 5.0),
            minimum_gpu_vram_gb=24.0,
            origin="cineos_native",
        )
