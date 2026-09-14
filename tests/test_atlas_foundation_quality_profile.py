from cineos.atlas.foundation_profiles import (
    EXTERNAL_PRETRAINED_FOUNDATION,
    WAN22_I2V_A14B_DIFFUSERS_REVISION,
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
    build_wan22_i2v_a14b_renderer,
)


class FakePipeline:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


def test_wan22_i2v_a14b_quality_profile_is_pinned_external_and_i2v_only():
    profile = WAN22_I2V_A14B_PROFILE

    assert profile.origin == EXTERNAL_PRETRAINED_FOUNDATION
    assert profile.provenance.model_id == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert profile.provenance.revision == WAN22_I2V_A14B_DIFFUSERS_REVISION
    assert profile.provenance.license_id == "Apache-2.0"
    assert profile.provenance.foundation_name == "Wan2.2 I2V A14B"
    assert profile.minimum_gpu_vram_gb == 80.0
    assert profile.supported_features == frozenset({"image_to_video"})
    assert (1280, 720) in profile.resolutions
    assert (832, 480) in profile.resolutions
    assert profile.snapshot()["supported_features"] == ["image_to_video"]


def test_existing_ti2v_profile_retains_hybrid_capabilities():
    assert WAN22_TI2V_5B_PROFILE.supported_features == frozenset(
        {"text_to_video", "image_to_video"}
    )


def test_wan22_i2v_a14b_builder_forwards_pinned_revision_and_capabilities(tmp_path):
    calls = []
    pipeline = FakePipeline()

    def factory(model_id, **options):
        calls.append((model_id, options))
        return pipeline

    renderer = build_wan22_i2v_a14b_renderer(
        output_dir=tmp_path,
        pipeline_factory=factory,
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32", local_files_only=True)

    assert calls == [
        (
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            {
                "local_files_only": True,
                "revision": WAN22_I2V_A14B_DIFFUSERS_REVISION,
            },
        )
    ]
    assert pipeline.device == "cpu"
    assert renderer.capabilities.supported_features == frozenset({"image_to_video"})
    supported = {
        (resolution.width, resolution.height)
        for resolution in renderer.capabilities.supported_resolution
    }
    assert supported == {(1280, 720), (720, 1280), (832, 480), (480, 832)}
