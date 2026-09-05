from cineos.renderers.local_ai import LocalAIConfig, LocalAIRenderer


def test_production_mode_rejects_cpu_inference():
    renderer = LocalAIRenderer(
        LocalAIConfig(
            device="cpu",
            production_mode=True,
            allow_remote_model=True,
            model_path="example-org/example-video-model",
            model_revision="0123456789abcdef",
            model_license="Apache-2.0",
            model_provenance="Reviewed upstream model card",
        )
    )

    report = renderer.validate_environment()

    assert not report.valid
    assert "production_mode requires CUDA-backed inference" in report.errors
    assert report.details["production_mode"] is True


def test_production_remote_model_requires_pinned_revision():
    renderer = LocalAIRenderer(
        LocalAIConfig(
            device="cpu",
            production_mode=True,
            allow_remote_model=True,
            model_path="example-org/example-video-model",
            model_license="Apache-2.0",
            model_provenance="Reviewed upstream model card",
        )
    )

    report = renderer.validate_environment()

    assert not report.valid
    assert (
        "production remote models require model_revision to be pinned" in report.errors
    )
