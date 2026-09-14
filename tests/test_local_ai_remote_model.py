from cineos.renderers.local_ai import LocalAIConfig, LocalAIRenderer
from cineos.renderers.local_ai.environment import EnvironmentReport


class RecordingBackend:
    def __init__(self):
        self.load_call = None

    def load(self, model, **options):
        self.load_call = (model, options)

    def unload(self):
        return None


def test_remote_model_requires_declared_license_and_provenance():
    renderer = LocalAIRenderer(
        LocalAIConfig(
            model_path="example-org/example-video-model",
            allow_remote_model=True,
        )
    )

    report = renderer.validate_environment()

    assert not report.valid
    assert any("model_license" in error for error in report.errors)
    assert any("model_provenance" in error for error in report.errors)
    assert not any("missing model files" in error for error in report.errors)


def test_remote_model_policy_is_forwarded_to_backend(monkeypatch):
    backend = RecordingBackend()
    renderer = LocalAIRenderer(
        LocalAIConfig(
            model_path="example-org/example-video-model",
            allow_remote_model=True,
            model_revision="0123456789abcdef",
            model_license="Apache-2.0",
            model_provenance="Upstream model card reviewed for CINEOS evaluation",
            trust_remote_code=False,
        ),
        backend=backend,
    )
    monkeypatch.setattr(
        renderer, "validate_environment", lambda: EnvironmentReport(True)
    )

    renderer.initialize()
    renderer.load_model()

    model, options = backend.load_call
    assert model == "example-org/example-video-model"
    assert options["allow_remote_model"] is True
    assert options["model_revision"] == "0123456789abcdef"
    assert options["trust_remote_code"] is False
