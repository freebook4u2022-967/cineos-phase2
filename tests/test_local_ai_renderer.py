from pathlib import Path

import pytest

from cineos.cli.main import main
from cineos.compiler import compile
from cineos.compiler import save as save_film_package
from cineos.conditioning import (
    CameraConditioning,
    ConditioningPackage,
    ContinuityConditioning,
    RendererCapabilityRequirements,
)
from cineos.conditioning.serializer import save as save_conditioning
from cineos.core import MovieProject, Prop, Scene, Shot, Timeline
from cineos.renderers.local_ai import LocalAIConfig, LocalAIRenderer, RenderRequest
from cineos.renderers.local_ai.environment import EnvironmentReport
from cineos.renderers.local_ai.errors import RenderCancelled


class FakeBackend:
    def __init__(self):
        self.calls = []

    def load(self, model, **options):
        self.calls.append(("load", model, options))

    def warmup(self):
        self.calls.append(("warmup",))

    def generate(self, request, progress):
        self.calls.append(("generate", request.seed))
        progress(1, 2)
        progress(2, 2)
        return [b"frame"]

    def encode(self, frames, output, fps):
        self.calls.append(("encode", fps))
        Path(output).write_bytes(b"fake-mp4")

    def peak_vram(self):
        return 1234

    def unload(self):
        self.calls.append(("unload",))


def request(tmp_path, seed=41):
    return RenderRequest(
        "job-1",
        "shot-1",
        "a deterministic prompt",
        seed,
        tmp_path / "shot.mp4",
        576,
        320,
        8,
        2,
        2,
        9.0,
        (),
        (),
        {},
    )


def ready_renderer(tmp_path, monkeypatch, events=None):
    backend = FakeBackend()
    renderer = LocalAIRenderer(
        LocalAIConfig(model_path=str(tmp_path)),
        backend=backend,
        event_sink=(events.append if events is not None else lambda event: None),
    )
    monkeypatch.setattr(
        renderer, "validate_environment", lambda: EnvironmentReport(True)
    )
    renderer.initialize()
    renderer.load_model()
    renderer.warmup()
    return renderer, backend


def test_fake_backend_lifecycle_progress_and_structured_result(tmp_path, monkeypatch):
    events = []
    renderer, backend = ready_renderer(tmp_path, monkeypatch, events)
    result = renderer.render_shot(request(tmp_path))
    renderer.shutdown()

    assert [call[0] for call in backend.calls] == [
        "load",
        "warmup",
        "generate",
        "encode",
        "unload",
    ]
    assert [event.name for event in events] == [
        "renderer.initializing",
        "renderer.model_loading",
        "renderer.warming",
        "renderer.render_started",
        "renderer.progress",
        "renderer.progress",
        "renderer.encoding",
        "renderer.completed",
    ]
    assert result.job_id == "job-1"
    assert result.seed == 41
    assert result.peak_vram_bytes == 1234
    assert len(result.content_hash) == 64
    assert result.output_mp4_path.endswith("shot.mp4")


def test_cancellation_is_cooperative_and_emits_event(tmp_path, monkeypatch):
    events = []
    renderer, _ = ready_renderer(tmp_path, monkeypatch, events)
    renderer.cancel()
    with pytest.raises(RenderCancelled):
        renderer.render_shot(request(tmp_path))
    assert events[-1].name == "renderer.cancelled"


def test_same_input_constructs_same_backend_request(tmp_path):
    first = request(tmp_path)
    second = request(tmp_path)
    assert first == second
    assert first.frame_count == 16


def test_environment_reports_missing_dependencies_and_model(tmp_path):
    renderer = LocalAIRenderer(LocalAIConfig(model_path=str(tmp_path / "missing")))
    report = renderer.validate_environment()
    assert not report.valid
    assert any("missing model files" in error for error in report.errors)
    assert any("dependencies" in error for error in report.errors)


def test_renderer_cli_list_and_inspect(capsys):
    assert main(["renderer", "list"]) == 0
    assert "local-ai" in capsys.readouterr().out
    assert main(["renderer", "inspect", "local-ai"]) == 0
    assert "text-to-video-ms-1.7b" in capsys.readouterr().out


def test_renderer_cli_validate_fails_actionably(capsys):
    assert main(["renderer", "validate", "local-ai"]) == 4
    error = capsys.readouterr().err
    assert "environment is invalid" in error
    assert "no dependencies were installed" in error


def test_cli_dry_run_validates_without_loading_model(tmp_path, capsys):
    shot = Shot("shot-1", duration=2, action="a paper boat crossing a puddle")
    scene = Scene("scene-1", "Rain", shots=[shot], duration=2)
    package = compile(
        MovieProject(
            "Film",
            "Test",
            fps=8,
            resolution=(576, 320),
            props=[Prop("ref-1", "Paper boat")],
            scenes=[scene],
            timeline=Timeline(["scene-1"], {"scene-1": ["shot-1"]}),
        )
    )
    conditioning = ConditioningPackage(
        "shot-1",
        "scene-1",
        [],
        None,
        [],
        [],
        CameraConditioning(resolution=(576, 320), fps=8, duration=2),
        ContinuityConditioning(),
        ["ref-1"],
        RendererCapabilityRequirements(
            maximum_duration=2,
            supported_resolution=(576, 320),
            supported_fps=8,
        ),
        123,
    )
    package_path = tmp_path / "film-package.json"
    conditioning_path = tmp_path / "conditioning.json"
    save_film_package(package, package_path)
    save_conditioning(conditioning, conditioning_path)

    assert (
        main(
            [
                "renderer",
                "render",
                str(package_path),
                "--shot",
                "shot-1",
                "--conditioning",
                str(conditioning_path),
                "--output",
                str(tmp_path / "shot.mp4"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert "dry-run passed" in capsys.readouterr().out
    assert not (tmp_path / "shot.mp4").exists()
