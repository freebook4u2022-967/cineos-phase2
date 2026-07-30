from typing import Any

import pytest

from cineos.atlas import (
    BaseRenderer,
    CapabilityError,
    Range,
    RendererAdapter,
    RendererCapabilities,
    RendererLifecycleError,
    RendererRegistry,
    RendererSession,
    RendererState,
    Resolution,
)


class StubRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.calls: list[object] = []

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            supported_resolution=(Resolution(1920, 1080),),
            supported_duration=Range(1, 10),
            supported_fps=(24, 30),
            supported_features=frozenset({"audio"}),
        )

    def initialize(self) -> None:
        self.calls.append("initialize")

    def load_model(self, model: str | None = None, **options: Any) -> None:
        self.calls.append(("load_model", model, options))

    def warmup(self) -> None:
        self.calls.append("warmup")

    def render(self, request: Any) -> Any:
        self.calls.append(("render", request))
        return {"rendered": request}

    def shutdown(self) -> None:
        self.calls.append("shutdown")


def test_adapter_enforces_and_runs_lifecycle() -> None:
    renderer = StubRenderer()
    adapter = RendererAdapter(renderer)
    with pytest.raises(RendererLifecycleError):
        adapter.render("scene")

    adapter.initialize()
    adapter.load_model("example", revision="v1")
    adapter.warmup()
    assert adapter.render("scene") == {"rendered": "scene"}
    adapter.shutdown()
    adapter.shutdown()

    assert adapter.state is RendererState.SHUTDOWN
    assert renderer.calls == [
        "initialize",
        ("load_model", "example", {"revision": "v1"}),
        "warmup",
        ("render", "scene"),
        "shutdown",
    ]


def test_capabilities_negotiate_supported_request() -> None:
    capabilities = StubRenderer().capabilities
    negotiated = capabilities.negotiate(
        resolution=(1920, 1080), duration=5, fps=24, features=("audio",)
    )
    assert negotiated.resolution == Resolution(1920, 1080)
    assert negotiated.features == frozenset({"audio"})


def test_capabilities_report_all_unsupported_values() -> None:
    with pytest.raises(CapabilityError, match="resolution.*duration.*fps.*features"):
        StubRenderer().capabilities.negotiate(
            resolution=(640, 480), duration=20, fps=60, features=("depth",)
        )


def test_registry_normalizes_names_and_creates_fresh_renderers() -> None:
    registry = RendererRegistry()
    registry.register(" Stub ", StubRenderer)
    assert registry.names() == ("stub",)
    assert "STUB" in registry
    assert isinstance(registry.create("stub"), StubRenderer)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("stub", StubRenderer)


def test_session_context_runs_complete_lifecycle() -> None:
    renderer = StubRenderer()
    with RendererSession(renderer) as session:
        session.negotiate(resolution=(1920, 1080), duration=2, fps=30)
        assert session.render("shot") == {"rendered": "shot"}
    assert renderer.calls == [
        "initialize",
        ("load_model", None, {}),
        "warmup",
        ("render", "shot"),
        "shutdown",
    ]
