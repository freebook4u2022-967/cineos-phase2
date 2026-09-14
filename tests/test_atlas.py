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
    assert negotiated.character_count is None


def test_capabilities_report_all_unsupported_values() -> None:
    with pytest.raises(CapabilityError, match="resolution.*duration.*fps.*features"):
        StubRenderer().capabilities.negotiate(
            resolution=(640, 480), duration=20, fps=60, features=("depth",)
        )


def test_capabilities_enforce_declared_character_capacity() -> None:
    capabilities = RendererCapabilities(
        supported_resolution=(Resolution(1920, 1080),),
        supported_duration=Range(1, 10),
        supported_fps=(24,),
        maximum_character_count=2,
    )

    negotiated = capabilities.negotiate(
        resolution=(1920, 1080), duration=5, fps=24, character_count=2
    )
    assert negotiated.character_count == 2

    with pytest.raises(CapabilityError, match="character_count 3 exceeds maximum 2"):
        capabilities.negotiate(
            resolution=(1920, 1080), duration=5, fps=24, character_count=3
        )


def test_capabilities_reject_invalid_character_counts() -> None:
    with pytest.raises(ValueError, match="maximum character count"):
        RendererCapabilities(
            supported_resolution=(Resolution(1920, 1080),),
            supported_duration=Range(1, 10),
            supported_fps=(24,),
            maximum_character_count=True,
        )

    with pytest.raises(ValueError, match="character_count"):
        StubRenderer().capabilities.negotiate(
            resolution=(1920, 1080), duration=5, fps=24, character_count=-1
        )


def test_session_forwards_character_count_to_capability_negotiation() -> None:
    class TwoCharacterRenderer(StubRenderer):
        @property
        def capabilities(self) -> RendererCapabilities:
            return RendererCapabilities(
                supported_resolution=(Resolution(1920, 1080),),
                supported_duration=Range(1, 10),
                supported_fps=(24,),
                maximum_character_count=2,
            )

    session = RendererSession(TwoCharacterRenderer())
    negotiated = session.negotiate(
        resolution=(1920, 1080), duration=2, fps=24, character_count=2
    )
    assert negotiated.character_count == 2

    with pytest.raises(CapabilityError, match="character_count 3 exceeds maximum 2"):
        session.negotiate(
            resolution=(1920, 1080), duration=2, fps=24, character_count=3
        )


def test_session_rejects_stale_negotiated_cast_size_before_render() -> None:
    class TwoCharacterRenderer(StubRenderer):
        @property
        def capabilities(self) -> RendererCapabilities:
            return RendererCapabilities(
                supported_resolution=(Resolution(1920, 1080),),
                supported_duration=Range(1, 10),
                supported_fps=(24,),
                maximum_character_count=2,
            )

    class Request:
        characters = [{"character_uuid": "hero"}, {"character_uuid": "partner"}]

    renderer = TwoCharacterRenderer()
    session = RendererSession(renderer)
    session.start()
    session.negotiate(resolution=(1920, 1080), duration=2, fps=24, character_count=1)
    with pytest.raises(
        CapabilityError,
        match="request character_count 2 does not match negotiated character_count 1",
    ):
        session.render(Request())
    assert not any(
        isinstance(call, tuple) and call[0] == "render" for call in renderer.calls
    )


def test_session_enforces_actual_cast_capacity_for_legacy_negotiation() -> None:
    class TwoCharacterRenderer(StubRenderer):
        @property
        def capabilities(self) -> RendererCapabilities:
            return RendererCapabilities(
                supported_resolution=(Resolution(1920, 1080),),
                supported_duration=Range(1, 10),
                supported_fps=(24,),
                maximum_character_count=2,
            )

    class Request:
        characters = [
            {"character_uuid": "hero"},
            {"character_uuid": "partner"},
            {"character_uuid": "antagonist"},
        ]

    renderer = TwoCharacterRenderer()
    session = RendererSession(renderer)
    session.start()
    session.negotiate(resolution=(1920, 1080), duration=2, fps=24)
    with pytest.raises(
        CapabilityError, match="request character_count 3 exceeds maximum 2"
    ):
        session.render(Request())
    assert not any(
        isinstance(call, tuple) and call[0] == "render" for call in renderer.calls
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
