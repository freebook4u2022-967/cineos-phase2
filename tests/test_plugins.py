from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from cineos.plugins import (
    Plugin,
    PluginContext,
    PluginLifecycleError,
    PluginManager,
    PluginMetadata,
    PluginRegistrationError,
)


class RecordingPlugin(Plugin):
    metadata = PluginMetadata("recording", "1.0.0", "Records lifecycle calls")

    def __init__(self) -> None:
        self.events: list[tuple[str, PluginContext]] = []

    def activate(self, context: PluginContext) -> None:
        self.events.append(("activate", context))

    def deactivate(self, context: PluginContext) -> None:
        self.events.append(("deactivate", context))


def test_register_and_lifecycle_are_idempotent() -> None:
    manager = PluginManager()
    plugin = manager.register(RecordingPlugin)
    context = PluginContext(services={"clock": object()}, settings={"quality": "draft"})

    assert manager.plugins == (plugin,)
    assert manager.activate("recording", context) is plugin
    manager.activate("recording", context)
    assert manager.active_plugins == (plugin,)
    assert plugin.events == [("activate", context)]

    manager.deactivate("recording", context)
    manager.deactivate("recording", context)
    assert plugin.events == [("activate", context), ("deactivate", context)]
    assert manager.active_plugins == ()


def test_context_copies_and_protects_host_mappings() -> None:
    settings = {"quality": "draft"}
    context = PluginContext(settings=settings)
    settings["quality"] = "final"

    assert context.settings == {"quality": "draft"}
    with pytest.raises(TypeError):
        context.settings["quality"] = "final"  # type: ignore[index]


def test_registration_rejects_duplicates_and_incompatible_plugins() -> None:
    manager = PluginManager()
    manager.register(RecordingPlugin())
    with pytest.raises(PluginRegistrationError, match="already registered"):
        manager.register(RecordingPlugin())

    class FuturePlugin(Plugin):
        metadata = PluginMetadata("future", "1.0.0", api_version="2")

    with pytest.raises(PluginRegistrationError, match="requires API"):
        manager.register(FuturePlugin())


def test_discovery_loads_entry_points_in_name_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlphaPlugin(Plugin):
        metadata = PluginMetadata("alpha", "1.0.0")

    entry_points = [
        EntryPoint("recording", "test_plugins:RecordingPlugin", "cineos.plugins"),
        EntryPoint("alpha", "test_plugins:AlphaPlugin", "cineos.plugins"),
    ]
    monkeypatch.setattr(__import__(__name__), "AlphaPlugin", AlphaPlugin, raising=False)

    manager = PluginManager()
    discovered = manager.discover(entry_points)

    assert [plugin.metadata.name for plugin in discovered] == ["alpha", "recording"]


def test_activate_all_rolls_back_plugins_activated_by_the_call() -> None:
    class BrokenPlugin(Plugin):
        metadata = PluginMetadata("z-broken", "1.0.0")

        def activate(self, context: PluginContext) -> None:
            raise RuntimeError("broken")

    manager = PluginManager()
    recording = manager.register(RecordingPlugin())
    manager.register(BrokenPlugin())

    with pytest.raises(PluginLifecycleError, match="failed to activate"):
        manager.activate_all()

    assert manager.active_plugins == ()
    assert [event for event, _ in recording.events] == ["activate", "deactivate"]
