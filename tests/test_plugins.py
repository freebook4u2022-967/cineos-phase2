from dataclasses import dataclass, field

import pytest

from cineos.plugins import (
    Plugin,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginManager,
    PluginMetadata,
    PluginNotFoundError,
    PluginValidationError,
)


@dataclass
class RecordingPlugin(Plugin):
    metadata: PluginMetadata
    events: list[tuple[str, object]] = field(default_factory=list)

    def on_load(self, context: object) -> None:
        self.events.append(("load", context))

    def on_enable(self, context: object) -> None:
        self.events.append(("enable", context))

    def on_disable(self, context: object) -> None:
        self.events.append(("disable", context))

    def on_unload(self, context: object) -> None:
        self.events.append(("unload", context))


def make_plugin(name: str, dependencies: tuple[str, ...] = ()) -> RecordingPlugin:
    metadata = PluginMetadata(name, "1.2.3", "1.0.0", dependencies=dependencies)
    return RecordingPlugin(metadata)


def test_complete_lifecycle_is_idempotent_and_receives_context() -> None:
    context = object()
    plugin = make_plugin("notes")
    manager = PluginManager(context=context)
    assert manager.register(plugin) is plugin
    manager.enable("notes")
    manager.enable("notes")
    assert manager.is_loaded("notes")
    assert manager.is_enabled("notes")
    manager.disable("notes")
    manager.disable("notes")
    manager.unload("notes")
    manager.unload("notes")
    assert plugin.events == [
        ("load", context),
        ("enable", context),
        ("disable", context),
        ("unload", context),
    ]


def test_dependencies_transition_first_and_are_protected() -> None:
    dependency = make_plugin("base")
    dependent = make_plugin("feature", ("base",))
    manager = PluginManager()
    manager.register_all((dependent, dependency))
    manager.enable("feature")
    assert [plugin.metadata.name for plugin in manager.enabled_plugins] == [
        "base",
        "feature",
    ]
    with pytest.raises(PluginDependencyError, match="enabled dependents"):
        manager.disable("base")
    manager.disable("feature")
    with pytest.raises(PluginDependencyError, match="loaded dependents"):
        manager.unload("base")
    manager.unload("feature")
    manager.unload("base")


def test_missing_and_circular_dependencies_fail() -> None:
    manager = PluginManager()
    manager.register(make_plugin("missing-user", ("absent",)))
    with pytest.raises(PluginDependencyError, match="unregistered"):
        manager.load("missing-user")

    manager = PluginManager()
    manager.register_all((make_plugin("one", ("two",)), make_plugin("two", ("one",))))
    with pytest.raises(PluginDependencyError, match="circular"):
        manager.load("one")


def test_metadata_registration_and_lookup_validation() -> None:
    with pytest.raises(PluginValidationError):
        PluginMetadata("", "1.0.0", "1.0.0")
    with pytest.raises(PluginValidationError):
        PluginMetadata("bad", "one", "1.0.0")
    with pytest.raises(PluginCompatibilityError):
        PluginManager().register(
            RecordingPlugin(PluginMetadata("future", "1.0.0", "2.0.0"))
        )
    with pytest.raises(PluginNotFoundError):
        PluginManager().get("unknown")


def test_discovery_registers_entry_point_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = make_plugin("discovered")

    class Point:
        name = "discovered"

        @staticmethod
        def load() -> RecordingPlugin:
            return plugin

    class Points(list[Point]):
        def select(self, *, group: str) -> "Points":
            assert group == "cineos.plugins"
            return self

    monkeypatch.setattr(
        "cineos.plugins.manager.importlib_metadata.entry_points",
        lambda: Points([Point()]),
    )
    manager = PluginManager()
    assert manager.discover() == (plugin,)
    assert manager.plugins == (plugin,)
