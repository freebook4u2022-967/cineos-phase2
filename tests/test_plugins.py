from pathlib import Path

import pytest

from cineos.plugins import (
    Plugin,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginManager,
    PluginMetadata,
    PluginValidationError,
    discover_directory,
    version_satisfies,
)


class RecordingPlugin(Plugin):
    def __init__(self, metadata, events):
        self.metadata = metadata
        self.events = events

    def on_load(self, context):
        self.events.append(("load", context))

    def on_enable(self):
        self.events.append(("enable", None))

    def on_disable(self):
        self.events.append(("disable", None))

    def on_unload(self):
        self.events.append(("unload", None))


def make_plugin(name="sample", version="1.2.0", dependencies=None, events=None):
    return RecordingPlugin(
        PluginMetadata(name, version, dependencies=dependencies or {}),
        events if events is not None else [],
    )


def test_metadata_is_validated_and_dependencies_are_immutable():
    metadata = PluginMetadata("sample", "1.2.3", dependencies={"base": ">=1.0.0"})
    assert metadata.dependencies["base"] == ">=1.0.0"
    with pytest.raises(TypeError):
        metadata.dependencies["base"] = "==2.0.0"
    with pytest.raises(PluginValidationError, match="semantic version"):
        PluginMetadata("sample", "latest")


def test_version_constraints():
    assert version_satisfies("1.4.2", ">=1.2.0,<2.0.0")
    assert not version_satisfies("2.0.0", ">=1.2.0,<2.0.0")


def test_complete_lifecycle_and_idempotent_state_changes():
    events = []
    plugin = make_plugin(events=events)
    manager = PluginManager(context={"project": "demo"})
    assert manager.load(plugin) is plugin
    manager.enable("sample")
    manager.disable("sample")
    manager.disable("sample")
    assert manager.unload("sample") is plugin
    assert events == [
        ("load", {"project": "demo"}),
        ("enable", None),
        ("disable", None),
        ("unload", None),
    ]


def test_disabled_plugin_can_be_enabled_later():
    manager = PluginManager()
    manager.load(make_plugin(), enable=False)
    assert not manager.is_enabled("sample")
    manager.enable("sample")
    assert manager.is_enabled("sample")


def test_dependencies_are_loaded_in_order_and_protected():
    manager = PluginManager()
    base = make_plugin(name="base", version="1.5.0")
    dependent = make_plugin(dependencies={"base": ">=1.0.0,<2.0.0"})
    assert manager.load_many([dependent, base]) == (base, dependent)
    with pytest.raises(PluginDependencyError, match="enabled dependents"):
        manager.disable("base")
    with pytest.raises(PluginDependencyError, match="loaded dependents"):
        manager.unload("base")
    manager.unload_all()


def test_missing_and_incompatible_dependencies_are_rejected():
    manager = PluginManager()
    with pytest.raises(PluginDependencyError, match="missing"):
        manager.load(make_plugin(dependencies={"missing": ">=1.0.0"}))
    manager.load(make_plugin(name="base", version="2.0.0"))
    with pytest.raises(PluginDependencyError, match="found 2.0.0"):
        manager.load(make_plugin(dependencies={"base": "<2.0.0"}))


def test_incompatible_framework_api_is_rejected():
    plugin = make_plugin()
    plugin.metadata = PluginMetadata("sample", "1.0.0", api_version="2.0.0")
    with pytest.raises(PluginCompatibilityError, match="requires API"):
        PluginManager().load(plugin)


def test_directory_discovery_and_factory_loading(tmp_path: Path):
    (tmp_path / "example.py").write_text(
        """
from cineos.plugins import Plugin, PluginMetadata
class Example(Plugin):
    metadata = PluginMetadata("discovered", "1.0.0")
def create_plugin():
    return Example()
""",
        encoding="utf-8",
    )
    modules = discover_directory(tmp_path)
    assert list(modules) == ["example"]
    manager = PluginManager()
    manager.discover_and_load(str(tmp_path))
    assert manager.metadata("discovered").version == "1.0.0"
